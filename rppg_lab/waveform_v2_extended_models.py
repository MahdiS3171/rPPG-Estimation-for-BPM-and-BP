"""Small long-context, attention, spectral and temporal-gating hypotheses."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .waveform_v2_screen_models import AdaptiveWaveformNet


class LongTemporalEncoder(nn.Module):
    def __init__(self,channels,width):
        super().__init__();self.input=nn.Conv1d(channels,width,1)
        self.layers=nn.ModuleList([nn.Conv1d(width,width,5) for _ in range(4)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
    def forward(self,x,fs):
        z=self.input(x);rate=round(fs/30)
        for layer,norm,dilation in zip(self.layers,self.norms,(1,2,4,8)):
            d=rate*dilation
            update=F.conv1d(z,layer.weight,layer.bias,padding=2*d,dilation=d)
            z=z+.3*F.silu(norm(update.transpose(1,2)).transpose(1,2))
        return z


class PhysicalAttentionEncoder(nn.Module):
    def __init__(self,channels,width):
        super().__init__();self.local=LongTemporalEncoder(channels,width)
        self.qkv=nn.Conv1d(width,3*width,1);self.output=nn.Conv1d(width,width,1)
        self.norm=nn.LayerNorm(width);self.width=width;self.heads=2
        self.register_buffer('scales',torch.tensor([.2,1.0]))
    def forward(self,x,fs):
        z=self.local(x,fs);batch,width,length=z.shape
        q,k,v=self.qkv(z).reshape(batch,3,self.heads,width//self.heads,length).unbind(1)
        q,k,v=(t.transpose(-1,-2) for t in (q,k,v))
        times=torch.arange(length,device=x.device,dtype=x.dtype)/fs
        bias=-abs(times[:,None]-times[None,:])[None,None]/self.scales[None,:,None,None]
        probability=torch.softmax(q@k.transpose(-1,-2)/math.sqrt(width//self.heads)+bias,dim=-1)
        attended=(probability@v).transpose(-1,-2).reshape(batch,width,length)
        return z+.2*self.norm(self.output(attended).transpose(1,2)).transpose(1,2)


class SpectralLayer(nn.Module):
    def __init__(self,channels,width,modes=81):
        super().__init__();self.real=nn.Parameter(torch.randn(channels,width,modes)*.02)
        self.imag=nn.Parameter(torch.randn(channels,width,modes)*.02)
        self.local=nn.Conv1d(channels,width,1);self.modes=modes
    def forward(self,x):
        spectrum=torch.fft.rfft(x,norm='forward');count=min(self.modes,spectrum.shape[-1])
        output=spectrum.new_zeros((len(x),self.real.shape[1],spectrum.shape[-1]))
        weights=torch.complex(self.real[:,:,:count],self.imag[:,:,:count])
        output[:,:,:count]=torch.einsum('bim,iom->bom',spectrum[:,:,:count],weights)
        return self.local(x)+torch.fft.irfft(output,n=x.shape[-1],norm='forward')


class SpectralEncoder(nn.Module):
    def __init__(self,channels,width):
        super().__init__();self.first=SpectralLayer(channels,width);self.second=SpectralLayer(width,width)
    def forward(self,x,fs):
        if x.shape[-1]!=10*fs:raise ValueError('Physical modes require the declared ten-second window')
        z=F.silu(self.first(x));return z+.3*F.silu(self.second(z))


class BoundedTemporalGate(AdaptiveWaveformNet):
    def __init__(self,width=16):
        super().__init__('gate_residual',width=width)
        self.temporal_logits=nn.Conv1d(width,36,1)
        nn.init.zeros_(self.temporal_logits.weight);nn.init.zeros_(self.temporal_logits.bias)
    def forward(self,x,roi_valid,prior_valid,descriptors,context,context_valid,fs=30):
        out=super().forward(x,roi_valid,prior_valid,descriptors,context,context_valid,fs)
        mask=torch.cat((roi_valid[...,None].expand(-1,-1,3),prior_valid&roi_valid[...,None]),2)
        safe=torch.nan_to_num(torch.where(mask[...,None],x,torch.zeros_like(x)))
        hidden=self.encoder(safe.reshape(len(x),27,-1),fs)
        radius=8*round(fs/30)
        modulation=.5*torch.tanh(F.avg_pool1d(self.temporal_logits(hidden),2*radius+1,stride=1,padding=radius))
        logits=self.gate(descriptors)[:,:,None]+modulation
        valid=(prior_valid&roi_valid[...,None]).reshape(len(x),18)
        weights=torch.softmax(logits.masked_fill(~valid.repeat(1,2)[:,:,None],-1e4),dim=1)
        signed=weights[:,:18]-weights[:,18:]
        base=(signed*safe[:,:,3:,:].reshape(len(x),18,-1)).sum(1)
        out['ppg']=(base+.1*out['residual'])*out['usable'][:,None]
        out['temporal_gate_variation']=(torch.diff(signed,dim=-1)*fs).square().mean()
        return out


def make_extended_model(architecture,width=16):
    if architecture=='temporal_gate':return BoundedTemporalGate(width)
    model=AdaptiveWaveformNet('gate_residual',width=width)
    encoders={'long_tcn':LongTemporalEncoder,'physical_attention':PhysicalAttentionEncoder,'spectral':SpectralEncoder}
    model.encoder=encoders[architecture](27,width)
    return model
