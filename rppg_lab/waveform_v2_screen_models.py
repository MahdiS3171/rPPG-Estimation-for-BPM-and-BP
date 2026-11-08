"""Compact adaptive waveform candidates; separate from frozen V1.

Temporal kernels use the same physical support at 30/60 Hz by changing dilation.
Native-60 operation is not evidence of learned high-frequency physiological truth.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TemporalEncoder(nn.Module):
    def __init__(self,channels,width=16):
        super().__init__()
        self.first=nn.Conv1d(channels,width,9)
        self.second=nn.Conv1d(width,width,5)
    def forward(self,x,fs):
        dilation=round(fs/30)
        if fs not in (30,60):raise ValueError('Screen supports audited 30/60-Hz grids only')
        z=F.silu(F.conv1d(x,self.first.weight,self.first.bias,padding=4*dilation,dilation=dilation))
        return z+.5*F.silu(F.conv1d(z,self.second.weight,self.second.bias,padding=2*dilation,dilation=dilation))


class AdaptiveWaveformNet(nn.Module):
    def __init__(self,kind='gate_residual',context=False,width=16):
        super().__init__()
        if kind not in ('gate','gate_residual','raw_temporal','dual_band'):raise ValueError('Unknown candidate')
        self.kind=kind;self.use_context=context
        if kind!='raw_temporal':
            self.gate=nn.Sequential(nn.Linear(75,32),nn.SiLU(),nn.Linear(32,36))
            nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
            with torch.no_grad():self.gate[-1].bias[[2,8,14]]=3.0
        if kind!='gate':
            self.encoder=TemporalEncoder(9 if kind=='raw_temporal' else 27,width)
            self.output=nn.Conv1d(width,1,1)
            nn.init.normal_(self.output.weight,std=.01);nn.init.zeros_(self.output.bias)
        if kind=='dual_band':
            self.fine=nn.Conv1d(width,1,1)
            nn.init.normal_(self.fine.weight,std=.01);nn.init.zeros_(self.fine.bias)
        # Allocate after shared modules: adding zero-initialized context preserves
        # the common initialization under the same random seed.
        if context:
            self.context_gate=nn.Linear(75,36)
            nn.init.zeros_(self.context_gate.weight);nn.init.zeros_(self.context_gate.bias)

    def forward(self,x,roi_valid,prior_valid,descriptors,context,context_valid,fs=30):
        if x.ndim!=4 or tuple(x.shape[1:3])!=(3,9) or x.shape[-1]!=10*fs:
            raise ValueError('Require [B,3,9,10*fs] input on declared grid')
        mask=torch.cat((roi_valid[...,None].expand(-1,-1,3),prior_valid&roi_valid[...,None]),dim=2)
        x=torch.where(mask[...,None],x,torch.zeros_like(x))
        x=torch.nan_to_num(x)
        batch=len(x);weights=x.new_zeros((batch,36));base=x.new_zeros((batch,x.shape[-1]))
        if self.kind!='raw_temporal':
            logits=self.gate(torch.nan_to_num(descriptors))
            if self.use_context:
                logits=logits+.5*torch.tanh(self.context_gate(torch.nan_to_num(context)))*context_valid[:,None]
            valid=(prior_valid&roi_valid[...,None]).reshape(batch,18)
            logits=logits.masked_fill(~valid.repeat(1,2),-1e4)
            weights=torch.softmax(logits,dim=-1)
            signed=weights[:,:18]-weights[:,18:]
            base=(signed[...,None]*x[:,:,3:,:].reshape(batch,18,-1)).sum(1)
            usable=valid.any(1)
        else:usable=roi_valid.any(1)
        out=dict(weights=weights)
        if self.kind=='gate':wave=base
        else:
            inputs=x[:,:,:3,:].reshape(batch,9,-1) if self.kind=='raw_temporal' else x.reshape(batch,27,-1)
            hidden=self.encoder(inputs,fs)
            residual=self.output(hidden).squeeze(1)
            if self.kind=='raw_temporal':wave=residual
            elif self.kind=='dual_band':
                coarse=base+.1*residual;fine=.1*self.fine(hidden).squeeze(1)
                out.update(coarse=coarse,fine=fine);wave=coarse+fine
            else:wave=base+.1*residual
            out['residual']=residual
        usable=usable&(x[:,:,:3,:].std(-1).amax((1,2))>1e-7)
        out.update(ppg=wave*usable[:,None],usable=usable)
        return out
