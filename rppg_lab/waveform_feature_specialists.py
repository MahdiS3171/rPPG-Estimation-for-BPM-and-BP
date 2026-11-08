"""Three task groups with joint, shared-head and independent feature estimators.

Targets are computed contact-waveform summaries, not verified facial fiducials
or blood-pressure labels. Outputs are a feature vector, not a reconstructed wave.
"""
import numpy as np
from scipy.signal import butter,sosfiltfilt,find_peaks,welch
import torch
from torch import nn
from .waveform_v2_extended_models import LongTemporalEncoder

GROUPS={'rhythm':slice(0,3),'geometry':slice(3,7),'derivatives':slice(7,10)}
FEATURES=('hr_bpm','median_ibi_s','ibi_cv','rise_fraction','width50_fraction','width75_fraction','normalized_area',
    'max_up_slope_per_s','max_down_slope_per_s','curvature_rms_per_s2')
UNITS=('bpm','seconds','ratio','ratio','ratio','ratio','ratio','1/seconds','1/seconds','1/seconds^2')
STD_FLOORS=np.asarray([1.,.01,.01,.01,.01,.01,.01,.5,.5,5.])


def pulse_features(wave,fs=30):
    value=np.asarray(wave,float);features=np.zeros(10);valid=np.zeros(10,dtype=bool)
    if value.ndim!=1 or len(value)!=10*fs or not np.isfinite(value).all() or np.std(value)<1e-8:return features,valid
    filtered=sosfiltfilt(butter(3,[.5,8.],btype='bandpass',fs=fs,output='sos'),value)
    frequency,power=welch(filtered,fs,nperseg=len(filtered),nfft=4096)
    band=(frequency>=.7)&(frequency<=3.5)
    f0=float(frequency[band][np.argmax(power[band])]);features[0]=60*f0;valid[0]=True
    peaks,_=find_peaks(filtered,distance=max(1,round(.55*fs/f0)),prominence=.5*np.std(filtered))
    peaks=peaks[(peaks>=round(.5*fs))&(peaks<len(filtered)-round(.5*fs))]
    if len(peaks)>=4:
        intervals=np.diff(peaks)/fs;features[1]=np.median(intervals);features[2]=np.std(intervals)/(np.mean(intervals)+1e-8);valid[1:3]=True
    feet=[int(a+np.argmin(filtered[a:b+1])) for a,b in zip(peaks[:-1],peaks[1:])]
    pulses=[]
    for left,right in zip(feet[:-1],feet[1:]):
        duration=(right-left)/fs
        if not .5/f0<=duration<=1.8/f0 or right-left<5:continue
        beat=filtered[left:right+1].copy();beat-=np.linspace(beat[0],beat[-1],len(beat))
        amplitude=beat.max();peak=int(beat.argmax())
        if amplitude<.2*np.std(filtered) or not 1<=peak<len(beat)-1:continue
        beat=np.maximum(beat/amplitude,0);rise=peak/(len(beat)-1);widths=[]
        for level in (.5,.75):
            a=peak
            while a>0 and beat[a-1]>=level:a-=1
            b=peak
            while b<len(beat)-1 and beat[b+1]>=level:b+=1
            lo=a if a==0 else (a-1)+(level-beat[a-1])/(beat[a]-beat[a-1]+1e-12)
            hi=b if b==len(beat)-1 else b+(beat[b]-level)/(beat[b]-beat[b+1]+1e-12)
            widths.append((hi-lo)/(len(beat)-1))
        d1=np.gradient(beat,1/fs);d2=np.gradient(d1,1/fs)
        area=float(np.sum((beat[1:]+beat[:-1])*.5)/(len(beat)-1))
        pulses.append([rise,*widths,area,float(d1.max()),float(-d1.min()),float(np.sqrt(np.mean(d2*d2)))])
    if len(pulses)>=3:
        features[3:]=np.median(pulses,axis=0);valid[3:]=True
    valid&=np.isfinite(features);features[~valid]=0
    return features,valid


def targets(waves,fs=30):
    rows=[pulse_features(w,fs) for w in waves]
    return np.stack([r[0] for r in rows]),np.stack([r[1] for r in rows])


class FeatureEncoder(nn.Module):
    def __init__(self,width):
        super().__init__();self.temporal=LongTemporalEncoder(27,width);self.output_size=2*width+151
    def forward(self,x,roi,prior,descriptors,fs=30):
        if fs not in (30,60) or x.shape[1:]!=(3,9,10*fs):raise ValueError('Require declared 10-second 30/60-Hz features')
        mask=torch.cat((roi[...,None].expand(-1,-1,3),prior&roi[...,None]),dim=2)
        safe=torch.nan_to_num(torch.where(mask[...,None],x,torch.zeros_like(x)))
        hidden=self.temporal(safe.reshape(len(x),27,-1),fs)
        spectrum=torch.fft.rfft(safe[:,:,3:,:].reshape(len(x),18,-1)).abs().square().mean(1)
        frequency=torch.fft.rfftfreq(x.shape[-1],1/fs,device=x.device);band=(frequency>=.5)&(frequency<=8)
        spectrum=spectrum[:,band];spectrum=spectrum/spectrum.sum(-1,keepdim=True).clamp_min(1e-8)
        vector=torch.cat((hidden.mean(-1),hidden.std(-1,correction=0),torch.nan_to_num(descriptors),spectrum),dim=1)
        usable=roi.any(1)&(safe[:,:,:3].std(-1).amax((1,2))>1e-7)
        return vector,usable


def head(size,outputs):return nn.Sequential(nn.Linear(size,32),nn.SiLU(),nn.Linear(32,outputs))


class FeatureExpert(nn.Module):
    def __init__(self,width,outputs):
        super().__init__();self.encoder=FeatureEncoder(width);self.head=head(self.encoder.output_size,outputs)
    def forward(self,x,roi,prior,descriptors,fs=30):
        vector,usable=self.encoder(x,roi,prior,descriptors,fs)
        return self.head(vector)*usable[:,None],usable


class ThreePartFeatureModel(nn.Module):
    def __init__(self,mode,width=16):
        super().__init__();self.mode=mode;self.width=width
        if mode=='independent':self.experts=nn.ModuleList([FeatureExpert(width,s.stop-s.start) for s in GROUPS.values()])
        elif mode in ('joint','shared_heads'):
            self.encoder=FeatureEncoder(width)
            if mode=='joint':self.output=head(self.encoder.output_size,10)
            else:self.heads=nn.ModuleList([head(self.encoder.output_size,s.stop-s.start) for s in GROUPS.values()])
        else:raise ValueError('Unknown feature model mode')
    def forward(self,x,roi,prior,descriptors,fs=30):
        if self.mode=='independent':
            parts=[expert(x,roi,prior,descriptors,fs) for expert in self.experts]
            return dict(features=torch.cat([p for p,_ in parts],1),usable=parts[0][1])
        vector,usable=self.encoder(x,roi,prior,descriptors,fs)
        output=self.output(vector) if self.mode=='joint' else torch.cat([h(vector) for h in self.heads],1)
        return dict(features=output*usable[:,None],usable=usable)


def matched_widths():
    count=lambda m:sum(p.numel() for p in m.parameters())
    target=count(ThreePartFeatureModel('independent',16));result={'independent':16}
    for mode in ('joint','shared_heads'):
        result[mode]=min(range(8,65),key=lambda width:abs(count(ThreePartFeatureModel(mode,width))-target))
    return result


def feature_loss(prediction,target,valid):
    losses=[]
    for name,section in GROUPS.items():
        error=prediction[:,section]-target[:,section];mask=valid[:,section]
        if name=='geometry':element=error.square()
        else:
            magnitude=error.abs();element=torch.where(magnitude<=1,.5*error.square(),magnitude-.5)
        losses.append((element*mask).sum()/mask.sum().clamp_min(1))
    return sum(losses)/len(losses)
