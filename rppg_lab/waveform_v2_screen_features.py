"""Label-free window descriptors for the bounded adaptive-waveform screen.

Periodicity/illumination can mimic these features: they are not confidence labels.
"""
import numpy as np


def window_descriptors(x,fs=30.0):
    x=np.asarray(x,dtype=float)
    if x.ndim!=3 or x.shape[:2]!=(3,9) or not np.isfinite(x).all():
        raise ValueError('Require finite [3,9,T] video-derived input')
    priors=x[:,3:,:].reshape(18,-1)
    priors=priors-priors.mean(-1,keepdims=True)
    power=abs(np.fft.rfft(priors,axis=-1))**2
    frequency=np.fft.rfftfreq(x.shape[-1],1/fs)
    band=(frequency>=.7)&(frequency<=3.5)
    peaks=frequency[band][np.argmax(power[:,band],axis=-1)]
    concentrated=np.sum(power*(abs(frequency[None]-peaks[:,None])<=.15),axis=-1)
    denominator=np.sum(power[:,(frequency>=.5)&(frequency<=8)],axis=-1)+1e-12
    harmonics=np.sum(power*(abs(frequency[None]-2*peaks[:,None])<=.15),axis=-1)/denominator
    lag=max(1,round(.1*fs))
    a,b=priors[:,:-lag],priors[:,lag:]
    autocorr=np.sum(a*b,axis=-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)+1e-12)
    scaled=priors/(priors.std(-1,keepdims=True)+1e-8)
    skew=np.clip(np.mean(scaled**3,axis=-1),-3,3)/3
    features=np.stack((concentrated/denominator,harmonics,autocorr,skew),axis=-1).reshape(-1)
    motion=np.mean(abs(np.diff(x[:,:3,:],axis=-1)),axis=(1,2))*fs/30
    return np.r_[features,np.clip(motion,0,3)/3].astype('float32')
