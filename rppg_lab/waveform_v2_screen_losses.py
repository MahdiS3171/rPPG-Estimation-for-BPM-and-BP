"""Explicit competing loss hypotheses for contact-target reconstruction."""
import torch


def standardized(x):
    x=x-x.mean(-1,keepdim=True)
    return x/torch.sqrt(x.square().mean(-1,keepdim=True)+1e-6)


def pearson(x,y):
    x=x-x.mean(-1,keepdim=True);y=y-y.mean(-1,keepdim=True)
    return (x*y).sum(-1)/torch.sqrt(x.square().sum(-1)*y.square().sum(-1)+1e-8)


def aligned_components(p,y,fs=30):
    radius=round(.5*fs);length=p.shape[-1]-2*radius
    windows=p.unfold(-1,length,1)
    target=y[:,None,radius:-radius]
    correlations=pearson(windows,target)
    best,indices=correlations.max(-1)
    derivatives=pearson(torch.diff(windows,dim=-1),torch.diff(target,dim=-1))
    d1=derivatives.gather(1,indices[:,None]).squeeze(1)
    return best,d1


def spectral_distance(p,y,fs=30):
    frequency=torch.fft.rfftfreq(p.shape[-1],1/fs,device=p.device)
    band=(frequency>=.7)&(frequency<=8)
    a=torch.fft.rfft(p).abs().square()[:,band]+1e-6
    b=torch.fft.rfft(y).abs().square()[:,band]+1e-6
    return ((a/a.sum(-1,keepdim=True)).sqrt()-(b/b.sum(-1,keepdim=True)).sqrt()).abs().mean()


def training_loss(out,target,kind,step,total_steps,fs=30):
    p=standardized(out['ppg']);y=standardized(target)
    native=1-pearson(p,y).mean()
    derivative=1-pearson(torch.diff(p,dim=-1),torch.diff(y,dim=-1)).mean()
    spectrum=spectral_distance(p,y,fs)
    if kind=='native':loss=native+.1*(p-y).square().mean()
    elif kind=='morphology':
        corr,d1=aligned_components(p,y,fs);loss=1-corr.mean()+.2*(1-d1.mean())+.1*spectrum
    elif kind=='derivative_spectral':loss=native+.3*derivative+.3*spectrum
    elif kind=='staged':
        corr,d1=aligned_components(p,y,fs)
        progress=step/max(total_steps-1,1)
        native_weight=max(0,min(1,(progress-.33)/.67))
        loss=(1-native_weight)*(1-corr.mean()+.2*(1-d1.mean()))+native_weight*(native+.3*derivative)+.3*spectrum
    else:raise ValueError('Unknown loss hypothesis')
    if 'coarse' in out:
        frequency=torch.fft.rfftfreq(y.shape[-1],1/fs,device=y.device)
        low=torch.fft.irfft(torch.fft.rfft(y)*(frequency<=3.5),n=y.shape[-1])
        loss=loss+.1*(out['coarse']-low).square().mean()+.1*(out['fine']-(y-low)).square().mean()
    if 'residual' in out:loss=loss+1e-4*out['residual'].square().mean()
    return loss
