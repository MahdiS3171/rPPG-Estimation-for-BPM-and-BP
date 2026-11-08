"""Controlled synthetic-only falsification of spatial weighting; no SCAMPS labels."""
import numpy as np

from .waveform_v2_micro_rois import frozen_patch_mix, weight_change_diagnostics
from .waveform_v2_timing_contract import audit_synthetic_timing_transfer, synthetic_pulse


def timing_cases(fs=30.):
    if fs not in (30.,60.):
        raise ValueError('Predeclared synthetic rates are 30 and 60 Hz')
    n=int(20*fs)
    t=np.arange(n)/fs
    f,h=synthetic_pulse(t),synthetic_pulse(t,delay_ms=45.)
    cases=[]
    block=int(10*fs)
    def mix(patches,weights):
        return frozen_patch_mix(patches[None],np.isfinite(patches).all(-1)[None],weights[None],block)[0][0,:,1]
    def audit(name,face,hand,source_face=f,source_hand=h,**metadata):
        result=audit_synthetic_timing_transfer(t,source_face=source_face,source_hand=source_hand,
            reconstructed_face=face,reconstructed_hand=hand,known_source_delay_ms=45.)
        row=dict(result,case=name,fs=fs)
        row.update(metadata)
        cases.append(row)
    weights=np.tile([2/3,1/3],(2,1))
    pface=np.repeat(np.stack((f,f))[:,:,None],3,axis=-1)
    phand=np.repeat(np.stack((h,h))[:,:,None],3,axis=-1)
    audit('identical_phase_frozen_weights',mix(pface,weights),mix(phand,weights),
          scope='synthetic_only',weight_ratio_bound=2.)
    # Symmetric phase offsets cancel for an equal-weight sinusoidal control;
    # bounded unequal weights alone can introduce a common absolute phase shift.
    omega=2*np.pi*1.2
    sf,sh=np.sin(omega*t),np.sin(omega*(t-.045))
    pf=np.repeat(np.stack((np.sin(omega*(t+.12)),np.sin(omega*(t-.12))))[:,:,None],3,axis=-1)
    ph=np.repeat(np.stack((np.sin(omega*(t-.045+.12)),np.sin(omega*(t-.045-.12))))[:,:,None],3,axis=-1)
    audit('symmetric_phase_uniform_control',mix(pf,np.full((2,2),.5)),mix(ph,np.full((2,2),.5)),sf,sh,
          scope='synthetic_only',signal='sinusoidal_positive_control')
    audit('heterogeneous_phase_bounded_unequal_weights',mix(pf,weights),mix(ph,weights),sf,sh,
          scope='synthetic_only',weight_ratio_bound=2.)
    offsets=np.array([100.,140.])[:,None,None]
    pf=pface+offsets; ph=phand+offsets
    dynamic=np.zeros((1,2,n))
    chosen=(np.arange(n)//max(1,int(.5*fs)))%2
    dynamic[0,0]=np.where(chosen==0,2/3,1/3);dynamic[0,1]=1-dynamic[0,0]
    df=(dynamic[0,:,:,None]*pf).sum(0)[:,1]
    dh=(dynamic[0,:,:,None]*ph).sum(0)[:,1]
    diagnostics=weight_change_diagnostics(pf[None],dynamic)
    audit('unsafe_rapid_switching_negative_control',df,dh,
          scope='synthetic_only_not_admitted_candidate',
          maximum_weight_induced_rgb_step=float(np.nanmax(np.abs(diagnostics['weight_change_rgb_term']))))
    boundary_weights=np.array([[2/3,1/3],[1/3,2/3]])
    bf,bv,bw=frozen_patch_mix(pf[None],np.ones((1,2,n),dtype=bool),boundary_weights[None],block)
    bh=mix(ph,boundary_weights)
    diagnostics=weight_change_diagnostics(pf[None],bw)
    audit('block_boundary_weight_change',bf[0,:,1],bh,
          scope='synthetic_only',boundary_index=block,
          maximum_weight_induced_rgb_step=float(np.nanmax(np.abs(diagnostics['weight_change_rgb_term']))))
    lost=pface.copy();lost[0,int(5*fs)]=np.nan
    audit('missing_weighted_patch_no_rescue',mix(lost,weights),mix(phand,weights),
          scope='synthetic_only',missing_source_index=int(5*fs))
    return cases
