"""Reference-free output support and same-clock 30/60-Hz agreement."""
import json
from pathlib import Path
import numpy as np
from .artifacts import write_json
from .signals import estimate_hr_welch
from .pilot6_delay_validation import timing_diagnostics


def correlation(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.isfinite(a).all() and np.isfinite(b).all() and min(np.std(a),np.std(b))>1e-8 else None


def describe_outputs(folder):
    folder=Path(folder);summary=json.loads((folder/'inference_summary.json').read_text());outputs=[];paired=[]
    with np.load(folder/'face_waveforms.npz',allow_pickle=False) as wave,np.load(folder/'features.npz',allow_pickle=False) as features,np.load(folder/'feature_validity.npz',allow_pickle=False) as flags:
        for key in wave.files:
            fs_text,model=key.split('Hz__',1);fs=int(fs_text);values=wave[key]
            rows=summary['rates'][str(fs)]['window_rows'];eligible=flags[f'{fs}Hz__candidate_cuff_free']&flags[f'{fs}Hz__face_quality_good']
            finite=np.isfinite(values).all(1);nonflat=finite&(np.nanstd(values,axis=1)>1e-8)
            hr=np.array([estimate_hr_welch(v,fs).hr_bpm if ok else np.nan for v,ok in zip(values,nonflat)])
            baseline=np.array([r['face_green_hr_bpm'] if r['face_green_hr_bpm'] is not None else np.nan for r in rows])
            use=eligible&nonflat&np.isfinite(hr)&np.isfinite(baseline)
            outputs.append(dict(model=model,sample_rate_hz=fs,attempted=len(values),finite=int(finite.sum()),nonflat=int(nonflat.sum()),
                candidate_quality_windows=int(eligible.sum()),hr_comparison_windows=int(use.sum()),
                median_abs_hr_difference_from_GREEN_bpm=float(np.median(abs(hr[use]-baseline[use]))) if use.any() else None,
                hr_difference_is_not_reference_error=True))
            if fs==30:
                high=wave['60Hz__'+model];same=high[:,::2]
                if same.shape!=values.shape:raise ValueError('30/60 grids lost physical correspondence')
                for i,(a,b) in enumerate(zip(values,same)):
                    row=dict(model=model,window_index=i,start_sec=rows[i]['start_sec'],candidate_quality_window=bool(eligible[i]),
                        same_clock_correlation=correlation(a,b),scope='30_vs_actual_native60_processing_agreement')
                    if row['same_clock_correlation'] is not None:
                        d=timing_diagnostics(a,b,30,baseline[i]/60,.3,landmarks=False)
                        row['relative_xcorr_lag_sec']=d['delay_xcorr_sec'];row['relative_xcorr_score']=d['delay_xcorr_score']
                    paired.append(row)
        feature_support=[]
        for key in features.files:
            if key.endswith(('__start','__end')):continue
            values=features[key];valid=flags[key]&np.isfinite(values)
            feature_support.append(dict(model_rate=key,attempted_windows=len(values),valid_windows_per_feature=valid.sum(0),
                output_std_per_feature=[float(np.std(values[valid[:,i],i])) if valid[:,i].any() else None for i in range(10)]))
    result=dict(waveforms=outputs,rate_agreement=paired,feature_support=feature_support,
        contact_waveform_reference_available=False,agreement_is_not_physiological_validation=True)
    write_json(folder/'reference_free_diagnostics.json',result)
    return result
