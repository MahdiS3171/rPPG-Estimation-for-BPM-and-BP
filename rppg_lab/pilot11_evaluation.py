"""Label-free orchestration and processing-sensitivity diagnostics for Pilot11."""
import json
from pathlib import Path
import numpy as np
from .artifacts import file_sha256,write_json
from .pilot11_inference import without_pressure_values,infer_recording,extraction_config
from .pilot6_protocol_timeline import cuff_events
from .pilot6_delay_feasibility import analyze_recording,primary_rows,stability_summary
from .pilot6_delay_validation import timing_diagnostics
from .pilot11_waveform_diagnostics import describe_outputs
from .waveform_visualization import write_csv


def learned_timing(inference,timing,out):
    """Facial model versus classical palm; a sensitivity, never a PTT claim."""
    rows=[]
    summary=json.loads((inference/'inference_summary.json').read_text())
    with np.load(inference/'face_waveforms.npz',allow_pickle=False) as neural,np.load(timing/'classical_waveforms.npz',allow_pickle=False) as classic:
        t=classic['uniform_pts'];palm=classic['B_hand_palm_GREEN']
        for key in neural.files:
            fs_text,name=key.split('Hz__',1);fs=int(fs_text);waveforms=neural[key]
            for i,(face,window) in enumerate(zip(waveforms,summary['rates'][str(fs)]['window_rows'])):
                left=window['start_sec'];grid=left+np.arange(10*fs)/fs
                hand=np.interp(grid,t,palm,left=np.nan,right=np.nan)
                row=dict(model=name,sample_rate_hz=fs,window_index=i,start_sec=left,end_sec=left+10,
                    scope='learned_face_classical_palm_processing_sensitivity',physical_timing_calibrated=False,computable=False)
                if np.isfinite(face).all() and np.isfinite(hand).all() and np.std(face)>1e-8 and np.std(hand)>1e-8:
                    hr=window['face_green_hr_bpm']/60
                    if hr is not None and np.isfinite(hr) and hr>0:
                        diagnostic=timing_diagnostics(face,hand,fs,hr,.3,landmarks=False)
                        row.update(diagnostic);row['computable']=bool(np.isfinite(diagnostic['delay_xcorr_sec']))
                rows.append(row)
    write_csv(out/'learned_face_timing_sensitivity.csv',rows)
    return dict(attempted=len(rows),computable=sum(r['computable'] for r in rows),
        physical_timing_calibrated=False,validated_hand_neural_model=False)


def evaluate_session(native_path,session,models,device,config,out,scope_path):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);native_path=Path(native_path)
    sources=('pilot11_evaluation','pilot11_inference','pilot11_waveform_diagnostics','pilot6_delay_feasibility','pilot6_delay_validation',
        'pilot6_protocol_timeline','frozen_timebase','processing','quality','pilot_analysis','waveform_feature_specialists',
        'waveform_v2_screen_models','waveform_v2_extended_models','waveform_v2_screen_features','window_priors','signals','classical')
    clean=without_pressure_values(session)
    identity=dict(native_sha256=file_sha256(native_path),scope_sha256=file_sha256(scope_path),
        session=clean,config=config,source_sha256={n:file_sha256(Path(__file__).with_name(n+'.py')) for n in sources})
    receipt=out/'evaluation_complete.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['identity']!=identity:raise ValueError('Completed evaluation identity differs; use a new namespace')
        for name,digest in saved['artifact_sha256'].items():
            if file_sha256(out/name)!=digest:raise ValueError('Completed evaluation artifact changed')
        return saved
    with np.load(native_path,allow_pickle=False) as archive:
        native={k:archive[k].copy() for k in archive.files}
    acquisition=json.loads((native_path.parent/'acquisition.json').read_text())
    recorded=json.loads(str(native['identity_json'].item()))
    if recorded!=acquisition['identity'] or recorded['video_sha256']!=file_sha256(session['video']):raise ValueError('Acquisition/video identity mismatch')
    if recorded['version']!='pilot11_native_every_frame_single_decode_v1' or not recorded['every_frame_detection'] or recorded['spatial_resize']:raise ValueError('Consistent every-frame acquisition required')
    for name,digest in recorded['source_sha256'].items():
        if file_sha256(Path(__file__).with_name(name+'.py'))!=digest:raise ValueError('Acquisition source changed')
    clean['events']=cuff_events(clean,float(native['timestamps'][-1]));clean['manual_excluded']=bool(session.get('manual_excluded',False))
    timing=out/'timing';timing.mkdir(exist_ok=True)
    q,d,a,c=analyze_recording(native,clean,config,timing)
    primary=primary_rows(d)
    independent=[r for r in primary if r['primary_nonoverlapping'] and r['cuff_phase']=='candidate_cuff_free']
    label_free=dict(primary_independent_attempts=len(independent),primary_independent_computable=sum(r['computable'] for r in independent),
        primary_independent_strict=sum(r['status']=='strict' for r in independent),
        strict_at_least_two=sum(r['status']=='strict' for r in independent)>=2,
        timing_verified_attempts=sum(r['timing_verified'] for r in independent),
        physical_timing_calibrated=False,sampling_step_is_not_physical_error_bound=True,
        primary_stability=stability_summary(primary),acquisition=a,camera=c)
    write_json(timing/'label_free_summary.json',label_free)
    inference=out/'inference';face=infer_recording(native,clean,models,device,inference,extraction_config())
    describe_outputs(inference)
    sensitivity=learned_timing(inference,timing,out)
    artifacts={str(p.relative_to(out)):file_sha256(p) for p in out.rglob('*') if p.is_file() and p!=receipt}
    result=dict(identity=identity,session_id=session['session_id'],subject_id=session['subject_id'],timing=label_free,
        face_rates={k:{n:v for n,v in r.items() if n not in ('window_rows','adapter')} for k,r in face['rates'].items()},
        learned_timing=sensitivity,artifact_sha256=artifacts)
    write_json(receipt,result)
    return result
