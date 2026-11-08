"""Frozen facial inference and label-free features for the authorized pilot."""
import json
from pathlib import Path
import re
import numpy as np
import torch
from .artifacts import file_sha256,write_json
from .config import PipelineConfig
from .extraction_cache import FaceROIExtraction
from .frozen_timebase import adapt_native_face,TimebaseConfig,antialias_fir,bounded_regularize
from .processing import shared_grid,resample_trace
from .types import RGBTrace,SharedTimebase
from .window_priors import build_window_priors
from .signals import standardize_channels,estimate_hr_welch
from .quality import signal_quality
from .waveform_v2_screen_features import window_descriptors
from .waveform_v2_screen_models import AdaptiveWaveformNet
from .waveform_v2_extended_models import make_extended_model
from .waveform_feature_specialists import ThreePartFeatureModel,targets,FEATURES
from .pilot6_protocol_timeline import cuff_events,timeline_masks

FACE_NAMES=('forehead','left_cheek','right_cheek')


def without_pressure_values(session):
    result=dict(session);result['raw']={k:v for k,v in session['raw'].items() if not re.match(r'^(SBP|DBP|CuffHR)[123]_',k)}
    result['subject_raw']={}  # Demographics/geometry/identity are not predictors.
    return result


def extraction_config():
    manifest=Path('checkpoints/waveform_v1_official_ubfc/waveform_v1_freeze_manifest.json')
    if file_sha256(manifest)!='da5ba397b3be389deec8821f8ea16944b5f5b93e2834bb79c3c147eb3d1f89dc':raise ValueError('Frozen manifest changed')
    values=json.loads(manifest.read_text())['extraction_config']
    for name in ('regions','skin_cr_range','skin_cb_range','face_rois'):values[name]=tuple(values[name])
    return PipelineConfig(**values)


def prepared_windows(native,fs,config):
    t=np.asarray(native['timestamps'],float);traces={}
    for name in FACE_NAMES:
        x=native[name+'_rgb'];valid=np.isfinite(x).all(1)&native[name+'_extraction_valid'].astype(bool)
        quality={k[len(name)+1:]:v for k,v in native.items() if k.startswith(name+'_') and v.shape==t.shape}
        traces[name]=RGBTrace(t,x,name,valid,quality)
    grid=shared_grid(t,fs)
    if fs==30:
        original=FaceROIExtraction(SharedTimebase(shared_grid(t,30),t),traces)
        adapted,diagnostics=adapt_native_face(original,config,TimebaseConfig())
        rgb={name:resample_trace(adapted.traces[name],grid,config.max_gap_sec)[0] for name in FACE_NAMES}
    elif fs==60:
        rgb={name:bounded_regularize(trace,grid,config.max_gap_sec,True)[0] for name,trace in traces.items()}
        diagnostics=dict(operation='native60_bounded_cubic',new_physiological_supervision=False)
    else:raise ValueError('Only predeclared 30/60-Hz inputs')
    # Both rates start at the same 30-Hz-aligned full-FIR-support instant,
    # independent of BP values, model performance or per-ROI quality.
    half=(len(antialias_fir())-1)/(2*60);offset=np.ceil(half*30)/30
    first=int(round(offset*fs));length=10*fs;starts=range(first,len(grid)-length+1,length)
    windows=[];rows=[]
    for start in starts:
        end=start+length;left=float(grid[start]);right=left+10
        x=np.zeros((3,9,length),np.float32);roi=np.zeros(3,bool);prior=np.zeros((3,6),bool)
        for r,name in enumerate(FACE_NAMES):
            source=(t>=left)&(t<right);coverage=float(traces[name].validity_mask[source].mean()) if source.any() else 0.
            piece=rgb[name][start:end]
            if coverage<.9 or not np.isfinite(piece).all():continue
            values,success=build_window_priors(piece,fs,('GREEN','CHROM','CHROM_WIN','PBV','POS_WIN','OMIT'))
            if success.any():x[r,:3]=standardize_channels(piece).T;x[r,3:]=values;roi[r]=True;prior[r]=success
        descriptor=window_descriptors(x,fs);windows.append(dict(x=x,roi=roi,prior=prior,descriptors=descriptor))
        selected=roi&prior[:,0]
        green=x[selected,3].mean(0) if selected.any() else np.zeros(length)
        quality=signal_quality(green,fs,.7,3.5)
        rows.append(dict(start_sec=left,end_sec=right,source_roi_count=int(roi.sum()),face_green_hr_bpm=float(estimate_hr_welch(green,fs).hr_bpm),
            face_green_snr_db=float(quality['snr_db']),face_green_periodicity=float(quality['periodicity']),face_green_waveform=green))
    if not windows:raise ValueError('No complete ten-second model windows')
    arrays={k:np.stack([row[k] for row in windows]) for k in windows[0]}
    return arrays,rows,diagnostics


def load_models(scope_path,device):
    scope=json.loads(Path(scope_path).read_text());models={}
    for row in scope['checkpoints']:
        path=Path(row['path'])
        if file_sha256(path)!=row['sha256']:raise ValueError('Locked model bytes changed')
        if row['kind']=='waveform':
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)  # own hash-pinned artifact; NumPy scalar metadata
            c=row['identity']['configuration'];seed=row['identity']['seed'];name=c['name']+'__seed'+str(seed)
            model=AdaptiveWaveformNet(c['kind'],context=c['context'],width=c['width']) if c['architecture']=='compact' else make_extended_model(c['architecture'],c['width'])
            info=dict(kind='waveform',configuration=c,seed=seed)
        else:
            checkpoint=torch.load(path,map_location='cpu',weights_only=True);c=row['identity'];name=c['mode']+'__seed'+str(c['seed'])
            model=ThreePartFeatureModel(c['mode'],c['width']);info=dict(kind='feature',mode=c['mode'],seed=c['seed'],
                center=checkpoint['center'].numpy(),scale=checkpoint['scale'].numpy())
        model.load_state_dict(checkpoint['model_state'],strict=True);model.to(device).eval()
        models[name]=(model,info)
    return models


def infer_models(models,arrays,fs,device):
    waves={};features={};flags={}
    for name,(model,info) in models.items():
        outputs=[];usable=[]
        with torch.inference_mode():
            for begin in range(0,len(arrays['x']),32):
                batch={k:torch.as_tensor(v[begin:begin+32],device=device) for k,v in arrays.items()};n=len(batch['x'])
                if info['kind']=='waveform':
                    result=model(batch['x'],batch['roi'],batch['prior'],batch['descriptors'],torch.zeros(n,75,device=device),torch.zeros(n,dtype=torch.bool,device=device),fs)
                    value=result['ppg'].cpu().numpy()
                else:
                    result=model(batch['x'],batch['roi'],batch['prior'],batch['descriptors'],fs)
                    value=result['features'].cpu().numpy()*info['scale']+info['center']
                outputs.append(value);usable.extend(result['usable'].cpu().numpy().tolist())
        outputs=np.concatenate(outputs);usable=np.asarray(usable,bool)
        if info['kind']=='waveform':
            waves[name]=outputs;values,valid=targets(outputs,fs);valid&=usable[:,None]
        else:values=outputs;valid=np.isfinite(values)&usable[:,None]
        features[name]=values;flags[name]=valid
    return waves,features,flags


def infer_recording(native,session,models,device,out,config):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);all_features={};all_flags={};all_waveforms={};rates={}
    clean=without_pressure_values(session);events=cuff_events(clean,float(native['timestamps'][-1]))
    for fs in (30,60):
        arrays,rows,diagnostics=prepared_windows(native,fs,config)
        waves,features,flags=infer_models(models,arrays,fs,device)
        starts=np.asarray([r['start_sec'] for r in rows]);ends=starts+10
        candidate=[];verified=[];good=[]
        for row in rows:
            times=row['start_sec']+np.arange(10*fs)/fs;masks=timeline_masks(times,events,5.)
            candidate.append(bool(masks['candidate_cuff_free'].all()));verified.append(bool(masks['verified_cuff_free'].all()))
            good.append(row['source_roi_count']>0 and row['face_green_snr_db']>=0 and row['face_green_periodicity']>=.5)
        baseline=np.stack([r.pop('face_green_waveform') for r in rows]);values,valid=targets(baseline,fs)
        features['face_GREEN']=values;flags['face_GREEN']=valid;waves['face_GREEN']=baseline
        for name,value in features.items():all_features[f'{fs}Hz__'+name]=value;all_flags[f'{fs}Hz__'+name]=flags[name]
        for name,value in waves.items():all_waveforms[f'{fs}Hz__'+name]=value
        all_features[f'{fs}Hz__start']=starts;all_features[f'{fs}Hz__end']=ends
        all_flags[f'{fs}Hz__candidate_cuff_free']=np.asarray(candidate);all_flags[f'{fs}Hz__verified_cuff_free']=np.asarray(verified);all_flags[f'{fs}Hz__face_quality_good']=np.asarray(good)
        rates[str(fs)]=dict(windows=len(rows),candidate_cuff_free=sum(candidate),verified_cuff_free=sum(verified),face_quality_good=sum(good),window_rows=rows,adapter=diagnostics)
    np.savez_compressed(out/'face_waveforms.npz',**all_waveforms);np.savez_compressed(out/'features.npz',**all_features);np.savez_compressed(out/'feature_validity.npz',**all_flags)
    summary=dict(session_id=session['session_id'],subject_id=session['subject_id'],rates=rates,feature_names=FEATURES,
        no_bp_values_used=True,no_waveform_fitting=True,no_hand_neural_inference=True,physical_timing_calibrated=False,
        artifact_sha256={name:file_sha256(out/name) for name in ('face_waveforms.npz','features.npz','feature_validity.npz')})
    write_json(out/'inference_summary.json',summary)
    return summary
