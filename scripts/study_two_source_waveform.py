"""Bounded train-only, two-source native waveform baseline; no frozen-model edits."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import zipfile

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import numpy as np

from rppg_lab.artifacts import file_sha256
from rppg_lab.config import PipelineConfig
from rppg_lab.datasets import UBFCSubject, UBFCMultiROIRPPGDataset, sliding_windows
from rppg_lab.extraction_cache import FaceROIExtraction, extraction_cache_identity, _validate
from rppg_lab.signals import standardize_1d, estimate_hr_welch
from rppg_lab.types import RGBTrace, SharedTimebase
from rppg_lab.waveform_v2_classical_study import correlation, finite_mean
from rppg_lab.waveform_v2_frozen_cache import verified_artifact, clean

MANIFEST=Path('checkpoints/waveform_v1_official_ubfc/waveform_v1_freeze_manifest.json')
MANIFEST_HASH='da5ba397b3be389deec8821f8ea16944b5f5b93e2834bb79c3c147eb3d1f89dc'
SCAMPS_IDS=(1,2,3,6,7,8,9,10)
METHODS=('GREEN','negative_GREEN','CHROM','CHROM_WIN','POS_WIN',
         'priors_UBFC','priors_two_source','RGB_priors_UBFC','RGB_priors_two_source')


def install_guard(root,train_ids):
    forbidden=[root/p for p in ('Dataset','sheets','rPPG-10','SCAMPSData','data_sessions')]
    def guard(event,args):
        if event not in ('open','os.listdir','os.scandir') or not args or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        p=Path(os.path.abspath(os.fsdecode(args[0])))
        if any(p==x or x in p.parents for x in forbidden):
            raise PermissionError('Private/raw nonstudy source is closed')
        ubfc=root/'UBFCData'
        if ubfc in p.parents and p.relative_to(ubfc).parts[0] not in train_ids:
            raise PermissionError('Only original UBFC Train subjects are admitted')
    sys.addaudithook(guard)


def historical_cache_map(manifest,config):
    result={}
    for path in Path('cache_roi_multi_phase1').glob('phase1_face_*.npz'):
        with zipfile.ZipFile(path) as archive, archive.open('metadata_json.npy') as stream:
            meta=json.loads(str(np.lib.format.read_array(stream,allow_pickle=False).item()))
        ident=meta['identity']
        subject=Path(ident['video']['path']).parent.name
        if subject not in manifest['train_ids'] or ident['max_frames'] is not None or ident['config']!=manifest['extraction_config']:
            continue
        current=extraction_cache_identity(Path(ident['video']['path']),config,None)
        differences={k for k in current if current[k]!=ident[k]}
        if differences-{'face_model'} or ident['face_model']['sha256'] is not None:
            raise ValueError('Unexpected historical cache provenance difference')
        if subject in result:
            raise ValueError('Ambiguous full-length training cache')
        result[subject]=(path,meta)
    if set(result)!=set(manifest['train_ids']):
        raise ValueError('Every original Train subject needs one historical full cache')
    return result


def load_cached(video,config,cache_dir=None,max_frames=None,*,mapping):
    subject=Path(video).parent.name
    path,meta=mapping[subject]
    with np.load(path,allow_pickle=False) as archive:
        source=archive['timestamps_original']
        traces={}
        for name in config.requested_face_rois:
            prefix='face_roi_'+name+'_'
            quality={k:archive[prefix+k] for k in meta['quality_keys'][name]}
            traces[name]=RGBTrace(source,archive[prefix+'rgb'],meta['trace_roi_names'][name],archive[prefix+'valid'],quality)
        return _validate(FaceROIExtraction(SharedTimebase(archive['timestamps_uniform'],source),traces),config)


def fit_ridge(groups,indices):
    """Each source receives equal mass, then each subject, window and time point."""
    k=len(indices); gram=np.zeros((k,k)); rhs=np.zeros(k)
    for source in groups:
        people=sorted({r['id'] for r in source})
        for person in people:
            rows=[r for r in source if r['id']==person]
            for row in rows:
                x=row['x'][indices].T.astype(float); y=row['y'].astype(float)
                weight=1/(len(groups)*len(people)*len(rows)*len(y))
                gram+=weight*(x.T@x); rhs+=weight*(x.T@y)
    return np.linalg.solve(gram+.01*np.eye(k),rhs)


def metrics(prediction,target):
    valid=np.isfinite(prediction).all() and np.std(prediction)>1e-8
    if not valid:
        return dict(valid=False,native_r=None,nrmse=None,derivative_r=None,hr_mae=None)
    p=standardize_1d(prediction); y=standardize_1d(target)
    return dict(valid=True,native_r=correlation(p,y),nrmse=float(np.sqrt(np.mean((p-y)**2))),
        derivative_r=correlation(np.diff(p),np.diff(y)),
        hr_mae=abs(float(estimate_hr_welch(p,30).hr_bpm)-float(estimate_hr_welch(y,30).hr_bpm)))


def run(output):
    started=time.perf_counter(); root=Path.cwd()
    if file_sha256(MANIFEST)!=MANIFEST_HASH: raise ValueError('Frozen manifest changed')
    manifest=json.loads(MANIFEST.read_text())
    install_guard(root,set(manifest['train_ids']))
    ids=sorted(manifest['train_ids']); shuffled=list(np.random.default_rng(20261108).permutation(ids))
    folds=[list(x) for x in np.array_split(shuffled,5)]
    config_data=dict(manifest['extraction_config'])
    for key in ('regions','skin_cr_range','skin_cb_range','face_rois'): config_data[key]=tuple(config_data[key])
    config=PipelineConfig(**config_data)
    mapping=historical_cache_map(manifest,config)
    hashes={str(path):file_sha256(path) for path,_ in mapping.values()}
    subjects=[UBFCSubject(s,Path(mapping[s][1]['identity']['video']['path']),Path('UBFCData')/s/'ground_truth.txt') for s in ids]
    hashes.update({str(s.gt_path):file_sha256(s.gt_path) for s in subjects})
    def reader(*args,**kwargs): return load_cached(*args,**kwargs,mapping=mapping)
    with patch('rppg_lab.extraction_cache.load_face_roi_extraction',side_effect=reader), patch('rppg_lab.extraction_cache.process_recording',side_effect=AssertionError('No video decode/rebuild')):
        dataset=UBFCMultiROIRPPGDataset(subjects,fs_target=30,win_sec=10,stride_sec=10,extraction_config=config)
        ubfc=[]
        for i in range(len(dataset)):
            item=dataset[i]
            ubfc.append(dict(id=item['subject_id'],start=float(item['start_sec']),
                x=item['x'].numpy().reshape(27,300),y=item['y_ppg'].numpy(),roi_valid=item['roi_valid'].numpy()))
    intended=sum(len(sliding_windows(len(r['timestamps']),30,10,10)) for r in dataset.records)
    print(f'Prepared UBFC Train only: {len(ids)} subjects, {len(ubfc)}/{intended} windows',flush=True)
    gate=json.loads(Path('docs/waveform_v2/classical_gate08/local_artifact_manifest.json').read_text())
    source=Path(gate['local_root'])
    input_path=verified_artifact(source,'locked_predictions.npz')
    reference_path=verified_artifact(source,'reference_windows.npz')
    hashes.update({str(p):file_sha256(p) for p in (input_path,reference_path)})
    scamps=[]
    with np.load(input_path,allow_pickle=False) as features,np.load(reference_path,allow_pickle=False) as references:
        for number in SCAMPS_IDS:
            stem=f'P{number:06d}'
            x=features[stem+'_x']; y=references[stem]
            if x.shape!=(2,3,9,300) or y.shape!=(2,300): raise ValueError('SCAMPS fixed shape changed')
            for w in range(2):
                scamps.append(dict(id=stem,start=10*w,x=x[w].reshape(27,300),y=standardize_1d(y[w])))
    output.mkdir(parents=True,exist_ok=False)
    predictions={}; rows=[]; coefficients={}
    for fold,held in enumerate(folds):
        fit=[r for r in ubfc if r['id'] not in held]
        held_rows=[r for r in ubfc if r['id'] in held]
        assert {r['id'] for r in fit}.isdisjoint(held)
        learned={}
        for feature_name,indices in [('priors',np.array([9*r+c for r in range(3) for c in range(3,9)])),('RGB_priors',np.arange(27))]:
            for scheme,groups in [('UBFC',[fit]),('two_source',[fit,scamps])]:
                name=feature_name+'_'+scheme
                coef=fit_ridge(groups,indices)
                coefficients[str(fold)+'/'+name]=coef.tolist()
                learned[name]=(indices,coef)
        for record in held_rows:
            x=record['x'].reshape(3,9,300); valid=record['roi_valid']
            fixed={name:x[valid,channel].mean(0) for name,channel in [('GREEN',3),('CHROM',4),('CHROM_WIN',5),('POS_WIN',7)]}
            fixed['negative_GREEN']=-fixed['GREEN']
            fixed.update({name:coef@record['x'][indices] for name,(indices,coef) in learned.items()})
            for name in METHODS:
                prediction=fixed[name]
                key=record['id']+'_'+str(record['start'])+'_'+name
                predictions[key]=prediction.astype(np.float32)
                rows.append(dict(subject=record['id'],start=record['start'],fold=fold,method=name,
                    prediction_sha256=hashlib.sha256(prediction.astype(np.float32).tobytes()).hexdigest(),
                    metrics=metrics(prediction,record['y'])))
        print(f'Fold {fold+1}/5: {len(held)} development subjects predicted; no fit overlap',flush=True)
    summaries=[]
    for name in METHODS:
        selected=[r for r in rows if r['method']==name]
        per_subject={s:{k:finite_mean([r['metrics'][k] for r in selected if r['subject']==s and r['metrics'][k] is not None])
            for k in ('native_r','nrmse','derivative_r','hr_mae')} for s in ids}
        summaries.append(dict(method=name,intended_windows=intended,scored_windows=len(selected),
            valid_predictions=sum(r['metrics']['valid'] for r in selected),subjects=len(per_subject),
            subject_balanced={k:finite_mean([v[k] for v in per_subject.values()]) for k in ('native_r','nrmse','derivative_r','hr_mae')},
            per_subject=per_subject))
    if any(file_sha256(p)!=h for p,h in hashes.items()): raise RuntimeError('Historical source changed')
    np.savez_compressed(output/'locked_predictions.npz',**predictions)
    report=dict(status='DEVELOPMENT_ONLY_NOT_THIRD_DATASET_TEST',train_sources=['UBFC_original_Train','SCAMPS_original_Train_examples'],
        original_ubfc_val_test_used=False,scamps_val_test_used=False,private_data_used=False,neural_training=False,
        global_linear_models_fitted=20,lambda_fixed=.01,source_balanced=True,folds=folds,
        ubfc_subjects=len(ids),ubfc_intended_windows=intended,ubfc_prepared_windows=len(ubfc),scamps_recordings=8,scamps_windows=16,
        cache_policy='Read-only historical legacy-detector traces; current Tasks bundle is not retroactively assigned',
        data_sha256=hashes,prediction_archive_sha256=file_sha256(output/'locked_predictions.npz'),
        coefficients=coefficients,summaries=summaries,window_rows=rows,exclusions=dataset.exclusions,
        runtime_seconds=time.perf_counter()-started,git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (output/'results.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(clean([dict(method=r['method'],**r['subject_balanced']) for r in summaries]),indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args(); root=Path(__file__).resolve().parents[1]; out=args.out.absolute()
    if Path.cwd()!=root or (root/'outputs/two_source_waveform').resolve() not in out.resolve().parents or out.exists():
        parser.error('Run at root with a NEW directory inside outputs/two_source_waveform/')
    run(out)


if __name__=='__main__': main()
