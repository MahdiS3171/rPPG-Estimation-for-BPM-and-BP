"""Prepare small immutable development arrays from allowed historical sources."""
import argparse
from pathlib import Path
import hashlib
import json
from unittest.mock import patch

import numpy as np

from scripts.study_two_source_waveform import (MANIFEST,MANIFEST_HASH,SCAMPS_IDS,
    install_guard,historical_cache_map,load_cached)
from rppg_lab.artifacts import file_sha256
from rppg_lab.config import PipelineConfig
from rppg_lab.datasets import UBFCSubject,UBFCMultiROIRPPGDataset
from rppg_lab.signals import standardize_1d
from rppg_lab.waveform_v2_frozen_cache import verified_artifact
from rppg_lab.waveform_v2_screen_features import window_descriptors


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=Path.cwd();out=args.out.absolute()
    if (root/'outputs/waveform_screen').resolve() not in out.resolve().parents or out.exists():
        parser.error('Use a NEW directory within outputs/waveform_screen/')
    if file_sha256(MANIFEST)!=MANIFEST_HASH:raise ValueError('Frozen manifest changed')
    manifest=json.loads(MANIFEST.read_text());install_guard(root,set(manifest['train_ids']))
    c=dict(manifest['extraction_config'])
    for key in ('regions','skin_cr_range','skin_cb_range','face_rois'):c[key]=tuple(c[key])
    config=PipelineConfig(**c);mapping=historical_cache_map(manifest,config)
    hashes={str(path):file_sha256(path) for path,_ in mapping.values()}
    subjects=[UBFCSubject(s,Path(mapping[s][1]['identity']['video']['path']),Path('UBFCData')/s/'ground_truth.txt') for s in sorted(mapping)]
    hashes.update({str(s.gt_path):file_sha256(s.gt_path) for s in subjects})
    def reader(*a,**kw):return load_cached(*a,**kw,mapping=mapping)
    records=[]
    with patch('rppg_lab.extraction_cache.load_face_roi_extraction',side_effect=reader),patch('rppg_lab.extraction_cache.process_recording',side_effect=AssertionError('No rebuild')):
        dataset=UBFCMultiROIRPPGDataset(subjects,fs_target=30,win_sec=10,stride_sec=10,extraction_config=config)
        for i in range(len(dataset)):
            item=dataset[i]
            records.append(dict(id='UBFC:'+item['subject_id'],source=0,start=float(item['start_sec']),
                x=item['x'].numpy(),y=item['y_ppg'].numpy(),roi=item['roi_valid'].numpy(),prior=item['prior_valid'].numpy()))
            if (i+1)%25==0:print(f'Prepared UBFC input {i+1}/{len(dataset)}',flush=True)
    meta=json.loads(Path('docs/waveform_v2/classical_gate08/local_artifact_manifest.json').read_text())
    source=Path(meta['local_root']);a=verified_artifact(source,'locked_predictions.npz');b=verified_artifact(source,'reference_windows.npz')
    hashes.update({str(p):file_sha256(p) for p in (a,b)})
    coverage=json.loads(Path('docs/waveform_v2/scamps_gate06/roi_coverage_and_inputs.json').read_text())
    geometry={r['filename']:r for r in coverage['recordings']}
    with np.load(a,allow_pickle=False) as inputs,np.load(b,allow_pickle=False) as labels:
        for number in SCAMPS_IDS:
            name=f'P{number:06d}'
            for w in range(2):
                m=geometry[name+'.mat']['windows'][w]
                records.append(dict(id='SCAMPS:'+name,source=1,start=10.0*w,x=inputs[name+'_x'][w],
                    y=standardize_1d(labels[name][w]),roi=np.asarray(m['roi_valid'],bool),prior=np.asarray(m['prior_valid'],bool)))
    # Fixed initial ten-second INPUT calibration, used only after it has ended.
    first={}
    for r in records:
        r['descriptors']=window_descriptors(r['x'])
        if r['id'] not in first or r['start']<first[r['id']]['start']:first[r['id']]=r
    for r in records:
        calibration=first[r['id']];valid=r['start']>=calibration['start']+10-1e-8
        r['context']=calibration['descriptors'].copy() if valid else np.zeros(75,np.float32)
        r['context_valid']=valid
    ubfc=sorted({r['id'] for r in records if r['source']==0});scamps=sorted({r['id'] for r in records if r['source']==1})
    # Same six real development identities as the first fixed baseline fold.
    ubfc_val=list(np.random.default_rng(20261108).permutation(ubfc)[:6])
    scamps_val=list(np.random.default_rng(20261109).permutation(scamps)[:2])
    validation=set(ubfc_val+scamps_val)
    out.mkdir(parents=True,exist_ok=False)
    arrays={k:np.stack([r[k] for r in records]) for k in ('x','y','roi','prior','descriptors','context','context_valid')}
    arrays.update(record_id=np.asarray([r['id'] for r in records]),source=np.asarray([r['source'] for r in records]),
        start=np.asarray([r['start'] for r in records]),validation=np.asarray([r['id'] in validation for r in records]))
    np.savez_compressed(out/'development_arrays.npz',**arrays)
    if any(file_sha256(p)!=h for p,h in hashes.items()):raise RuntimeError('Historical source changed')
    receipt=dict(status='DEVELOPMENT_ONLY',sample_rate_hz=30,window_seconds=10,
        examples=len(records),ubfc_examples=sum(r['source']==0 for r in records),scamps_examples=16,
        validation_subjects=ubfc_val,validation_scamps=scamps_val,calibration='First 10 seconds of input only; unavailable for the first target window',
        no_original_val_test=True,no_private_data=True,no_native60_training_labels=True,
        data_sha256=hashes,arrays_sha256=file_sha256(out/'development_arrays.npz'))
    (out/'data_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='data_sha256'},indent=2))


if __name__=='__main__':main()
