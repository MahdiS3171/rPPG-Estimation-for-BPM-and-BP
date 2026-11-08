"""Prepare all eligible public records under a new, precommitted subject split.

Frozen identities and raw historical caches are read-only. No video extraction,
raw private/pilot access or model fitting occurs. Test arrays are stored apart.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np

from scripts.study_two_source_waveform import MANIFEST,MANIFEST_HASH,historical_cache_map,load_cached
from rppg_lab.artifacts import file_sha256
from rppg_lab.config import PipelineConfig
from rppg_lab.datasets import UBFCSubject,UBFCMultiROIRPPGDataset
from rppg_lab.signals import standardize_1d
from rppg_lab.waveform_v2_frozen_cache import verified_artifact
from rppg_lab.waveform_v2_screen_features import window_descriptors
from rppg_lab.waveform_research_split import make_reset_manifest,json_sha256,ROLES


def install_reset_guard(root,identities):
    blocked=[root/name for name in ('Dataset','sheets','data_sessions','rPPG-10','SCAMPSData')]
    def guard(event,args):
        if event not in ('open','os.listdir','os.scandir') or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.path.abspath(os.fsdecode(args[0])))
        if any(path==p or p in path.parents for p in blocked):raise PermissionError('Private/raw nonstudy data is closed')
        ubfc=root/'UBFCData'
        if ubfc in path.parents and path.relative_to(ubfc).parts[0] not in identities:
            raise PermissionError('Outside explicitly admitted public subjects')
    sys.addaudithook(guard)


def pack(records):
    first={}
    for row in records:
        row['descriptors']=window_descriptors(row['x'])
        if row['id'] not in first or row['start']<first[row['id']]['start']:first[row['id']]=row
    for row in records:
        initial=first[row['id']];valid=row['start']>=initial['start']+10-1e-8
        row['context_valid']=valid;row['context']=initial['descriptors'].copy() if valid else np.zeros(75,np.float32)
    values={k:np.stack([r[k] for r in records]) for k in ('x','y','roi','prior','descriptors','context','context_valid')}
    values.update(record_id=np.asarray([r['id'] for r in records]),source=np.asarray([r['source'] for r in records]),
        start=np.asarray([r['start'] for r in records]),validation=np.asarray([r['role']=='validation' for r in records]))
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--split',type=Path,required=True)
    parser.add_argument('--out',type=Path);parser.add_argument('--lock-only',action='store_true');args=parser.parse_args()
    if file_sha256(MANIFEST)!=MANIFEST_HASH:raise ValueError('Frozen manifest changed')
    frozen=json.loads(MANIFEST.read_text());lock=make_reset_manifest(frozen)
    if args.lock_only:
        with args.split.open('x',encoding='utf-8') as stream:json.dump(lock,stream,indent=2);stream.write('\n')
        print(json.dumps({s:{r:len(v) for r,v in lock[s].items()} for s in ('UBFC','SCAMPS')}));return
    if json.loads(args.split.read_text())!=lock:raise ValueError('Split differs from predeclared allocation')
    if not args.out:parser.error('--out is required for preparation')
    root=Path.cwd().resolve();out=args.out.resolve();allowed=(root/'outputs/waveform_reset').resolve()
    if allowed not in out.parents or out.exists():parser.error('Use a NEW reset output directory')
    all_ids=sorted(sum([lock['UBFC'][role] for role in ROLES],[]));install_reset_guard(root,set(all_ids))
    c=dict(frozen['extraction_config'])
    for key in ('regions','skin_cr_range','skin_cb_range','face_rois'):c[key]=tuple(c[key])
    config=PipelineConfig(**c)
    # Only widen the helper's local admission view. The frozen manifest bytes and
    # its historical split remain unchanged.
    mapping=historical_cache_map(dict(frozen,train_ids=all_ids),config)
    hashes={str(path):file_sha256(path) for path,_ in mapping.values()}
    role_of={f'{source}:{identity}':role for source in ('UBFC','SCAMPS') for role in ROLES for identity in lock[source][role]}
    records=[];exclusions=[]
    out.mkdir(parents=True,exist_ok=False)
    (out/'split_manifest.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
    for subject in all_ids:
        item=UBFCSubject(subject,Path(mapping[subject][1]['identity']['video']['path']),root/'UBFCData'/subject/'ground_truth.txt')
        hashes[str(item.gt_path)]=file_sha256(item.gt_path)
        def reader(*a,**kw):return load_cached(*a,**kw,mapping=mapping)
        with patch('rppg_lab.extraction_cache.load_face_roi_extraction',side_effect=reader),patch('rppg_lab.extraction_cache.process_recording',side_effect=AssertionError('No cache rebuild')):
            dataset=UBFCMultiROIRPPGDataset([item],fs_target=30,win_sec=10,stride_sec=2,extraction_config=config)
            exclusions.extend(dict(row,role=role_of['UBFC:'+subject]) for row in dataset.exclusions)
            for i in range(len(dataset)):
                value=dataset[i];identity='UBFC:'+subject
                records.append(dict(id=identity,source=0,role=role_of[identity],start=float(value['start_sec']),
                    x=value['x'].numpy(),y=value['y_ppg'].numpy(),roi=value['roi_valid'].numpy(),prior=value['prior_valid'].numpy()))
        print(json.dumps(dict(subject=subject,role=role_of['UBFC:'+subject],windows=len(dataset))),flush=True)
    meta=json.loads(Path('docs/waveform_v2/classical_gate08/local_artifact_manifest.json').read_text())
    parent=Path(meta['local_root']);a=verified_artifact(parent,'locked_predictions.npz');b=verified_artifact(parent,'reference_windows.npz')
    hashes.update({str(p):file_sha256(p) for p in (a,b)})
    geometry={r['filename']:r for r in json.loads(Path('docs/waveform_v2/scamps_gate06/roi_coverage_and_inputs.json').read_text())['recordings']}
    with np.load(a,allow_pickle=False) as inputs,np.load(b,allow_pickle=False) as labels:
        for number in range(1,11):
            name=f'P{number:06d}';identity='SCAMPS:'+name
            for w in range(2):
                m=geometry[name+'.mat']['windows'][w]
                records.append(dict(id=identity,source=1,role=role_of[identity],start=10.*w,x=inputs[name+'_x'][w],
                    y=standardize_1d(labels[name][w]),roi=np.asarray(m['roi_valid'],bool),prior=np.asarray(m['prior_valid'],bool)))
    receipts={}
    for role in ROLES:
        selected=[r for r in records if r['role']==role];arrays=pack(selected);path=out/(role+'_arrays.npz')
        if set(arrays['record_id'])!={key for key,value in role_of.items() if value==role}:raise ValueError('A partition identity lost every window')
        if not all(np.isfinite(arrays[key]).all() for key in ('x','y','descriptors','context')):raise ValueError('Nonfinite prepared input/reference')
        np.savez_compressed(path,**arrays)
        receipts[role]=dict(path=path.name,sha256=file_sha256(path),windows=len(selected),
            source_window_counts=dict(Counter('UBFC' if r['source']==0 else 'SCAMPS' for r in selected)),
            source_identity_counts={source:len(lock[source][role]) for source in ('UBFC','SCAMPS')})
    # Exact duplicate input screening does not select or score against references.
    fingerprints={role:{__import__('hashlib').sha256(r['x'].tobytes()).hexdigest() for r in records if r['role']==role} for role in ROLES}
    overlap={a+'/'+b:len(fingerprints[a]&fingerprints[b]) for a,b in (('train','validation'),('train','test'),('validation','test'))}
    if any(overlap.values()):raise ValueError('Exact duplicate inputs cross partition roles')
    if any(file_sha256(p)!=h for p,h in hashes.items()):raise ValueError('Read-only source changed')
    receipt=dict(status='RESET_PUBLIC_SUBJECT_SPLIT_NOT_UNSEEN_DOMAIN',split_sha256=json_sha256(lock),partitions=receipts,
        exact_cross_role_duplicates=overlap,source_sha256=hashes,exclusions=exclusions,all_41_real_and_10_synthetic_assigned=True,
        windows_overlap_within_subject=True,labels_hz=30,no_frozen_changes=True,no_private_data=True,
        test_scores_computed=False,git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (out/'data_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(partitions=receipts,exclusions=len(exclusions),duplicates=overlap),indent=2))


if __name__=='__main__':main()
