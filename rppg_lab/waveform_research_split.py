"""Versioned subject/clip partitions for the explicitly authorized research reset."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np

RESET_SEED=20261110
ROLES=('train','validation','test')


def grouped_partition(identities,counts,seed):
    identities=sorted(identities)
    if len(set(identities))!=len(identities) or len(counts)!=3 or sum(counts)!=len(identities) or min(counts)<1:
        raise ValueError('Every unique identity must have exactly one nonempty role')
    order=np.random.default_rng(seed).permutation(identities).tolist();result={};offset=0
    for role,count in zip(ROLES,counts):
        result[role]=sorted(order[offset:offset+count]);offset+=count
    return result


def validate_partition(partition,expected):
    groups=[set(partition[role]) for role in ROLES]
    if any(not group for group in groups) or any(groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2))):
        raise ValueError('Partition identities overlap or a role is empty')
    if set.union(*groups)!=set(expected) or sum(map(len,groups))!=len(expected):
        raise ValueError('Partition loses or duplicates identities')


def make_reset_manifest(frozen):
    identities=sorted(frozen['train_ids']+frozen['validation_ids']+frozen['test_ids'])
    if len(identities)!=41:raise ValueError('Expected the audited 41 UBFC subjects')
    ubfc=grouped_partition(identities,(29,6,6),RESET_SEED)
    scamps=grouped_partition([f'P{i:06d}' for i in range(1,11)],(8,1,1),RESET_SEED+1)
    validate_partition(ubfc,identities);validate_partition(scamps,[f'P{i:06d}' for i in range(1,11)])
    return dict(version='waveform_subject_reset_v1',date='2026-11-08',seed=RESET_SEED,UBFC=ubfc,SCAMPS=scamps,
        authorization='User explicitly requested repartitioning all eligible public Train/Val/Test records for new models',
        exposure='Historically exposed internal benchmark; fresh weights do not erase prior analyst/model-selection exposure',
        windows=dict(seconds=10,ubfc_stride_seconds=2,scamps_saved_window_starts_seconds=[0,10],sample_rate_hz=30),
        test_policy='No test scoring until waveform and specialist validation comparisons and selection are locked',
        no_private_pilot=True,no_sealed_subject_access=True,frozen_v1_unchanged=True)


def json_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_development_arrays(data):
    """Open Train/Validation only; refuse test-array reads in this process."""
    from .artifacts import file_sha256
    data=Path(data).resolve();sealed=(data/'test_arrays.npz').resolve()
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            if Path(os.path.abspath(os.fsdecode(args[0])))==sealed:
                raise PermissionError('New Test is closed during development')
    sys.addaudithook(guard)
    receipt=json.loads((data/'data_receipt.json').read_text());parts=[]
    for role in ('train','validation'):
        record=receipt['partitions'][role];path=data/record['path']
        if path.resolve()!=data/(role+'_arrays.npz') or file_sha256(path)!=record['sha256']:
            raise ValueError('Partition file or checksum changed')
        with np.load(path,allow_pickle=False) as archive:parts.append({k:archive[k] for k in archive.files})
    if set(parts[0]['record_id'])&set(parts[1]['record_id']):raise ValueError('Train/Validation identity leakage')
    if parts[0]['validation'].any() or not parts[1]['validation'].all():raise ValueError('Incorrect role flags')
    arrays={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return arrays,np.arange(len(parts[0]['x'])),np.arange(len(parts[0]['x']),len(arrays['x'])),receipt
