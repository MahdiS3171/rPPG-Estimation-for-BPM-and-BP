"""Recompute saved validation metrics without loading/executing neural models."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from rppg_lab.artifacts import file_sha256
from rppg_lab.signals import estimate_hr_welch


def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);a=a-a.mean();b=b-b.mean()
    return float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)+1e-8))


def metric(p,y):
    p=np.asarray(p,float);y=np.asarray(y,float)
    if not np.isfinite(p).all() or np.std(p)<1e-8:return dict(valid=False)
    p=(p-p.mean())/(p.std()+1e-8);y=(y-y.mean())/(y.std()+1e-8)
    scores=[corr(p[start:start+270],y[15:-15]) for start in range(31)]
    best=int(np.argmax(scores))
    return dict(valid=True,native_r=corr(p,y),nrmse=float(np.sqrt(np.mean((p-y)**2))),
        derivative_r=corr(np.diff(p),np.diff(y)),morphology_r=scores[best],
        morphology_d1_r=corr(np.diff(p[best:best+270]),np.diff(y[15:-15])),
        hr_mae=abs(float(estimate_hr_welch(p,30).hr_bpm)-float(estimate_hr_welch(y,30).hr_bpm)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--screen',type=Path,required=True);args=parser.parse_args()
    report=json.loads((args.screen/'results.json').read_text())
    if file_sha256(args.data/'development_arrays.npz')!=report['data_sha256']:raise ValueError('Data hash mismatch')
    with np.load(args.data/'development_arrays.npz',allow_pickle=False) as source:
        target=source['y'];ids=source['record_id'];sources=source['source'];indices=np.flatnonzero(source['validation'])
    checked=0;max_error=0.;checkpoints=0
    with np.load(args.screen/'validation_predictions.npz',allow_pickle=False) as archive:
        for result in report['results']:
            if result.get('learned'):
                if file_sha256(args.screen/(result['name']+'.pt'))!=result['checkpoint_sha256']:raise ValueError('Checkpoint changed')
                checkpoints+=1
            for suffix,summary in [('',result['final'])]+([('__initial',result['initial'])] if result.get('learned') else []):
                p=archive[result['name']+suffix]
                if p.shape!=(len(indices),300):raise ValueError('Prediction denominator changed')
                values=[metric(prediction,target[index]) for prediction,index in zip(p,indices)]
                for source,name in ((0,'UBFC'),(1,'SCAMPS')):
                    selected=[j for j,i in enumerate(indices) if sources[i]==source]
                    subject_ids=sorted({str(ids[indices[j]]) for j in selected})
                    if len(selected)!=summary[name]['intended_windows']:raise ValueError('Window count changed')
                    if sum(values[j]['valid'] for j in selected)!=summary[name]['valid_predictions']:raise ValueError('Failure count changed')
                    for key,expected in summary[name]['subject_balanced'].items():
                        means=[]
                        for person in subject_ids:
                            v=[values[j][key] for j in selected if str(ids[indices[j]])==person and values[j]['valid']]
                            means.append(float(np.mean(v)) if v else np.nan)
                        actual=float(np.nanmean(means))
                        error=abs(actual-expected);max_error=max(max_error,error)
                        if error>2e-5:raise ValueError(f'Metric replay failed: {result["name"]}/{suffix}/{name}/{key}: {error}')
                        checked+=1
    receipt=dict(saved_aggregates_recomputed=checked,checkpoint_hashes_verified=checkpoints,
        max_absolute_metric_difference=max_error,predictions_sha256=file_sha256(args.screen/'validation_predictions.npz'),
        neural_forwards=0,checkpoint_loads=0,original_dataset_access=False)
    with (args.screen/'independent_verification.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
