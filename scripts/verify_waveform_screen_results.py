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
    length=len(p)-30
    scores=[corr(p[start:start+length],y[15:-15]) for start in range(31)]
    best=int(np.argmax(scores))
    return dict(valid=True,native_r=corr(p,y),nrmse=float(np.sqrt(np.mean((p-y)**2))),
        derivative_r=corr(np.diff(p),np.diff(y)),morphology_r=scores[best],
        morphology_d1_r=corr(np.diff(p[best:best+length]),np.diff(y[15:-15])),
        hr_mae=abs(float(estimate_hr_welch(p,30).hr_bpm)-float(estimate_hr_welch(y,30).hr_bpm)))


def verify_grouped(args,report):
    with np.load(args.data/'development_arrays.npz',allow_pickle=False) as source:
        target=source['y'];ids=source['record_id'];sources=source['source']
    if file_sha256(args.screen/'predictions.npz')!=report['predictions_sha256'] or file_sha256(args.screen/'prediction_indices.npz')!=report['indices_sha256']:
        raise ValueError('Grouped prediction/index archive changed')
    checked=0;max_error=0.;checkpoints=0;cache={}
    def compare(values,indices,table):
        nonlocal checked,max_error
        for source,name in ((0,'UBFC'),(1,'SCAMPS')):
            chosen=[j for j,i in enumerate(indices) if sources[i]==source]
            people=sorted({str(ids[indices[j]]) for j in chosen});expected=table[name]
            if len(chosen)!=expected['intended_windows'] or len(people)!=expected['subjects']:
                raise ValueError('Grouped denominator changed')
            if sum(values[j]['valid'] for j in chosen)!=expected['valid_predictions']:raise ValueError('Failure denominator changed')
            for key,want in expected['subject_balanced'].items():
                if want is None:
                    if chosen:raise ValueError('Unexpected nonempty missing metric')
                    continue
                means=[np.mean([values[j][key] for j in chosen if str(ids[indices[j]])==person and values[j]['valid']]) for person in people]
                error=abs(float(np.mean(means))-want);max_error=max(max_error,error)
                if error>2e-5:raise ValueError(f'Grouped replay failed: {name}/{key}: {error}')
                checked+=1
    with np.load(args.screen/'predictions.npz',allow_pickle=False) as predictions,np.load(args.screen/'prediction_indices.npz',allow_pickle=False) as index_archive:
        for key in predictions.files:
            indices=index_archive[key];p=predictions[key]
            if len(np.unique(indices))!=len(indices) or p.shape!=(len(indices),300):raise ValueError('Invalid grouped indices')
            cache[key]=(indices,[metric(v,target[i]) for v,i in zip(p,indices)])
        for row in report['runs']:
            name=row.get('variant') or row['configuration']['name'];key=f"fold{row['fold']}_{name}"
            compare(cache[key][1],cache[key][0],row['summary'])
            if file_sha256(args.screen/(key+'.pt'))!=row['checkpoint_sha256']:raise ValueError('Grouped checkpoint changed')
            checkpoints+=1
        for name,entry in report['summaries'].items():
            parts=[cache[f'fold{fold}_{name}'] for fold in range(5)]
            indices=np.concatenate([i for i,_ in parts]);values=sum([v for _,v in parts],[])
            if len(indices)!=152 or len(set(indices))!=152:raise ValueError('OOF participant/window denominator changed')
            compare(values,indices,entry.get('summary',entry))
        for row in report.get('seed_replications',[]):
            indices,values=cache[row['name']];compare(values,indices,row['summary'])
    return dict(saved_aggregates_recomputed=checked,registered_checkpoint_hashes_verified=checkpoints,
        max_absolute_metric_difference=max_error,neural_forwards=0,checkpoint_loads=0,original_dataset_access=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--screen',type=Path,required=True);args=parser.parse_args()
    report=json.loads((args.screen/'results.json').read_text())
    if file_sha256(args.data/'development_arrays.npz')!=report['data_sha256']:raise ValueError('Data hash mismatch')
    if 'runs' in report:
        receipt=verify_grouped(args,report)
        with (args.screen/'independent_verification.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2)
        print(json.dumps(receipt,indent=2));return
    with np.load(args.data/'development_arrays.npz',allow_pickle=False) as source:
        target=source['y'];ids=source['record_id'];sources=source['source'];indices=np.flatnonzero(source['validation'])
    checked=0;max_error=0.;checkpoints=0
    extended='configuration' in report['results'][0]
    prediction_path=args.screen/('predictions.npz' if extended else 'validation_predictions.npz')
    if file_sha256(prediction_path)!=report['predictions_sha256']:raise ValueError('Prediction archive changed')
    with np.load(prediction_path,allow_pickle=False) as archive:
        for result in report['results']:
            if extended:
                result=dict(result,name=result['configuration']['name'],final=result['summary'],learned=True)
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
        max_absolute_metric_difference=max_error,predictions_sha256=file_sha256(prediction_path),
        neural_forwards=0,checkpoint_loads=0,original_dataset_access=False)
    with (args.screen/'independent_verification.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
