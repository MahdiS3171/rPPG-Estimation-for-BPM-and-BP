"""Replay reset waveform metrics in NumPy; no neural inference or Test access."""
import argparse
import json
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from rppg_lab.artifacts import file_sha256
from rppg_lab.waveform_research_split import load_development_arrays
from rppg_lab.signals import estimate_hr_welch

FIELDS=('native_r','nrmse','derivative_r','morphology_r','morphology_d1_r','hr_mae')


def correlation(a,b):
    a=a-a.mean(-1,keepdims=True);b=b-b.mean(-1,keepdims=True)
    return (a*b).sum(-1)/np.sqrt((a*a).sum(-1)*(b*b).sum(-1)+1e-8)


def metrics(prediction,target):
    p=np.asarray(prediction,float);y=np.asarray(target,float)
    usable=np.isfinite(p).all(-1)&(p.std(-1)>1e-8)
    p=(p-p.mean(-1,keepdims=True))/(p.std(-1,keepdims=True)+1e-8)
    y=(y-y.mean(-1,keepdims=True))/(y.std(-1,keepdims=True)+1e-8)
    windows=sliding_window_view(p,270,axis=-1);central=y[:,None,15:-15]
    shape=correlation(windows,central);best=shape.argmax(-1)
    aligned=windows[np.arange(len(p)),best]
    errors=[abs(estimate_hr_welch(a,30).hr_bpm-estimate_hr_welch(b,30).hr_bpm) for a,b in zip(p,y)]
    values=np.stack((correlation(p,y),np.sqrt(((p-y)**2).mean(-1)),correlation(np.diff(p),np.diff(y)),
        shape[np.arange(len(p)),best],correlation(np.diff(aligned),np.diff(y[:,15:-15])),errors),axis=1)
    if not np.isfinite(values[usable]).all():raise ValueError('Nonfinite valid metric')
    return values,usable


def verify_table(values,usable,arrays,indices,expected):
    checked=0;maximum=0.
    for source,name in ((0,'UBFC'),(1,'SCAMPS')):
        positions=np.flatnonzero(arrays['source'][indices]==source);ids=arrays['record_id'][indices]
        people=sorted(set(ids[positions]));table=expected[name]
        if (len(positions),len(people),int(usable[positions].sum()))!=(table['intended_windows'],table['subjects'],table['valid_predictions']):
            raise ValueError('Source/failure denominator mismatch')
        means=[]
        for person in people:
            group=positions[(ids[positions]==person)&usable[positions]]
            if not len(group):continue
            mean=values[group].mean(0);means.append(mean)
            for column,key in enumerate(FIELDS):
                difference=abs(float(mean[column])-table['per_subject'][person][key]);maximum=max(maximum,difference)
                if not np.isfinite(difference) or difference>2e-5:raise ValueError('Per-subject value mismatch')
                checked+=1
        mean=np.mean(means,axis=0) if means else [None]*6
        for column,key in enumerate(FIELDS):
            wanted=table['subject_balanced'][key]
            if wanted is None:
                if mean[column] is not None:raise ValueError('Unexpected defined source metric')
                continue
            difference=abs(float(mean[column])-wanted);maximum=max(maximum,difference)
            if not np.isfinite(difference) or difference>2e-5:raise ValueError('Source mean mismatch')
            checked+=1
    return checked,maximum


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True);args=parser.parse_args()
    arrays,fit,validation,data=load_development_arrays(args.data)
    report=json.loads((args.campaign/'results.json').read_text())
    if not report['complete'] or report['split_sha256']!=data['split_sha256']:raise ValueError('Incomplete/mismatched campaign')
    checked=0;maximum=0.;hashes=0;files=0
    controls=report['controls'];path=args.campaign/'controls_predictions.npz'
    if file_sha256(path)!=controls['predictions_sha256']:raise ValueError('Controls changed')
    with np.load(path,allow_pickle=False) as archive:
        for name,table in controls['summaries'].items():
            count,error=verify_table(*metrics(archive[name],arrays['y'][validation]),arrays,validation,table)
            checked+=count;maximum=max(maximum,error)
    files+=1
    for row in report['runs']:
        checkpoint=args.campaign/row['checkpoint'];path=args.campaign/row['predictions']
        if file_sha256(checkpoint)!=row['checkpoint_sha256'] or file_sha256(path)!=row['predictions_sha256']:raise ValueError('Fit artifact changed')
        hashes+=1;files+=1
        with np.load(path,allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['indices'],validation)
            for phase,table in (('initial',row['initial']),('final',row['summary'])):
                count,error=verify_table(*metrics(archive[phase],arrays['y'][validation]),arrays,validation,table)
                checked+=count;maximum=max(maximum,error)
        if hashes%24==0:print(json.dumps(dict(fits_replayed=hashes,of=len(report['runs']))),flush=True)
    receipt=dict(per_subject_and_source_aggregates_recomputed=checked,checkpoint_hashes_verified=hashes,prediction_archives_verified=files,
        maximum_absolute_metric_difference=maximum,neural_forwards=0,checkpoint_loads=0,no_test_array_access=True,
        split_sha256=data['split_sha256'],source_results_sha256=file_sha256(args.campaign/'results.json'))
    with (args.campaign/'independent_verification.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
