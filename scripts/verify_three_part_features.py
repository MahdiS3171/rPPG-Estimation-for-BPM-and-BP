"""Independently replay saved feature statistics/errors without neural execution."""
import argparse
import json
from pathlib import Path
import numpy as np
from rppg_lab.artifacts import file_sha256
from rppg_lab.waveform_research_split import load_development_arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True);args=parser.parse_args()
    arrays,fit,validation,data=load_development_arrays(args.data);root=args.results
    report=json.loads((root/'results.json').read_text())
    if not report['complete'] or report['split_sha256']!=data['split_sha256']:raise ValueError('Need completed matched feature results')
    for filename,key in (('predictions.npz','predictions_sha256'),('prediction_valid.npz','validity_sha256'),('feature_targets.npz','targets_sha256')):
        if file_sha256(root/filename)!=report[key]:raise ValueError('Feature artifact changed')
    with np.load(root/'feature_targets.npz',allow_pickle=False) as archive:
        target=archive['target'];valid=archive['valid'];center=archive['center'];scale=archive['scale']
        np.testing.assert_array_equal(archive['fit'],fit);np.testing.assert_array_equal(archive['validation'],validation)
    weights=np.zeros(len(target))
    for source,mass in ((0,.8),(1,.2)):
        people=sorted(set(arrays['record_id'][fit][arrays['source'][fit]==source]))
        for person in people:
            indices=fit[arrays['record_id'][fit]==person];weights[indices]=mass/len(people)/len(indices)
    mass=(weights[:,None]*valid).sum(0);mean=(weights[:,None]*valid*target).sum(0)/mass
    variance=(weights[:,None]*valid*(target-mean)**2).sum(0)/mass
    expected_scale=np.maximum(np.sqrt(variance),[1.,.01,.01,.01,.01,.01,.01,.5,.5,5.])
    np.testing.assert_allclose(center,mean,rtol=1e-10,atol=1e-10);np.testing.assert_allclose(scale,expected_scale,rtol=1e-10,atol=1e-10)
    checked=0;max_error=0.
    def check(actual,want):
        nonlocal checked,max_error
        if want is None:
            if actual is not None:raise ValueError('Unexpected defined feature score')
            return
        difference=abs(float(actual)-float(want));max_error=max(max_error,difference)
        if not np.isfinite(difference) or difference>1e-8:raise ValueError('Feature replay mismatch')
        checked+=1
    target=target[validation];valid=valid[validation]
    with np.load(root/'predictions.npz',allow_pickle=False) as predictions,np.load(root/'prediction_valid.npz',allow_pickle=False) as flags:
        for method,table in report['summaries'].items():
            prediction=predictions[method];available=flags[method]&np.isfinite(prediction)
            if prediction.shape!=target.shape or available.shape!=valid.shape:raise ValueError('Feature denominator changed')
            for source,name in ((0,'UBFC'),(1,'SCAMPS')):
                indices=np.flatnonzero(arrays['source'][validation]==source);ids=arrays['record_id'][validation]
                people=sorted(set(ids[indices]));entry=table[name];scaled=[]
                check(len(people),entry['subjects']);check(len(indices),entry['intended_windows'])
                for column,feature in enumerate(report['feature_names']):
                    row=entry['features'][feature];ref=valid[indices,column];good=available[indices,column]
                    bad=prediction[indices,column]<0
                    if column in (3,4,5,6):bad|=prediction[indices,column]>1
                    if column==0:bad|=(prediction[indices,column]<42)|(prediction[indices,column]>210)
                    for actual,key in ((len(indices),'intended_windows'),(ref.sum(),'reference_valid'),((ref&good).sum(),'paired_valid'),
                        ((ref&~good).sum(),'missing_prediction_for_valid_reference'),((good&bad).sum(),'invalid_physical_values')):check(actual,row[key])
                    means=[];penalties=[]
                    for person in people:
                        group=indices[ids[indices]==person];observed=valid[group,column];paired=observed&available[group,column]
                        absolute=abs(prediction[group,column]-target[group,column])
                        if paired.any():means.append(float(absolute[paired].mean()))
                        if observed.any():
                            penalty=[float(absolute[j]/scale[column]) if available[group[j],column] else 5. for j in np.flatnonzero(observed)]
                            penalties.append(float(np.mean(penalty)))
                    check(np.mean(means) if means else None,row['subject_balanced_mae'])
                    value=float(np.mean(penalties)) if penalties else None;check(value,row['subject_balanced_penalized_scaled_mae']);scaled.append(value)
                for group,positions in (('rhythm',range(3)),('geometry',range(3,7)),('derivatives',range(7,10))):
                    check(np.mean([scaled[p] for p in positions]),entry['group_scaled_errors'][group])
    for row in report['runs']:
        if file_sha256(root/row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Specialist checkpoint changed')
    receipt=dict(saved_scalar_values_recomputed=checked,maximum_absolute_difference=max_error,
        training_only_center_scale_verified=True,checkpoint_hashes_verified=len(report['runs']),neural_forwards=0,checkpoint_loads=0,
        no_test_array_access=True,source_results_sha256=file_sha256(root/'results.json'),split_sha256=data['split_sha256'])
    with (root/'independent_verification.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
