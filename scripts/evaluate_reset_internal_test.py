"""Lock validation selection, then evaluate its fixed public internal Test.

This benchmark is historically exposed. The consumption receipt records a
single evaluation campaign, not an independent external validation claim.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from rppg_lab.artifacts import file_sha256
from rppg_lab.waveform_research_split import json_sha256
from rppg_lab.waveform_v2_frozen_cache import clean
from rppg_lab.waveform_v2_screen_models import AdaptiveWaveformNet
from rppg_lab.waveform_v2_extended_models import make_extended_model
from rppg_lab.waveform_feature_specialists import ThreePartFeatureModel,targets
from scripts.run_reset_waveform_campaign import fixed_controls
from scripts.run_waveform_screen import infer,summarize
from scripts.run_three_part_features import (training_weights,statistics,input_features,infer as infer_features,summarize_features)
from scripts.verify_reset_waveform_results import metrics,verify_table


def selection(waveforms,features,data):
    w=json.loads((waveforms/'results.json').read_text());f=json.loads((features/'results.json').read_text())
    if not w['complete'] or not f['complete'] or not w['split_sha256']==f['split_sha256']==data['split_sha256']:
        raise ValueError('Need completed matched validation studies')
    for root in (waveforms,features):
        if not (root/'independent_verification.json').exists():raise ValueError('Selection needs numerical replay first')
    groups={}
    for row in w['runs']:
        if row['job']['scope']=='full_train' and not row['job']['configuration']['shuffle']:
            groups.setdefault(row['job']['configuration']['name'],[]).append(row)
    hr_limit=w['controls']['summaries']['CHROM_WIN']['UBFC']['subject_balanced']['hr_mae']+.5
    eligible={}
    for name,rows in groups.items():
        if len(rows)!=3 or any(r['summary']['UBFC']['valid_predictions']!=r['summary']['UBFC']['intended_windows'] for r in rows):continue
        values=[r['summary']['UBFC']['subject_balanced'] for r in rows]
        if np.mean([v['hr_mae'] for v in values])<=hr_limit:
            eligible[name]=float(np.mean([.5*(v['morphology_r']+v['morphology_d1_r']) for v in values]))
    preferred=max(sorted(eligible),key=lambda name:eligible[name]);families=sorted(set((preferred,'long_tcn__roi_consistency0_2')))
    scores={mode:float(np.mean([np.mean(list(f['summaries'][r['name']]['UBFC']['group_scaled_errors'].values()))
        for r in f['runs'] if r['mode']==mode])) for mode in ('independent','joint','shared_heads')}
    wave_rows=[r for r in w['runs'] if r['job']['scope']=='full_train' and r['job']['configuration']['name'] in families]
    return dict(status='LOCKED_INTERNAL_TEST_SELECTION',split_sha256=data['split_sha256'],
        waveform_validation_results_sha256=file_sha256(waveforms/'results.json'),feature_validation_results_sha256=file_sha256(features/'results.json'),
        waveform_primary=preferred,waveform_families=families,waveform_hr_eligibility_limit_bpm=hr_limit,
        waveform_selection='Maximum three-seed mean of (aligned shape r + aligned d1 r)/2 on real Validation, finite complete coverage and HR <= CHROM + 0.5 bpm',
        feature_validation_preferred_mode=min(scores,key=scores.get),feature_mode_validation_scores=scores,
        feature_test_comparators='All three fixed matched-capacity modes, each at the same three seeds; no Test-based mode/feature selection',
        waveform_checkpoints=[dict(key=r['key'],job=r['job'],checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256']) for r in wave_rows],
        feature_checkpoints=[dict(name=r['name'],mode=r['mode'],width=r['width'],seed=r['seed'],checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256']) for r in f['runs']],
        fixed_waveform_controls=sorted(w['controls']['summaries']),no_refitting_to_test=True,no_private_data=True,
        exposure='Historically exposed public internal benchmark; not an independent new domain',
        source_sha256=file_sha256(Path(__file__)),git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())


def load_test_data(data,receipt):
    parts=[]
    for role in ('train','test'):
        path=data/(role+'_arrays.npz')
        if file_sha256(path)!=receipt['partitions'][role]['sha256']:raise ValueError('Partition changed')
        with np.load(path,allow_pickle=False) as archive:parts.append({k:archive[k] for k in archive.files})
    if set(parts[0]['record_id'])&set(parts[1]['record_id']):raise ValueError('Train/Test identities overlap')
    arrays={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return arrays,np.arange(len(parts[0]['x'])),np.arange(len(parts[0]['x']),len(arrays['x']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','waveforms','features','selection'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--lock-only',action='store_true');parser.add_argument('--out',type=Path);args=parser.parse_args()
    namespace=(Path.cwd()/'outputs/waveform_reset').resolve()
    if any(namespace not in p.resolve().parents for p in (args.data,args.waveforms,args.features)):parser.error('Use reset artifacts')
    data=json.loads((args.data/'data_receipt.json').read_text())
    if args.lock_only:
        lock=selection(args.waveforms,args.features,data)
        with args.selection.open('x',encoding='utf-8') as stream:json.dump(lock,stream,indent=2);stream.write('\n')
        print(json.dumps(dict(waveform_primary=lock['waveform_primary'],waveform_families=lock['waveform_families'],
            feature_preferred=lock['feature_validation_preferred_mode'],waveform_checkpoints=len(lock['waveform_checkpoints']),feature_checkpoints=len(lock['feature_checkpoints']))));return
    lock=json.loads(args.selection.read_text())
    # The source is locked before the selection commit; a later docs-only commit
    # can change HEAD without changing the registered evaluator bytes.
    if file_sha256(Path(__file__))!=lock['source_sha256'] or lock['split_sha256']!=data['split_sha256']:raise ValueError('Selection/evaluator changed')
    if file_sha256(args.waveforms/'results.json')!=lock['waveform_validation_results_sha256'] or file_sha256(args.features/'results.json')!=lock['feature_validation_results_sha256']:
        raise ValueError('Validation evidence changed')
    if not args.out or namespace not in args.out.resolve().parents or args.out.exists():parser.error('Use a NEW evaluation output directory')
    for kind,root in (('waveform',args.waveforms),('feature',args.features)):
        for row in lock[kind+'_checkpoints']:
            if file_sha256(root/row['checkpoint'])!=row['sha256']:raise ValueError('Selected checkpoint changed')
    args.out.mkdir(parents=True,exist_ok=False)
    consumption=dict(status='INTERNAL_TEST_CONSUMPTION_STARTED',selection_sha256=json_sha256(lock),split_sha256=data['split_sha256'],
        test_arrays_sha256=data['partitions']['test']['sha256'],no_test_based_tuning=True,
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    with (args.data/'internal_test_consumption.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(consumption,output=str(args.out)),stream,indent=2);stream.write('\n')
    with (args.out/'test_consumption.json').open('x',encoding='utf-8') as stream:json.dump(consumption,stream,indent=2);stream.write('\n')
    started=time.perf_counter();arrays,fit,test=load_test_data(args.data,data)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    waveform_predictions=fixed_controls(arrays,fit,test);wave_summaries={};feature_predictions={};feature_flags={};feature_summaries={}
    feature_report=json.loads((args.features/'results.json').read_text());center=np.asarray(feature_report['center']);scale=np.asarray(feature_report['scale'])
    for row in lock['waveform_checkpoints']:
        path=args.waveforms/row['checkpoint']
        # Only owned, hash-verified checkpoints; NumPy scalar metadata requires
        # the trusted-artifact loader for the waveform training files.
        checkpoint=torch.load(path,map_location='cpu',weights_only=False);config=row['job']['configuration']
        if checkpoint['job']!=row['job']:raise ValueError('Checkpoint identity differs from selection')
        model=AdaptiveWaveformNet(config['kind'],context=config['context'],width=config['width']) if config['architecture']=='compact' else make_extended_model(config['architecture'],config['width'])
        model.load_state_dict(checkpoint['model_state'],strict=True);model.to(device)
        name=config['name']+'__seed'+str(row['job']['seed']);waveform_predictions[name]=infer(model,arrays,test,device)
        del model
    waveform_replay_count=0;waveform_replay_error=0.
    for name,prediction in waveform_predictions.items():
        summary,_=summarize(prediction,arrays,test);wave_summaries[name]=summary
        count,error=verify_table(*metrics(prediction,arrays['y'][test]),arrays,test,summary)
        waveform_replay_count+=count;waveform_replay_error=max(waveform_replay_error,error)
    truth,reference_valid=targets(arrays['y']);weights=training_weights(arrays,fit)
    checked_center,checked_scale=statistics(truth,reference_valid,weights)
    np.testing.assert_allclose(center,checked_center,rtol=1e-10,atol=1e-10);np.testing.assert_allclose(scale,checked_scale,rtol=1e-10,atol=1e-10)
    def record(name,prediction,valid):
        feature_predictions[name]=prediction;feature_flags[name]=valid
        feature_summaries[name]=summarize_features(prediction,valid,truth[test],reference_valid[test],arrays,test,scale)
    record('Train_mean',np.broadcast_to(center,(len(test),10)),np.ones((len(test),10),bool))
    x=input_features(arrays);mean=(x*weights[:,None]).sum(0);std=np.sqrt(((x-mean)**2*weights[:,None]).sum(0)).clip(.01)
    design=np.c_[(x-mean)/std,np.ones(len(x))];regression=[]
    for c in range(10):
        group=fit[reference_valid[fit,c]];w=weights[group];w=w/w.sum();a=design[group]
        beta=np.linalg.solve((a.T*w)@a+.01*np.eye(a.shape[1]),(a.T*w)@((truth[group,c]-center[c])/scale[c]))
        regression.append(design[test]@beta*scale[c]+center[c])
    record('input_feature_ridge',np.stack(regression,1),np.ones((len(test),10),bool))
    for name,prediction in waveform_predictions.items():
        if name in ('CHROM_WIN','quality_weighted') or '__seed' in name:
            value,valid=targets(prediction);record('waveform__'+name,value,valid)
    for row in lock['feature_checkpoints']:
        checkpoint=torch.load(args.features/row['checkpoint'],map_location='cpu',weights_only=True)
        model=ThreePartFeatureModel(row['mode'],row['width']);model.load_state_dict(checkpoint['model_state'],strict=True);model.to(device)
        value,valid=infer_features(model,arrays,test,center,scale,device);record(row['name'],value,valid);del model
    np.savez_compressed(args.out/'waveform_predictions.npz',**waveform_predictions)
    np.savez_compressed(args.out/'feature_predictions.npz',**feature_predictions)
    np.savez_compressed(args.out/'feature_prediction_valid.npz',**feature_flags)
    np.savez_compressed(args.out/'feature_targets.npz',target=truth,valid=reference_valid,fit=fit,validation=test,center=center,scale=scale)
    report=dict(status='LOCKED_PUBLIC_INTERNAL_TEST_NOT_EXTERNAL_GENERALIZATION',complete=True,
        waveform_summaries=wave_summaries,feature_summaries=feature_summaries,selection_sha256=json_sha256(lock),split_sha256=data['split_sha256'],
        partition=data['partitions']['test'],waveform_numeric_replay=dict(aggregates=waveform_replay_count,max_absolute_difference=waveform_replay_error),
        artifact_sha256={name:file_sha256(args.out/name) for name in ('waveform_predictions.npz','feature_predictions.npz','feature_prediction_valid.npz','feature_targets.npz')},
        seconds=time.perf_counter()-started,no_neural_fitting=True,no_test_based_tuning=True,no_private_data=True,
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (args.out/'results.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,waveform_methods=len(wave_summaries),feature_methods=len(feature_summaries),seconds=report['seconds'])))


if __name__=='__main__':main()
