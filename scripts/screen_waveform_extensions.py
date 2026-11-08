"""Bounded comparison of longer context, attention, spectral and temporal gates."""
import argparse
import json
from pathlib import Path
import time
import subprocess
import numpy as np
import torch

from scripts.confirm_waveform_candidates import fit_one,baselines
from scripts.run_waveform_screen import summarize,SEED
from rppg_lab.artifacts import file_sha256
from rppg_lab.waveform_v2_extended_models import make_extended_model
from rppg_lab.waveform_v2_frozen_cache import clean


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--confirmation',type=Path,help='Reuse completed five-fold development partition; morphology losses only')
    parser.add_argument('--roi-robustness',action='store_true',help='Two prespecified long-TCN dropout/consistency controls')
    parser.add_argument('--max-seconds',type=int,default=900);args=parser.parse_args()
    root=Path.cwd();allowed=(root/'outputs/waveform_screen').resolve();out=args.out.resolve()
    if allowed not in out.parents or allowed not in args.data.resolve().parents or out.exists():parser.error('Use screen data and NEW output')
    receipt=json.loads((args.data/'data_receipt.json').read_text())
    if file_sha256(args.data/'development_arrays.npz')!=receipt['arrays_sha256']:raise ValueError('Data changed')
    with np.load(args.data/'development_arrays.npz',allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    if args.confirmation:
        confirm_folds(args,arrays,receipt,out,allowed);return
    if args.roi_robustness:parser.error('ROI controls require --confirmation')
    validation=np.flatnonzero(arrays['validation']);fit=np.flatnonzero(~arrays['validation'])
    configs=[dict(name=architecture+'__'+loss,architecture=architecture,kind='gate_residual',width=16,
        context=False,synthetic=.2,loss=loss) for architecture in ('long_tcn','physical_attention','spectral','temporal_gate')
        for loss in ('native','morphology')]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out.mkdir(parents=True,exist_ok=False)
    (out/'configuration_lock.json').write_text(json.dumps(dict(configurations=configs,steps=120,seed=SEED,
        data_sha256=receipt['arrays_sha256'],development_only=True),indent=2)+'\n')
    started=time.perf_counter();results=[];predictions={}
    for config in configs:
        def factory(kind,context=False,width=16):return make_extended_model(config['architecture'],width)
        before=time.perf_counter();model,initial,prediction,curve=fit_one(config,arrays,fit,validation,device,SEED,120,model_factory=factory)
        summary,_=summarize(prediction,arrays,validation);initial_summary,_=summarize(initial,arrays,validation)
        path=out/(config['name']+'.pt');torch.save(dict(model_state=model.state_dict(),configuration=config,seed=SEED,
            steps=120,data_sha256=receipt['arrays_sha256']),path)
        predictions[config['name']]=prediction;predictions[config['name']+'__initial']=initial
        result=dict(configuration=config,summary=summary,initial=initial_summary,fit_curve=curve,
            parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-before,checkpoint_sha256=file_sha256(path))
        results.append(result)
        print(json.dumps(clean(dict(name=config['name'],seconds=result['seconds'],UBFC=summary['UBFC']['subject_balanced']))),flush=True)
        (out/'partial_results.json').write_text(json.dumps(clean(results),indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'predictions.npz',**predictions)
    report=dict(status='EXPLORATORY_ARCHITECTURE_SCREEN',complete=len(results)==8,results=results,
        data_sha256=receipt['arrays_sha256'],steps=120,seed=SEED,seconds=time.perf_counter()-started,
        no_original_val_test=True,no_private_data=True,no_native60_training_labels=True,
        predictions_sha256=file_sha256(out/'predictions.npz'),git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (out/'results.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=report['complete'],seconds=report['seconds']),indent=2))


def confirm_folds(args,arrays,receipt,out,allowed):
    if allowed not in args.confirmation.resolve().parents:raise ValueError('Confirmation must be a screen artifact')
    original=json.loads((args.confirmation/'results.json').read_text())
    if not original['complete'] or original['data_sha256']!=receipt['arrays_sha256']:raise ValueError('Need matched completed folds')
    folds=original['folds']
    configs=[dict(name=name+'__morphology',architecture=name,kind='gate_residual',width=16,
        context=False,synthetic=.2,loss='morphology') for name in ('long_tcn','physical_attention','spectral','temporal_gate')]
    if args.roi_robustness:
        configs=[dict(name=name,architecture='long_tcn',kind='gate_residual',width=16,context=False,
            synthetic=.2,loss='morphology',roi_dropout=True,roi_consistency=weight)
            for name,weight in (('long_tcn__roi_dropout',0.),('long_tcn__roi_consistency0_2',.2))]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out.mkdir(parents=True,exist_ok=False)
    (out/'configuration_lock.json').write_text(json.dumps(dict(configurations=configs,folds=folds,
        steps=300,seed=SEED,data_sha256=receipt['arrays_sha256'],development_only=True),indent=2)+'\n')
    started=time.perf_counter();runs=[];saved={};index_archive={};parts={}
    for fold,held in enumerate(folds):
        validation=np.flatnonzero(np.isin(arrays['record_id'],held));fit=np.flatnonzero(~np.isin(arrays['record_id'],held))
        predictions=baselines(arrays,fit,validation)
        for config in configs:
            if time.perf_counter()-started>args.max_seconds:break
            def factory(kind,context=False,width=16):return make_extended_model(config['architecture'],width)
            before=time.perf_counter();model,initial,prediction,curve=fit_one(config,arrays,fit,validation,device,SEED,300,model_factory=factory)
            key=f"fold{fold}_{config['name']}";path=out/(key+'.pt')
            torch.save(dict(model_state=model.state_dict(),configuration=config,seed=SEED,steps=300,
                validation_ids=held,data_sha256=receipt['arrays_sha256']),path)
            summary,_=summarize(prediction,arrays,validation);predictions[config['name']]=prediction
            runs.append(dict(fold=fold,configuration=config,summary=summary,fit_curve=curve,
                seconds=time.perf_counter()-before,checkpoint_sha256=file_sha256(path)))
            print(json.dumps(clean(dict(fold=fold+1,name=config['name'],UBFC=summary['UBFC']['subject_balanced']))),flush=True)
        for name,prediction in predictions.items():
            key=f'fold{fold}_{name}';saved[key]=prediction;index_archive[key]=validation
            parts.setdefault(name,[]).append((validation,prediction))
        (out/'partial_results.json').write_text(json.dumps(clean(runs),indent=2,allow_nan=False)+'\n')
        if len(runs)!=len(configs)*(fold+1):break
    summaries={}
    for name,items in parts.items():
        indices=np.concatenate([i for i,_ in items]);prediction=np.concatenate([p for _,p in items])
        assert len(set(indices))==len(indices)
        summaries[name]=dict(complete_real_cohort=len(indices)==152,summary=summarize(prediction,arrays,indices)[0])
    np.savez_compressed(out/'predictions.npz',**saved);np.savez_compressed(out/'prediction_indices.npz',**index_archive)
    report=dict(status='ROI_ROBUSTNESS_FIVE_FOLD_DEVELOPMENT' if args.roi_robustness else 'EXPLORATORY_ARCHITECTURE_FIVE_FOLD_DEVELOPMENT',complete=len(runs)==5*len(configs),
        intended_neural_runs=5*len(configs),completed_neural_runs=len(runs),runs=runs,summaries=summaries,folds=folds,
        data_sha256=receipt['arrays_sha256'],steps=300,seed=SEED,seconds=time.perf_counter()-started,
        no_original_val_test=True,no_private_data=True,no_native60_training_labels=True,
        predictions_sha256=file_sha256(out/'predictions.npz'),indices_sha256=file_sha256(out/'prediction_indices.npz'),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (out/'results.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=report['complete'],seconds=report['seconds']),indent=2))


if __name__=='__main__':main()
