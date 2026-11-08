"""From-scratch waveform registry on the user-authorized public-data reset.

Reads Train/Validation only. Per-fit artifacts allow interrupted work to resume
without silently discarding runs. The prior checkpoint is never loaded.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from rppg_lab.artifacts import file_sha256
from rppg_lab.waveform_research_split import load_development_arrays,RESET_SEED
from rppg_lab.waveform_v2_screen_models import AdaptiveWaveformNet
from rppg_lab.waveform_v2_extended_models import make_extended_model
from rppg_lab.waveform_v2_frozen_cache import clean
from scripts.run_waveform_screen import configurations,summarize
from scripts.confirm_waveform_candidates import CONFIGS,fit_one,baselines
from scripts.ablate_waveform_hybrid import QualityResidualNet

SEEDS=(20261108,20261125,20261147)


def registry():
    output=[];seen=set()
    def add(name,**changes):
        config=dict(architecture='compact',kind='gate_residual',width=16,loss='morphology',synthetic=.2,
            context=False,variant='normal',shuffle=False,prior_dropout=0.,roi_dropout=False,roi_consistency=0.)
        config.update(changes);key=json.dumps(config,sort_keys=True)
        if key not in seen:seen.add(key);output.append(dict(config,name=name))
    for c in configurations():
        add(c['name'],kind=c['kind'],loss=c['loss'],synthetic=c['synthetic'],context=c['context'],
            shuffle=c['shuffle'],prior_dropout=.15 if c['dropout'] else 0.)
    for c in CONFIGS:add(c['name'],**{k:v for k,v in c.items() if k!='name'})
    for architecture in ('long_tcn','physical_attention','spectral','temporal_gate'):
        for loss in ('native','morphology'):add(architecture+'__'+loss,architecture=architecture,loss=loss)
    for variant in ('global_learned_gate','fixed_CHROM_gate','quality_weighted_residual'):
        add(variant,variant=variant)
    add('shuffled_fit_waveforms',shuffle=True)
    for name,weight in (('long_tcn__roi_dropout',0.),('long_tcn__roi_consistency0_2',.2)):
        add(name,architecture='long_tcn',roi_dropout=True,roi_consistency=weight)
    if len(output)!=40:raise ValueError(f'Expected 40 unique mechanisms, got {len(output)}')
    return output


def fit_polarities(arrays,fit):
    people=sorted(set(arrays['record_id'][fit][arrays['source'][fit]==0]));values=[]
    for person in people:
        group=fit[arrays['record_id'][fit]==person]
        values.append(np.mean([arrays['x'][i,:,3:,:].reshape(18,300)@arrays['y'][i]/300 for i in group],axis=0))
    signs=np.sign(np.mean(values,axis=0));signs[signs==0]=1
    return signs


def fixed_controls(arrays,fit,validation):
    output=baselines(arrays,fit,validation)
    for channel,name in enumerate(('GREEN','CHROM','CHROM_WIN','PBV','POS_WIN','OMIT')):
        if name in output:continue
        values=[]
        for i in validation:
            valid=arrays['roi'][i]&arrays['prior'][i,:,channel]
            values.append(arrays['x'][i,valid,3+channel].mean(0) if valid.any() else np.zeros(300))
        output[name]=np.asarray(values)
    output['negative_GREEN']=-output['GREEN']
    # Subject-balanced native ridge controls with explicit source masses.
    for rgb in (False,True):
        selected=np.arange(27) if rgb else np.array([r*9+c for r in range(3) for c in range(3,9)])
        for synthetic in (0.,.5):
            gram=np.zeros((len(selected),len(selected)));rhs=np.zeros(len(selected))
            for source,mass in ((0,1-synthetic),(1,synthetic)):
                if not mass:continue
                people=sorted(set(arrays['record_id'][fit][arrays['source'][fit]==source]))
                for person in people:
                    group=fit[arrays['record_id'][fit]==person]
                    for index in group:
                        x=arrays['x'][index].reshape(27,300)[selected].T.astype(float)
                        w=mass/(len(people)*len(group)*300);gram+=w*(x.T@x);rhs+=w*(x.T@arrays['y'][index])
            beta=np.linalg.solve(gram+.01*np.eye(len(selected)),rhs)
            name=('RGB_priors' if rgb else 'priors')+('__two_source50' if synthetic else '__real_only')
            output[name]=np.stack([beta@arrays['x'][i].reshape(27,300)[selected] for i in validation])
    return output


def run_one(config,arrays,fit,validation,device,seed):
    view=dict(arrays);variant=config['variant']
    if variant=='global_learned_gate':view['descriptors']=np.broadcast_to(arrays['descriptors'][fit].mean(0),arrays['descriptors'].shape).copy()
    if config['shuffle']:
        view['y']=arrays['y'].copy();rng=np.random.default_rng(seed+991)
        for source in (0,1):
            group=fit[arrays['source'][fit]==source];view['y'][group]=arrays['y'][rng.permutation(group)]
        np.testing.assert_array_equal(view['y'][validation],arrays['y'][validation])
    sign=fit_polarities(arrays,fit)
    def factory(kind,context=False,width=16):
        if variant=='quality_weighted_residual':return QualityResidualNet(sign,width)
        model=AdaptiveWaveformNet(kind,context=context,width=width) if config['architecture']=='compact' else make_extended_model(config['architecture'],width)
        if variant=='fixed_CHROM_gate':
            for parameter in model.gate.parameters():parameter.requires_grad_(False)
        return model
    return fit_one(config,view,fit,validation,device,seed,300,model_factory=factory)


def jobs_for(arrays,fit):
    configs=registry();jobs=[dict(configuration=c,seed=seed,scope='full_train') for c in configs for seed in SEEDS]
    chosen=('gate_residual__morphology','raw_temporal__native','long_tcn__roi_consistency0_2')
    for name in chosen:
        config=next(c for c in configs if c['name']==name)
        for scope in ('21_real_subjects','nonoverlapping_windows'):
            for seed in SEEDS[:2]:jobs.append(dict(configuration=config,seed=seed,scope=scope))
    return jobs


def scope_indices(arrays,fit,scope):
    if scope=='full_train':return fit
    synthetic=fit[arrays['source'][fit]==1];real=fit[arrays['source'][fit]==0]
    if scope=='21_real_subjects':
        ids=sorted(set(arrays['record_id'][real]));subset=np.random.default_rng(RESET_SEED+2223).permutation(ids)[:21]
        real=real[np.isin(arrays['record_id'][real],subset)]
    elif scope=='nonoverlapping_windows':
        chosen=[]
        for identity in sorted(set(arrays['record_id'][real])):
            group=real[arrays['record_id'][real]==identity];first=arrays['start'][group].min()
            chosen.extend(group[np.rint((arrays['start'][group]-first)/2).astype(int)%5==0])
        real=np.asarray(chosen)
    else:raise ValueError('Unknown fitting scope')
    return np.sort(np.r_[real,synthetic])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=1800);args=parser.parse_args()
    allowed=(Path.cwd()/'outputs/waveform_reset').resolve();out=args.out.resolve()
    if any(allowed not in p.resolve().parents for p in (args.data,out)):parser.error('Use reset data/output paths')
    if out.exists() and not args.resume:parser.error('Use a NEW directory or explicit --resume')
    arrays,fit,validation,data=load_development_arrays(args.data)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');jobs=jobs_for(arrays,fit)
    source_names=('scripts/run_reset_waveform_campaign.py','scripts/confirm_waveform_candidates.py','rppg_lab/waveform_research_split.py',
        'rppg_lab/waveform_v2_screen_models.py','rppg_lab/waveform_v2_screen_losses.py','rppg_lab/waveform_v2_extended_models.py')
    lock=dict(jobs=jobs,steps=300,split_sha256=data['split_sha256'],partitions={r:data['partitions'][r] for r in ('train','validation')},
        source_sha256={p:file_sha256(p) for p in source_names},no_test_scoring=True)
    if out.exists():
        if json.loads((out/'configuration_lock.json').read_text())!=lock:raise ValueError('Resume configuration/source changed')
    else:
        out.mkdir(parents=True,exist_ok=False);(out/'configuration_lock.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
    started=time.perf_counter();runs=[]
    controls_path=out/'controls.json'
    if not controls_path.exists():
        controls=fixed_controls(arrays,fit,validation);np.savez_compressed(out/'controls_predictions.npz',**controls)
        summaries={name:summarize(p,arrays,validation)[0] for name,p in controls.items()}
        controls_path.write_text(json.dumps(clean(dict(summaries=summaries,predictions_sha256=file_sha256(out/'controls_predictions.npz'))),indent=2,allow_nan=False)+'\n')
    controls=json.loads(controls_path.read_text())
    for number,job in enumerate(jobs):
        key=f'run{number:03d}';row_path=out/(key+'.json')
        if row_path.exists():
            row=json.loads(row_path.read_text())
            if row['job']!=job or any(file_sha256(out/row[k])!=row[k+'_sha256'] for k in ('checkpoint','predictions')):raise ValueError('Completed fit changed')
            runs.append(row);continue
        if time.perf_counter()-started>args.max_seconds:break
        selected=scope_indices(arrays,fit,job['scope']);before=time.perf_counter()
        model,initial,predictions,curve=run_one(job['configuration'],arrays,selected,validation,device,job['seed'])
        checkpoint=out/(key+'.pt');prediction_path=out/(key+'_predictions.npz')
        torch.save(dict(model_state=model.state_dict(),job=job,split_sha256=data['split_sha256'],steps=300,
            validation_ids=sorted(set(arrays['record_id'][validation]))),checkpoint)
        np.savez_compressed(prediction_path,initial=initial,final=predictions,indices=validation)
        row=dict(key=key,job=job,summary=summarize(predictions,arrays,validation)[0],initial=summarize(initial,arrays,validation)[0],
            fit_curve=curve,seconds=time.perf_counter()-before,parameters=sum(p.numel() for p in model.parameters()),
            fitting_windows=len(selected),real_fitting_subjects=len(set(arrays['record_id'][selected][arrays['source'][selected]==0])),
            checkpoint=checkpoint.name,checkpoint_sha256=file_sha256(checkpoint),predictions=prediction_path.name,predictions_sha256=file_sha256(prediction_path))
        row_path.write_text(json.dumps(clean(row),indent=2,allow_nan=False)+'\n',encoding='utf-8');runs.append(row)
        print(json.dumps(clean(dict(completed=len(runs),intended=len(jobs),name=job['configuration']['name'],scope=job['scope'],seed=job['seed'],
            UBFC=row['summary']['UBFC']['subject_balanced'],SCAMPS=row['summary']['SCAMPS']['subject_balanced']))),flush=True)
        (out/'progress.json').write_text(json.dumps(dict(completed=len(runs),intended=len(jobs),elapsed_seconds=time.perf_counter()-started),indent=2)+'\n')
    report=dict(status='RESET_TRAIN_VALIDATION_DEVELOPMENT',complete=len(runs)==len(jobs),completed_fits=len(runs),intended_fits=len(jobs),
        runs=runs,controls=controls,split_sha256=data['split_sha256'],partitions=lock['partitions'],seconds_this_invocation=time.perf_counter()-started,
        no_test_scoring=True,no_prior_weight_loading=True,no_private_data=True,no_native60_training_labels=True,source_sha256=lock['source_sha256'],
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (out/'results.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=report['complete'],fits=len(runs),seconds=report['seconds_this_invocation'])))


if __name__=='__main__':main()
