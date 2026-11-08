"""Fixed five-fold development confirmation and seed replication of screen ideas."""
import argparse
import json
from pathlib import Path
import time
import subprocess

import numpy as np
import torch

from scripts.run_waveform_screen import batch_tensors,forward,infer,summarize,SEED
from rppg_lab.artifacts import file_sha256
from rppg_lab.waveform_v2_screen_models import AdaptiveWaveformNet
from rppg_lab.waveform_v2_screen_losses import training_loss,standardized,spectral_distance,aligned_components
from rppg_lab.waveform_v2_frozen_cache import clean

CONFIGS=[
    dict(name='morph_mix20',kind='gate_residual',loss='morphology',synthetic=.2,context=False,width=16),
    dict(name='morph_real_only',kind='gate_residual',loss='morphology',synthetic=0.,context=False,width=16),
    dict(name='morph_mix50',kind='gate_residual',loss='morphology',synthetic=.5,context=False,width=16),
    dict(name='native_mix20',kind='gate_residual',loss='native',synthetic=.2,context=False,width=16),
    dict(name='morph_context',kind='gate_residual',loss='morphology',synthetic=.2,context=True,width=16),
    dict(name='morph_wider32',kind='gate_residual',loss='morphology',synthetic=.2,context=False,width=32),
    dict(name='spectral_then_morphology',kind='gate_residual',loss='spectral_curriculum',synthetic=.2,context=False,width=16),
    dict(name='dual_band_morphology',kind='dual_band',loss='morphology',synthetic=.2,context=False,width=16),
]


def fit_one(configuration,arrays,fit,validation,device,seed,steps,model_factory=AdaptiveWaveformNet):
    assert set(arrays['record_id'][fit]).isdisjoint(arrays['record_id'][validation])
    torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    augmentation_rng=np.random.default_rng(seed+771)
    pools={source:{s:fit[(arrays['source'][fit]==source)&(arrays['record_id'][fit]==s)]
        for s in sorted(set(arrays['record_id'][fit][arrays['source'][fit]==source]))} for source in (0,1)}
    model=model_factory(configuration['kind'],context=configuration['context'],width=configuration['width']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
    initial=infer(model,arrays,validation,device)
    curve=[]
    for step in range(steps):
        indices=[]
        for _ in range(16):
            source=int(rng.random()<configuration['synthetic'])
            subject=rng.choice(list(pools[source]));indices.append(int(rng.choice(pools[source][subject])))
        batch=batch_tensors(arrays,np.asarray(indices),device)
        if configuration.get('prior_dropout',0):
            drop=torch.as_tensor(augmentation_rng.random((len(indices),3,6))<configuration['prior_dropout'],device=device)
            batch['prior']=batch['prior']&~drop
            empty=~batch['prior'].any((1,2))
            batch['prior'][empty]=torch.as_tensor(arrays['prior'][np.asarray(indices)[empty.cpu().numpy()]],device=device)
            batch['descriptors'][:,:72]=batch['descriptors'][:,:72].reshape(len(indices),18,4).mul(batch['prior'].reshape(len(indices),18,1)).reshape(len(indices),72)
        model.train();optimizer.zero_grad(set_to_none=True)
        clean_output=None
        if configuration.get('roi_dropout',False):
            if configuration.get('roi_consistency',0):
                with torch.no_grad():clean_output=forward(model,batch)['ppg'].detach()
            dropped={key:value.clone() for key,value in batch.items()}
            remove=augmentation_rng.random(len(indices))<.5
            regions=augmentation_rng.integers(0,3,len(indices))
            for row in np.flatnonzero(remove):
                region=int(regions[row]);dropped['x'][row,region]=0
                dropped['roi'][row,region]=False;dropped['prior'][row,region]=False
                dropped['descriptors'][row,24*region:24*(region+1)]=0
                dropped['descriptors'][row,72+region]=0
            batch=dropped
        output=forward(model,batch)
        if configuration['loss']=='spectral_curriculum':
            p,y=standardized(output['ppg']),standardized(batch['y'])
            shape,d1=aligned_components(p,y)
            weight=max(0,min(1,(step/max(steps-1,1)-.25)/.5))
            loss=spectral_distance(p,y)+weight*(1-shape.mean()+.2*(1-d1.mean()))
            loss=loss+1e-4*output['residual'].square().mean()
        else:loss=training_loss(output,batch['y'],configuration['loss'],step,steps)
        if clean_output is not None:
            # A stop-gradient full-input prediction is an engineering consistency
            # target, not a physiological phase label. Native waveform MSE
            # penalizes both shape and phase changes under ROI removal.
            loss=loss+configuration['roi_consistency']*(standardized(output['ppg'])-standardized(clean_output)).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite confirmation loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if step in (0,steps//3,2*steps//3,steps-1):curve.append(dict(step=step+1,loss=float(loss.detach())))
    predictions=infer(model,arrays,validation,device)
    model=model.cpu();del optimizer
    if device.type=='cuda':torch.cuda.empty_cache()
    return model,initial,predictions,curve


def baselines(arrays,fit,validation):
    sign=np.sign(np.mean([arrays['x'][i,:,3:,:].reshape(18,300)@arrays['y'][i]/300 for i in fit if arrays['source'][i]==0],axis=0))
    sign[sign==0]=1
    output={name:[] for name in ('CHROM_WIN','quality_weighted','quality_hard')}
    for i in validation:
        p=arrays['x'][i,:,3:,:].reshape(18,300);valid=(arrays['prior'][i]&arrays['roi'][i,:,None]).reshape(18)
        selected=valid&(np.arange(18)%6==2)
        output['CHROM_WIN'].append(p[selected].mean(0))
        q=arrays['descriptors'][i,:72].reshape(18,4)[:,0].clip(0,None)*valid
        output['quality_weighted'].append(((q*sign)[:,None]*p).sum(0)/max(q.sum(),1e-8))
        output['quality_hard'].append(sign[np.argmax(q)]*p[np.argmax(q)])
    return {k:np.asarray(v) for k,v in output.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--max-seconds',type=int,default=900)
    args=parser.parse_args();root=Path.cwd();out=args.out.absolute();data=args.data.resolve()
    allowed=(root/'outputs/waveform_screen').resolve()
    if allowed not in out.resolve().parents or allowed not in data.parents or out.exists():parser.error('Use a new screen output directory')
    receipt=json.loads((data/'data_receipt.json').read_text())
    if file_sha256(data/'development_arrays.npz')!=receipt['arrays_sha256']:raise ValueError('Prepared data changed')
    with np.load(data/'development_arrays.npz',allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    ids=sorted(set(arrays['record_id'][arrays['source']==0]));folds=[list(x) for x in np.array_split(np.random.default_rng(SEED).permutation(ids),5)]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out.mkdir(parents=True,exist_ok=False)
    lock=dict(configurations=CONFIGS,folds=folds,steps=300,seed=SEED,data_sha256=receipt['arrays_sha256'],
        scamps_scope='All eight ORIGINAL Train clips enter fitting; no synthetic validation efficacy claimed',
        ensembles='Equal mean and pointwise median of morph_mix20, morph_context, dual_band_morphology, without alignment/sign repair',
        seed_replications='Original first-screen split: base residual/dual-band morphology at seeds +17/+39, 120 steps')
    (out/'configuration_lock.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
    started=time.perf_counter();saved={};all_indices={};runs=[];per_method={};replications=[]
    for fold,held in enumerate(folds):
        validation=np.flatnonzero(np.isin(arrays['record_id'],held));fit=np.flatnonzero(~np.isin(arrays['record_id'],held))
        predictions=baselines(arrays,fit,validation)
        for config in CONFIGS:
            if time.perf_counter()-started>args.max_seconds:break
            before=time.perf_counter();model,initial,prediction,curve=fit_one(config,arrays,fit,validation,device,SEED,300)
            checkpoint=out/f"fold{fold}_{config['name']}.pt"
            torch.save(dict(model_state=model.state_dict(),configuration=config,seed=SEED,steps=300,
                validation_ids=held,data_sha256=receipt['arrays_sha256']),checkpoint)
            predictions[config['name']]=prediction
            summary,_=summarize(prediction,arrays,validation)
            runs.append(dict(fold=fold,configuration=config,steps=300,fit_curve=curve,summary=summary,
                runtime_seconds=time.perf_counter()-before,checkpoint_sha256=file_sha256(checkpoint)))
            print(json.dumps(clean(dict(fold=fold+1,name=config['name'],seconds=time.perf_counter()-before,UBFC=summary['UBFC']['subject_balanced']))),flush=True)
        ensemble_names=('morph_mix20','morph_context','dual_band_morphology')
        if all(name in predictions for name in ensemble_names):
            stack=np.stack([predictions[name] for name in ensemble_names])
            stack=stack-stack.mean(-1,keepdims=True);stack=stack/(stack.std(-1,keepdims=True)+1e-8)
            predictions['fixed_mean_ensemble']=stack.mean(0);predictions['fixed_median_ensemble']=np.median(stack,axis=0)
        for name,prediction in predictions.items():
            saved[f'fold{fold}_{name}']=prediction;all_indices[f'fold{fold}_{name}']=validation
            per_method.setdefault(name,[]).append((validation,prediction))
        (out/'partial_runs.json').write_text(json.dumps(clean(runs),indent=2,allow_nan=False)+'\n',encoding='utf-8')
        if len(runs)<8*(fold+1):break
    # Replicate the original six-real/two-synthetic validation screen at the same
    # 120-step budget, without introducing fresh holdout evidence.
    if len(runs)==40:
        validation=np.flatnonzero(arrays['validation']);fit=np.flatnonzero(~arrays['validation'])
        for config in (CONFIGS[0],CONFIGS[-1]):
            for seed in (SEED+17,SEED+39):
                if time.perf_counter()-started>args.max_seconds:break
                model,initial,prediction,curve=fit_one(config,arrays,fit,validation,device,seed,120)
                key=config['name']+f'__replication_seed{seed}'
                saved[key]=prediction;all_indices[key]=validation
                summary,_=summarize(prediction,arrays,validation)
                replications.append(dict(name=key,configuration=config,seed=seed,steps=120,summary=summary))
                torch.save(dict(model_state=model.state_dict(),configuration=config,seed=seed,steps=120,
                    data_sha256=receipt['arrays_sha256']),out/(key+'.pt'))
                print(json.dumps(clean(dict(replication=key,UBFC=summary['UBFC']['subject_balanced']))),flush=True)
    summaries={}
    for name,parts in per_method.items():
        indices=np.concatenate([i for i,_ in parts]);predictions=np.concatenate([p for _,p in parts])
        assert len(np.unique(indices))==len(indices)
        summary,_=summarize(predictions,arrays,indices)
        summaries[name]=dict(complete_real_cohort=len(indices)==152,summary=summary)
    np.savez_compressed(out/'predictions.npz',**saved)
    np.savez_compressed(out/'prediction_indices.npz',**all_indices)
    report=dict(status='DEVELOPMENT_CONFIRMATION_NOT_FINAL_TEST',complete=len(runs)==40 and len(replications)==4,
        intended_neural_runs=40,completed_neural_runs=len(runs),seed_replications=replications,
        data_sha256=receipt['arrays_sha256'],runtime_seconds=time.perf_counter()-started,device=str(device),
        folds=folds,summaries=summaries,runs=runs,no_original_val_test=True,no_private_data=True,
        predictions_sha256=file_sha256(out/'predictions.npz'),indices_sha256=file_sha256(out/'prediction_indices.npz'),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (out/'results.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(clean(dict(complete=report['complete'],seconds=report['runtime_seconds'],summaries=summaries)),indent=2))


if __name__=='__main__':main()
