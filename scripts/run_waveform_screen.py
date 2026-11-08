"""Fixed compact adaptive/model/loss screen on existing development sources."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from rppg_lab.artifacts import file_sha256
from rppg_lab.waveform_v2_screen_models import AdaptiveWaveformNet
from rppg_lab.waveform_v2_screen_losses import training_loss,aligned_components,pearson
from rppg_lab.waveform_v2_classical_study import correlation,finite_mean
from rppg_lab.signals import standardize_1d,estimate_hr_welch
from rppg_lab.waveform_v2_frozen_cache import clean

KINDS=('gate','gate_residual','raw_temporal','dual_band')
LOSSES=('native','morphology','derivative_spectral','staged')
SEED=20261108


def configurations():
    runs=[dict(name=k+'__'+loss,kind=k,loss=loss,synthetic=.2,context=False,dropout=False,shuffle=False)
          for k,loss in itertools.product(KINDS,LOSSES)]
    base=dict(kind='gate_residual',loss='staged',synthetic=.2,context=False,dropout=False,shuffle=False)
    for name,changes in [('context',dict(context=True)),('real_only',dict(synthetic=0.)),
                         ('synthetic_half',dict(synthetic=.5)),('shuffled_targets',dict(shuffle=True)),
                         ('mask_dropout',dict(dropout=True))]:
        runs.append(dict(base,**{'name':'gate_residual__staged__'+name},**changes))
    return runs


def waveform_metrics(p,y):
    if not np.isfinite(p).all() or np.std(p)<1e-8:
        return dict(valid=False,native_r=None,nrmse=None,derivative_r=None,morphology_r=None,morphology_d1_r=None,hr_mae=None)
    p=standardize_1d(p);y=standardize_1d(y)
    with torch.inference_mode():
        r,d1=aligned_components(torch.tensor(p)[None],torch.tensor(y)[None])
    return dict(valid=True,native_r=correlation(p,y),nrmse=float(np.sqrt(np.mean((p-y)**2))),
        derivative_r=correlation(np.diff(p),np.diff(y)),morphology_r=float(r[0]),morphology_d1_r=float(d1[0]),
        hr_mae=abs(float(estimate_hr_welch(p,30).hr_bpm)-float(estimate_hr_welch(y,30).hr_bpm)))


def summarize(predictions,arrays,indices):
    rows=[]
    for j,index in enumerate(indices):
        rows.append(dict(id=str(arrays['record_id'][index]),source=int(arrays['source'][index]),
            start=float(arrays['start'][index]),metrics=waveform_metrics(predictions[j],arrays['y'][index])))
    result={}
    for source,name in [(0,'UBFC'),(1,'SCAMPS')]:
        selected=[r for r in rows if r['source']==source]
        ids=sorted({r['id'] for r in selected})
        fields=('native_r','nrmse','derivative_r','morphology_r','morphology_d1_r','hr_mae')
        subjects={s:{k:finite_mean([r['metrics'][k] for r in selected if r['id']==s and r['metrics'][k] is not None]) for k in fields} for s in ids}
        balanced={k:finite_mean([r[k] for r in subjects.values()]) for k in fields}
        result[name]=dict(intended_windows=len(selected),valid_predictions=sum(r['metrics']['valid'] for r in selected),
            subjects=len(ids),subject_balanced=balanced,per_subject=subjects)
    return result,rows


def batch_tensors(arrays,indices,device):
    return {k:torch.as_tensor(arrays[k][indices],device=device) for k in
        ('x','y','roi','prior','descriptors','context','context_valid')}


def forward(model,batch):
    return model(batch['x'],batch['roi'],batch['prior'],batch['descriptors'],batch['context'],batch['context_valid'])


def infer(model,arrays,indices,device):
    if not len(indices):return np.empty((0,arrays['x'].shape[-1]),dtype=np.float32)
    model.eval();predictions=[]
    with torch.inference_mode():
        for start in range(0,len(indices),32):
            predictions.append(forward(model,batch_tensors(arrays,indices[start:start+32],device))['ppg'].cpu().numpy())
    return np.concatenate(predictions)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=120);parser.add_argument('--max-seconds',type=int,default=900)
    args=parser.parse_args();root=Path.cwd();out=args.out.absolute();data=args.data.resolve()
    namespace=(root/'outputs/waveform_screen').resolve()
    if namespace not in out.resolve().parents or namespace not in data.parents or out.exists():
        parser.error('Use explicit screen data and NEW output directory inside outputs/waveform_screen/')
    if args.steps!=120:parser.error('Fixed first screen uses exactly 120 steps; revise protocol for a new campaign')
    receipt=json.loads((data/'data_receipt.json').read_text())
    if file_sha256(data/'development_arrays.npz')!=receipt['arrays_sha256']:raise ValueError('Prepared inputs changed')
    with np.load(data/'development_arrays.npz',allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    validation=np.flatnonzero(arrays['validation']);fit=np.flatnonzero(~arrays['validation'])
    assert set(arrays['record_id'][fit]).isdisjoint(arrays['record_id'][validation])
    for record_id in set(arrays['record_id']):
        group=np.flatnonzero(arrays['record_id']==record_id)
        first=group[np.argmin(arrays['start'][group])]
        for index in group:
            expected=arrays['start'][index]>=arrays['start'][first]+10-1e-8
            assert bool(arrays['context_valid'][index])==expected
            np.testing.assert_array_equal(arrays['context'][index],arrays['descriptors'][first] if expected else np.zeros(75,np.float32))
    train_hashes={__import__('hashlib').sha256(arrays['x'][i].tobytes()).hexdigest() for i in fit}
    validation_hashes={__import__('hashlib').sha256(arrays['x'][i].tobytes()).hexdigest() for i in validation}
    if train_hashes&validation_hashes:raise ValueError('Exact duplicate input crosses development split')
    pools={source:{s:fit[(arrays['source'][fit]==source)&(arrays['record_id'][fit]==s)]
        for s in sorted(set(arrays['record_id'][fit][arrays['source'][fit]==source]))} for source in (0,1)}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out.mkdir(parents=True,exist_ok=False);registry=configurations()
    (out/'configuration_lock.json').write_text(json.dumps(dict(configurations=registry,steps=120,seed=SEED,
        data_sha256=receipt['arrays_sha256'],validation_ids=sorted(set(arrays['record_id'][validation])),
        device=str(device),third_dataset_used=False),indent=2)+'\n',encoding='utf-8')
    started=time.perf_counter();results=[];prediction_archive={};metrics_archive={}
    # Fit-only global polarity for the transparent quality-based baselines.
    signed=np.sign(np.mean([arrays['x'][i,:,3:,:].reshape(18,300)@arrays['y'][i]/300 for i in fit if arrays['source'][i]==0],axis=0))
    signed[signed==0]=1
    for name in ('CHROM_WIN','GREEN','quality_hard','quality_weighted'):
        values=[]
        for i in validation:
            priors=arrays['x'][i,:,3:,:].reshape(18,300)
            valid=(arrays['prior'][i]&arrays['roi'][i,:,None]).reshape(18)
            if name in ('CHROM_WIN','GREEN'):
                channel=2 if name=='CHROM_WIN' else 0
                use=np.arange(18)%6==channel;use&=valid
                pred=priors[use].mean(0) if use.any() else np.zeros(300)
            else:
                score=arrays['descriptors'][i,:72].reshape(18,4)[:,0].clip(0,None)*valid
                if name=='quality_hard':pred=signed[np.argmax(score)]*priors[np.argmax(score)]
                else:pred=((score*signed)[:,None]*priors).sum(0)/max(score.sum(),1e-8)
            values.append(pred)
        values=np.asarray(values);summary,rows=summarize(values,arrays,validation)
        prediction_archive[name]=values;results.append(dict(name=name,learned=False,final=summary))
        metrics_archive[name]=rows
    for configuration in registry:
        if time.perf_counter()-started>args.max_seconds:
            print('Local time budget reached; remaining configurations are not claimed tested',flush=True);break
        run_start=time.perf_counter();random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
        if device.type=='cuda':torch.cuda.manual_seed_all(SEED)
        rng=np.random.default_rng(SEED)
        model=AdaptiveWaveformNet(configuration['kind'],context=configuration['context']).to(device)
        initial=infer(model,arrays,validation,device);initial_summary,_=summarize(initial,arrays,validation)
        prediction_archive[configuration['name']+'__initial']=initial
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
        curve=[]
        for step in range(120):
            indices=[]
            for _ in range(16):
                source=int(rng.random()<configuration['synthetic'])
                subject=rng.choice(list(pools[source]));indices.append(int(rng.choice(pools[source][subject])))
            indices=np.asarray(indices);batch=batch_tensors(arrays,indices,device)
            if configuration['shuffle']:
                batch['y']=torch.as_tensor(arrays['y'][rng.choice(fit,16)],device=device)
            if configuration['dropout']:
                drop=torch.as_tensor(rng.random((16,3,6))<.15,device=device)
                batch['prior']=batch['prior']&~drop
                empty=~batch['prior'].any((1,2))
                batch['prior'][empty]=torch.as_tensor(arrays['prior'][indices[empty.cpu().numpy()]],device=device)
                available=batch['prior'].reshape(16,18)
                batch['descriptors'][:,:72]=batch['descriptors'][:,:72].reshape(16,18,4).mul(available[...,None]).reshape(16,72)
            model.train();optimizer.zero_grad(set_to_none=True)
            output=forward(model,batch);loss=training_loss(output,batch['y'],configuration['loss'],step,120)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss for '+configuration['name'])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            if step in (0,39,79,119):curve.append(dict(step=step+1,fit_loss=float(loss.detach())))
        prediction=infer(model,arrays,validation,device);summary,rows=summarize(prediction,arrays,validation)
        checkpoint=out/(configuration['name']+'.pt')
        torch.save(dict(model_state=model.cpu().state_dict(),configuration=configuration,training_rate_hz=30,
            validation_ids=sorted(set(arrays['record_id'][validation])),data_sha256=receipt['arrays_sha256'],seed=SEED),checkpoint)
        prediction_archive[configuration['name']]=prediction;metrics_archive[configuration['name']]=rows
        result=dict(configuration,learned=True,steps=120,parameters=sum(p.numel() for p in model.parameters()),
            initial=initial_summary,final=summary,fit_curve=curve,runtime_seconds=time.perf_counter()-run_start,checkpoint_sha256=file_sha256(checkpoint))
        results.append(result)
        print(json.dumps(clean(dict(name=configuration['name'],seconds=result['runtime_seconds'],UBFC=summary['UBFC']['subject_balanced']))),flush=True)
        (out/'partial_results.json').write_text(json.dumps(clean(results),indent=2,allow_nan=False)+'\n',encoding='utf-8')
        del optimizer,model
        if device.type=='cuda':torch.cuda.empty_cache()
    np.savez_compressed(out/'validation_predictions.npz',**prediction_archive)
    report=dict(status='DEVELOPMENT_SCREEN_NOT_FINAL_TEST',complete=len([r for r in results if r['learned']])==len(registry),
        intended_configurations=len(registry),trained_configurations=sum(r['learned'] for r in results),
        seed=SEED,steps_per_configuration=120,device=str(device),gpu_name=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        data_sha256=receipt['arrays_sha256'],results=results,window_metrics=metrics_archive,
        no_original_val_test=True,no_private_data=True,no_native60_training_labels=True,
        runtime_seconds=time.perf_counter()-started,git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={p:file_sha256(p) for p in ('scripts/run_waveform_screen.py','rppg_lab/waveform_v2_screen_models.py','rppg_lab/waveform_v2_screen_losses.py','rppg_lab/waveform_v2_screen_features.py')})
    (out/'results.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=report['complete'],trained=report['trained_configurations'],seconds=report['runtime_seconds']),indent=2))


if __name__=='__main__':main()
