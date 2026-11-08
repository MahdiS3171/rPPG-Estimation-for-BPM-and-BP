"""Compare independently trained feature groups with joint/shared-head controls."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from rppg_lab.artifacts import file_sha256
from rppg_lab.waveform_research_split import load_development_arrays
from rppg_lab.waveform_feature_specialists import (FEATURES,UNITS,GROUPS,STD_FLOORS,targets,
    ThreePartFeatureModel,matched_widths,feature_loss)
from rppg_lab.waveform_v2_frozen_cache import clean
from scripts.run_reset_waveform_campaign import SEEDS


def training_weights(arrays,fit):
    weights=np.zeros(len(arrays['x']))
    for source,mass in ((0,.8),(1,.2)):
        people=sorted(set(arrays['record_id'][fit][arrays['source'][fit]==source]))
        for person in people:
            group=fit[arrays['record_id'][fit]==person];weights[group]=mass/(len(people)*len(group))
    return weights


def statistics(values,valid,weights):
    mass=(valid*weights[:,None]).sum(0)
    if np.any(mass<=0):raise ValueError('A task has no valid training labels')
    center=(values*valid*weights[:,None]).sum(0)/mass
    scale=np.sqrt(((values-center)**2*valid*weights[:,None]).sum(0)/mass)
    return center,np.maximum(scale,STD_FLOORS)


def input_features(arrays):
    x=arrays['x'][:,:,3:,:].reshape(len(arrays['x']),18,300)
    power=(abs(np.fft.rfft(x,axis=-1))**2).mean(1);frequency=np.fft.rfftfreq(300,1/30)
    power=power[:,(frequency>=.5)&(frequency<=8)];power/=np.maximum(power.sum(-1,keepdims=True),1e-8)
    return np.c_[arrays['descriptors'],power]


def summarize_features(prediction,pred_valid,truth,target_valid,arrays,validation,scale):
    output={}
    for source,name in ((0,'UBFC'),(1,'SCAMPS')):
        positions=np.flatnonzero(arrays['source'][validation]==source);people=sorted(set(arrays['record_id'][validation[positions]]))
        rows={}
        for column,feature in enumerate(FEATURES):
            observed=target_valid[positions,column];available=pred_valid[positions,column]&np.isfinite(prediction[positions,column])
            physical_invalid=prediction[positions,column]<0
            if 3<=column<7:physical_invalid|=prediction[positions,column]>1
            if column==0:physical_invalid|=(prediction[positions,column]<42)|(prediction[positions,column]>210)
            error=abs(prediction[positions,column]-truth[positions,column]);means=[];penalties=[]
            for person in people:
                person_positions=positions[arrays['record_id'][validation[positions]]==person]
                ref=target_valid[person_positions,column]
                if not ref.any():continue
                good=pred_valid[person_positions,column]&np.isfinite(prediction[person_positions,column])
                absolute=abs(prediction[person_positions,column]-truth[person_positions,column])
                if (ref&good).any():means.append(float(np.mean(absolute[ref&good])))
                penalties.append(float(np.mean(np.where(good[ref],absolute[ref]/scale[column],5.))))
            rows[feature]=dict(unit=UNITS[column],intended_windows=len(positions),reference_valid=int(observed.sum()),
                paired_valid=int((observed&available).sum()),missing_prediction_for_valid_reference=int((observed&~available).sum()),
                invalid_physical_values=int((available&physical_invalid).sum()),
                subject_balanced_mae=float(np.mean(means)) if means else None,
                subject_balanced_penalized_scaled_mae=float(np.mean(penalties)) if penalties else None)
        output[name]=dict(subjects=len(people),intended_windows=len(positions),features=rows,
            group_scaled_errors={g:float(np.mean([rows[FEATURES[c]]['subject_balanced_penalized_scaled_mae'] for c in range(s.start,s.stop)])) for g,s in GROUPS.items()})
    return output


def infer(model,arrays,indices,center,scale,device):
    output=[];flags=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(indices),32):
            group=indices[start:start+32]
            result=model(*[torch.as_tensor(arrays[k][group],device=device) for k in ('x','roi','prior','descriptors')])
            output.append(result['features'].cpu().numpy()*scale+center);flags.append(result['usable'].cpu().numpy())
    value=np.concatenate(output);valid=np.repeat(np.concatenate(flags)[:,None],10,axis=1)&np.isfinite(value)
    return value,valid


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--waveforms',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    allowed=(Path.cwd()/'outputs/waveform_reset').resolve();out=args.out.resolve()
    if any(allowed not in p.resolve().parents for p in (args.data,args.waveforms,out)) or out.exists():parser.error('Use reset data and NEW output')
    arrays,fit,validation,data=load_development_arrays(args.data)
    waveform_report=json.loads((args.waveforms/'results.json').read_text())
    if not waveform_report['complete'] or waveform_report['split_sha256']!=data['split_sha256']:raise ValueError('Need completed matched waveform study')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    widths=matched_widths();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out.mkdir(parents=True,exist_ok=False)
    lock=dict(features=FEATURES,units=UNITS,widths=widths,seeds=SEEDS,steps=600,source_mix=.2,split_sha256=data['split_sha256'],
        source_sha256={p:file_sha256(p) for p in ('rppg_lab/waveform_feature_specialists.py','scripts/run_three_part_features.py')})
    (out/'configuration_lock.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
    started=time.perf_counter();truth,valid=targets(arrays['y']);weights=training_weights(arrays,fit);center,scale=statistics(truth,valid,weights)
    np.savez_compressed(out/'feature_targets.npz',target=truth,valid=valid,fit=fit,validation=validation,center=center,scale=scale)
    predictions={};pred_valid={};summaries={};runs=[]
    def record(name,value,mask):
        predictions[name]=value;pred_valid[name]=mask
        summaries[name]=summarize_features(value,mask,truth[validation],valid[validation],arrays,validation,scale)
    record('Train_mean',np.broadcast_to(center,(len(validation),10)),np.ones((len(validation),10),bool))
    x=input_features(arrays);xmean=(x*weights[:,None]).sum(0);xscale=np.sqrt(((x-xmean)**2*weights[:,None]).sum(0)).clip(.01)
    design=np.c_[(x-xmean)/xscale,np.ones(len(x))];regression=[]
    for c in range(10):
        group=fit[valid[fit,c]];w=weights[group];w=w/w.sum();a=design[group]
        beta=np.linalg.solve((a.T*w)@a+.01*np.eye(a.shape[1]),(a.T*w)@((truth[group,c]-center[c])/scale[c]))
        regression.append(design[validation]@beta*scale[c]+center[c])
    record('input_feature_ridge',np.stack(regression,1),np.ones((len(validation),10),bool))
    controls=waveform_report['controls']
    if file_sha256(args.waveforms/'controls_predictions.npz')!=controls['predictions_sha256']:raise ValueError('Waveform controls changed')
    with np.load(args.waveforms/'controls_predictions.npz',allow_pickle=False) as archive:
        for name in ('CHROM_WIN','quality_weighted'):
            value,mask=targets(archive[name]);record('waveform__'+name,value,mask)
    for row in waveform_report['runs']:
        job=row['job']
        if job['scope']=='full_train' and job['configuration']['name']=='long_tcn__roi_consistency0_2':
            path=args.waveforms/row['predictions']
            if file_sha256(path)!=row['predictions_sha256']:raise ValueError('Waveform predictions changed')
            with np.load(path,allow_pickle=False) as archive:value,mask=targets(archive['final'])
            record('waveform__TCN_consistency_seed'+str(job['seed']),value,mask)
    z=(truth-center)/scale
    pools={source:{s:fit[(arrays['source'][fit]==source)&(arrays['record_id'][fit]==s)] for s in sorted(set(arrays['record_id'][fit][arrays['source'][fit]==source]))} for source in (0,1)}
    for mode,width in widths.items():
        for seed in SEEDS:
            before=time.perf_counter();torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.manual_seed_all(seed)
            rng=np.random.default_rng(seed);model=ThreePartFeatureModel(mode,width).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4);curve=[]
            for step in range(600):
                indices=[]
                for _ in range(16):
                    source=int(rng.random()<.2);person=rng.choice(list(pools[source]));indices.append(int(rng.choice(pools[source][person])))
                indices=np.asarray(indices);model.train();optimizer.zero_grad(set_to_none=True)
                result=model(*[torch.as_tensor(arrays[k][indices],device=device) for k in ('x','roi','prior','descriptors')])
                loss=feature_loss(result['features'],torch.as_tensor(z[indices],dtype=torch.float32,device=device),torch.as_tensor(valid[indices],device=device))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite specialist loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                if step in (0,199,399,599):curve.append(dict(step=step+1,loss=float(loss.detach())))
            name=mode+'__seed'+str(seed);value,mask=infer(model,arrays,validation,center,scale,device);record(name,value,mask)
            path=out/(name+'.pt');torch.save(dict(model_state=model.cpu().state_dict(),mode=mode,width=width,seed=seed,steps=600,
                feature_names=FEATURES,center=torch.tensor(center),scale=torch.tensor(scale),split_sha256=data['split_sha256']),path)
            row=dict(name=name,mode=mode,width=width,seed=seed,steps=600,parameters=sum(p.numel() for p in model.parameters()),
                fit_curve=curve,seconds=time.perf_counter()-before,checkpoint=path.name,checkpoint_sha256=file_sha256(path))
            runs.append(row);print(json.dumps(dict(name=name,UBFC=summaries[name]['UBFC']['group_scaled_errors'],SCAMPS=summaries[name]['SCAMPS']['group_scaled_errors'])),flush=True)
            del optimizer,model
            if device.type=='cuda':torch.cuda.empty_cache()
    np.savez_compressed(out/'predictions.npz',**predictions);np.savez_compressed(out/'prediction_valid.npz',**pred_valid)
    report=dict(status='THREE_PART_FEATURE_DEVELOPMENT_NOT_BP',complete=len(runs)==9,runs=runs,summaries=summaries,
        feature_names=FEATURES,units=UNITS,center=center,scale=scale,widths=widths,split_sha256=data['split_sha256'],
        predictions_sha256=file_sha256(out/'predictions.npz'),validity_sha256=file_sha256(out/'prediction_valid.npz'),targets_sha256=file_sha256(out/'feature_targets.npz'),
        no_test_scoring=True,no_prior_weight_loading=True,no_private_data=True,seconds=time.perf_counter()-started,
        source_sha256=lock['source_sha256'],git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (out/'results.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=report['complete'],seconds=report['seconds'])))


if __name__=='__main__':main()
