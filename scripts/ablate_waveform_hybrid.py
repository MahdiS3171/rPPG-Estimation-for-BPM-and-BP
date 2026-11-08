"""Matched controls for whether adaptive gating and correct labels add value."""
import argparse
import json
from pathlib import Path
import time
import subprocess

import numpy as np
import torch

from scripts.confirm_waveform_candidates import fit_one
from scripts.run_waveform_screen import summarize,SEED
from rppg_lab.artifacts import file_sha256
from rppg_lab.waveform_v2_screen_models import AdaptiveWaveformNet
from rppg_lab.waveform_v2_frozen_cache import clean

VARIANTS=('global_learned_gate','fixed_CHROM_gate','quality_weighted_residual','shuffled_fit_waveforms')


class QualityResidualNet(AdaptiveWaveformNet):
    def __init__(self,sign,width=16):
        super().__init__('gate_residual',context=False,width=width)
        self.register_buffer('prior_sign',torch.tensor(sign,dtype=torch.float32))
        for parameter in self.gate.parameters():parameter.requires_grad_(False)
    def forward(self,x,roi_valid,prior_valid,descriptors,context,context_valid,fs=30):
        out=super().forward(x,roi_valid,prior_valid,descriptors,context,context_valid,fs)
        valid=(prior_valid&roi_valid[...,None]).reshape(len(x),18)
        score=descriptors[:,:72].reshape(len(x),18,4)[:,:,0].clamp_min(0)*valid
        weights=score*self.prior_sign[None]/score.sum(-1,keepdim=True).clamp_min(1e-8)
        priors=torch.nan_to_num(x[:,:,3:,:].reshape(len(x),18,-1))
        base=(weights[...,None]*priors).sum(1)
        out['ppg']=(base+.1*out['residual'])*out['usable'][:,None]
        out['quality_signed_weights']=weights
        return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--confirmation',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=Path.cwd();out=args.out.absolute();allowed=(root/'outputs/waveform_screen').resolve()
    if any(allowed not in p.resolve().parents for p in (args.data,args.confirmation,out)) or out.exists():parser.error('Use screen paths and NEW output')
    receipt=json.loads((args.data/'data_receipt.json').read_text())
    if file_sha256(args.data/'development_arrays.npz')!=receipt['arrays_sha256']:raise ValueError('Input archive changed')
    with np.load(args.data/'development_arrays.npz',allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    confirmation=json.loads((args.confirmation/'results.json').read_text())
    if not confirmation['complete'] or confirmation['data_sha256']!=receipt['arrays_sha256']:raise ValueError('Need completed matched confirmation')
    folds=confirmation['folds'];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    config=dict(kind='gate_residual',context=False,width=16,loss='morphology',synthetic=.2)
    out.mkdir(parents=True,exist_ok=False)
    (out/'configuration_lock.json').write_text(json.dumps(dict(variants=VARIANTS,folds=folds,configuration=config,
        seed=SEED,steps=300,data_sha256=receipt['arrays_sha256'],control='Matches morph_mix20 confirmation'),indent=2)+'\n')
    started=time.perf_counter();runs=[];predictions={};indices_archive={};parts={v:[] for v in VARIANTS}
    for fold,held in enumerate(folds):
        validation=np.flatnonzero(np.isin(arrays['record_id'],held));fit=np.flatnonzero(~np.isin(arrays['record_id'],held))
        sign=np.sign(np.mean([arrays['x'][i,:,3:,:].reshape(18,300)@arrays['y'][i]/300 for i in fit if arrays['source'][i]==0],axis=0));sign[sign==0]=1
        mean_descriptor=arrays['descriptors'][fit].mean(0)
        for variant in VARIANTS:
            view=dict(arrays)
            if variant=='global_learned_gate':view['descriptors']=np.broadcast_to(mean_descriptor,arrays['descriptors'].shape).copy()
            if variant=='shuffled_fit_waveforms':
                view['y']=arrays['y'].copy();rng=np.random.default_rng(SEED+991+fold)
                for source in (0,1):
                    group=fit[arrays['source'][fit]==source]
                    view['y'][group]=arrays['y'][rng.permutation(group)]
                np.testing.assert_array_equal(view['y'][validation],arrays['y'][validation])
            def factory(kind,context=False,width=16):
                if variant=='quality_weighted_residual':return QualityResidualNet(sign,width)
                model=AdaptiveWaveformNet(kind,context=context,width=width)
                if variant=='fixed_CHROM_gate':
                    for parameter in model.gate.parameters():parameter.requires_grad_(False)
                return model
            before=time.perf_counter();model,initial,prediction,curve=fit_one(config,view,fit,validation,device,SEED,300,model_factory=factory)
            key=f'fold{fold}_{variant}';predictions[key]=prediction;indices_archive[key]=validation;parts[variant].append((validation,prediction))
            path=out/(key+'.pt')
            torch.save(dict(model_state=model.state_dict(),configuration=config,variant=variant,validation_ids=held,
                seed=SEED,steps=300,data_sha256=receipt['arrays_sha256']),path)
            summary,_=summarize(prediction,arrays,validation)
            runs.append(dict(fold=fold,variant=variant,summary=summary,fit_curve=curve,seconds=time.perf_counter()-before,checkpoint_sha256=file_sha256(path)))
            print(json.dumps(clean(dict(fold=fold+1,variant=variant,UBFC=summary['UBFC']['subject_balanced']))),flush=True)
            (out/'partial_results.json').write_text(json.dumps(clean(runs),indent=2,allow_nan=False)+'\n')
    summaries={}
    for name,items in parts.items():
        indices=np.concatenate([i for i,_ in items]);prediction=np.concatenate([p for _,p in items])
        assert len(indices)==152 and len(np.unique(indices))==152
        summaries[name]=summarize(prediction,arrays,indices)[0]
    np.savez_compressed(out/'predictions.npz',**predictions);np.savez_compressed(out/'prediction_indices.npz',**indices_archive)
    report=dict(status='MATCHED_DEVELOPMENT_ABLATION',complete=len(runs)==20,completed_fits=len(runs),steps=300,
        seed=SEED,folds=folds,data_sha256=receipt['arrays_sha256'],summaries=summaries,runs=runs,
        no_original_val_test=True,no_private_data=True,shuffled_targets_fit_only=True,
        predictions_sha256=file_sha256(out/'predictions.npz'),indices_sha256=file_sha256(out/'prediction_indices.npz'),
        seconds=time.perf_counter()-started,git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (out/'results.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(clean(dict(complete=report['complete'],seconds=report['seconds'],summaries=summaries)),indent=2))


if __name__=='__main__':main()
