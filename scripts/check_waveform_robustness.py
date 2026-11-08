"""Feature-level shift, missing-input and equivalent-grid checks on owned models.

No fitting or original-source access. Interpolated inputs test numerical
consistency only; they supply no native-60 physiological ground truth.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.signal import resample_poly
import torch

from rppg_lab.artifacts import file_sha256
from rppg_lab.waveform_v2_screen_features import window_descriptors
from rppg_lab.waveform_v2_screen_models import AdaptiveWaveformNet
from rppg_lab.waveform_v2_extended_models import make_extended_model
from scripts.verify_waveform_screen_results import corr


def shift_edge(value,amount):
    if amount>0:return np.pad(value,[(0,0)]*(value.ndim-1)+[(amount,0)],mode='edge')[...,:value.shape[-1]]
    return np.pad(value,[(0,0)]*(value.ndim-1)+[(0,-amount)],mode='edge')[...,-amount:]


def conditions(x,roi,prior):
    for amount in (-4,-2,2,4):yield f'shift_{amount:+d}',shift_edge(x,amount),roi.copy(),prior.copy(),30,amount
    for index in range(3):
        value=x.copy();value[:,index]=0;r=roi.copy();r[:,index]=False;p=prior.copy();p[:,index]=False
        yield f'drop_roi_{index}',value,r,p,30,0
    yield 'no_priors',x.copy(),roi.copy(),np.zeros_like(prior),30,0
    yield 'flat',np.zeros_like(x),roi.copy(),prior.copy(),30,0
    yield 'equivalent_bandlimited_60',resample_poly(x,2,1,axis=-1).astype('float32'),roi.copy(),prior.copy(),60,0


def predict(model,x,roi,prior,descriptors,fs):
    with torch.inference_mode():
        out=model(torch.from_numpy(x),torch.from_numpy(roi),torch.from_numpy(prior),torch.from_numpy(descriptors),
            torch.zeros(len(x),75),torch.zeros(len(x),dtype=torch.bool),fs)
    return out['ppg'].numpy(),out['usable'].numpy()


def agreement(p,expected):
    # Discard one second at both edges; assess additional lag at the original
    # 30-Hz resolution. A zero-bin result is not sub-frame timing qualification.
    scores=[corr(p[30+lag:270+lag],expected[30:270]) for lag in range(-5,6)]
    return dict(zero_lag_r=corr(p[30:270],expected[30:270]),best_additional_lag_ms=(int(np.argmax(scores))-5)*1000/30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','confirmation','extensions','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--roi-trained',type=Path,help='Optional completed region-dropout training controls')
    args=parser.parse_args();allowed=(Path.cwd()/'outputs/waveform_screen').resolve()
    if any(allowed not in p.resolve().parents for p in (args.data,args.confirmation,args.extensions,args.out)) or args.out.exists():
        parser.error('Use owned screen artifacts and a NEW output')
    roots={'compact_residual':args.confirmation,'long_tcn':args.extensions,'physical_attention':args.extensions}
    if args.roi_trained:
        if allowed not in args.roi_trained.resolve().parents:raise ValueError('Use owned screen artifacts')
        roots.update(long_tcn__roi_dropout=args.roi_trained,long_tcn__roi_consistency0_2=args.roi_trained)
    reports={name:json.loads((root/'results.json').read_text()) for name,root in roots.items()}
    expected_hash=reports['compact_residual']['data_sha256']
    if file_sha256(args.data/'development_arrays.npz')!=expected_hash or any(r['data_sha256']!=expected_hash or not r['complete'] for r in reports.values()):
        raise ValueError('Need complete matched studies and unchanged inputs')
    with np.load(args.data/'development_arrays.npz',allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    torch.set_num_threads(1);rows=[];hashes={};replay_max=0.
    for name,root in roots.items():
        report=reports[name];variant='morph_mix20' if name=='compact_residual' else (name if name.startswith('long_tcn__roi_') else name+'__morphology')
        with np.load(root/'predictions.npz',allow_pickle=False) as saved:
            for fold,held in enumerate(report['folds']):
                indices=np.flatnonzero(np.isin(arrays['record_id'],held));key=f'fold{fold}_{variant}';path=root/(key+'.pt')
                run=next(r for r in report['runs'] if r['fold']==fold and r['configuration']['name']==variant)
                digest=file_sha256(path)
                if digest!=run['checkpoint_sha256']:raise ValueError('Owned checkpoint changed')
                hashes[name+'/'+key]=digest
                # Own hash-pinned local outputs contain NumPy scalar metadata.
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                if set(checkpoint['validation_ids'])!=set(held):raise ValueError('Wrong fold checkpoint')
                architecture='long_tcn' if name.startswith('long_tcn__roi_') else name
                model=AdaptiveWaveformNet('gate_residual') if name=='compact_residual' else make_extended_model(architecture)
                model.load_state_dict(checkpoint['model_state'],strict=True);model.eval()
                x=arrays['x'][indices];roi=arrays['roi'][indices];prior=arrays['prior'][indices]
                original,usable=predict(model,x,roi,prior,arrays['descriptors'][indices],30)
                replay_max=max(replay_max,float(abs(original-saved[key]).max()))
                np.testing.assert_allclose(original,saved[key],rtol=3e-4,atol=3e-4)
                for condition,value,r,p,fs,shift in conditions(x,roi,prior):
                    descriptors=np.stack([window_descriptors(v,fs) for v in value])
                    prediction,valid=predict(model,value,r,p,descriptors,fs)
                    if fs==60:prediction=prediction[:,::2]
                    expected=shift_edge(original,shift) if shift else original
                    for j,index in enumerate(indices):
                        finite=bool(np.isfinite(prediction[j]).all())
                        has_wave=finite and bool(np.std(prediction[j])>1e-8)
                        row=dict(model=name,condition=condition,index=int(index),subject=str(arrays['record_id'][index]),
                            usable=bool(valid[j]),finite=finite,has_waveform=has_wave)
                        if has_wave:row.update(agreement(prediction[j],expected[j]))
                        rows.append(row)
                print(json.dumps(dict(model=name,fold=fold+1,windows=len(indices))),flush=True)
    groups=defaultdict(list)
    for row in rows:groups[(row['model'],row['condition'])].append(row)
    summaries={}
    for (model,condition),group in groups.items():
        assert len(group)==152 and len({r['index'] for r in group})==152
        valid=[r for r in group if r['has_waveform']];people=sorted({r['subject'] for r in group})
        summary=dict(intended_windows=len(group),subjects=len(people),finite_outputs=sum(r['finite'] for r in group),
            usable_outputs=sum(r['usable'] for r in group),nonflat_waveforms=len(valid))
        if valid:
            summary.update(subject_balanced_zero_lag_r=float(np.mean([np.mean([r['zero_lag_r'] for r in valid if r['subject']==s]) for s in people])),
                window_p05_zero_lag_r=float(np.percentile([r['zero_lag_r'] for r in valid],5)),
                p95_absolute_additional_lag_ms=float(np.percentile([abs(r['best_additional_lag_ms']) for r in valid],95)),
                maximum_absolute_additional_lag_ms=float(max(abs(r['best_additional_lag_ms']) for r in valid)))
        summaries.setdefault(model,{})[condition]=summary
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'per_window.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    result=dict(status='DEVELOPMENT_FEATURE_LEVEL_ROBUSTNESS_NOT_PHONE_CALIBRATION',summaries=summaries,
        data_sha256=expected_hash,checkpoint_hashes=hashes,cpu_checkpoint_prediction_replay_max_absolute_difference=replay_max,
        per_window_sha256=file_sha256(args.out/'per_window.json'),no_fitting=True,no_original_source_access=True,
        interpolated_60_is_not_native60_evidence=True,lag_grid_ms=1000/30,
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (args.out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
