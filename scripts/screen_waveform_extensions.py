"""Bounded comparison of longer context, attention, spectral and temporal gates."""
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
from rppg_lab.waveform_v2_extended_models import make_extended_model
from rppg_lab.waveform_v2_frozen_cache import clean


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    root=Path.cwd();allowed=(root/'outputs/waveform_screen').resolve();out=args.out.resolve()
    if allowed not in out.parents or allowed not in args.data.resolve().parents or out.exists():parser.error('Use screen data and NEW output')
    receipt=json.loads((args.data/'data_receipt.json').read_text())
    if file_sha256(args.data/'development_arrays.npz')!=receipt['arrays_sha256']:raise ValueError('Data changed')
    with np.load(args.data/'development_arrays.npz',allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
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


if __name__=='__main__':main()
