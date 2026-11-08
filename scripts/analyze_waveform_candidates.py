"""Explain adaptive weights, residual contributions and local waveform examples."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.run_waveform_screen import batch_tensors,forward,summarize
from rppg_lab.artifacts import file_sha256
from rppg_lab.waveform_v2_screen_models import AdaptiveWaveformNet
from rppg_lab.waveform_v2_screen_losses import standardized,pearson
from rppg_lab.waveform_v2_frozen_cache import clean

NAMES=('gate__morphology','gate_residual__morphology','gate_residual__staged__context','dual_band__morphology')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--screen',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=Path.cwd();allowed=(root/'outputs/waveform_screen').resolve();out=args.out.resolve()
    if any(allowed not in p.resolve().parents for p in (args.data,args.screen,out)) or out.exists():parser.error('Use screen paths and NEW output')
    report=json.loads((args.screen/'results.json').read_text())
    if file_sha256(args.data/'development_arrays.npz')!=report['data_sha256']:raise ValueError('Prepared archive changed')
    with np.load(args.data/'development_arrays.npz',allow_pickle=False) as source:arrays={k:source[k] for k in source.files}
    validation=np.flatnonzero(arrays['validation']);fit=np.flatnonzero(~arrays['validation'])
    torch.set_num_threads(1);device=torch.device('cpu');batch=batch_tensors(arrays,validation,device)
    # This counterfactual uses fitting-input statistics only, never validation labels.
    mean_descriptor=torch.as_tensor(arrays['descriptors'][fit].mean(0))[None].repeat(len(validation),1)
    lookup={r['name']:r for r in report['results']};analysis={}
    for name in NAMES:
        path=args.screen/(name+'.pt')
        if file_sha256(path)!=lookup[name]['checkpoint_sha256']:raise ValueError('Owned checkpoint changed')
        # These are our just-created hash-pinned local artifacts; metadata contains
        # NumPy scalar strings, so use the trusted-artifact loader deliberately.
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        config=checkpoint['configuration'];model=AdaptiveWaveformNet(config['kind'],context=config['context']).eval()
        model.load_state_dict(checkpoint['model_state'],strict=True)
        with torch.inference_mode():
            full=forward(model,batch)
            fixed=forward(model,dict(batch,descriptors=mean_descriptor))
            no_context=forward(model,dict(batch,context_valid=torch.zeros_like(batch['context_valid'])))
            signed=full['weights'][:,:18]-full['weights'][:,18:]
            gate_only=(signed[...,None]*batch['x'][:,:,3:,:].reshape(len(validation),18,300)).sum(1)
        summaries={}
        for condition,pred in [('full',full['ppg']),('constant_fit_descriptor',fixed['ppg']),('no_past_context',no_context['ppg']),('gate_without_temporal_residual',gate_only)]:
            summaries[condition]=summarize(pred.numpy(),arrays,validation)[0]
        coefficient=signed.numpy()
        analysis[name]=dict(counterfactual_metrics=summaries,mean_signed_weights=coefficient.mean(0).reshape(3,6).tolist(),
            mean_absolute_window_weight_deviation=float(abs(coefficient-coefficient.mean(0)).mean()),
            net_signed_weight_l1_mean=float(abs(coefficient).sum(1).mean()),
            counterfactual_is_not_retraining=True,physiological_confidence_validated=False)
    out.mkdir(parents=True,exist_ok=False)
    (out/'analysis.json').write_text(json.dumps(clean(analysis),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with np.load(args.screen/'validation_predictions.npz',allow_pickle=False) as archive:
        predictions={n:archive[n] for n in ('CHROM_WIN','quality_weighted','gate_residual__morphology')}
    real=np.flatnonzero(arrays['source'][validation]==0)
    target=arrays['y'][validation]
    p=torch.tensor(predictions['gate_residual__morphology']);y=torch.tensor(target)
    native=pearson(p,y).numpy();order=real[np.argsort(native[real])]
    cases=(order[0],order[len(order)//2],order[-1])
    fig,axes=plt.subplots(3,2,figsize=(14,8),layout='constrained')
    colors={'CHROM_WIN':'#888888','quality_weighted':'#2b9958','gate_residual__morphology':'#2365b3'}
    for row,j in enumerate(cases):
        truth=standardized(y[j:j+1]).numpy()[0];time_axis=np.arange(300)/30
        axes[row,0].plot(time_axis,truth,color='black',label='Contact target',linewidth=1.3)
        axes[row,1].plot(np.arange(270)/30,truth[15:-15],color='black',label='Contact target',linewidth=1.3)
        for name,prediction in predictions.items():
            wave=standardized(torch.tensor(prediction[j:j+1])).numpy()[0]
            axes[row,0].plot(time_axis,wave,color=colors[name],alpha=.8,label=name,linewidth=.9)
            scores=[np.corrcoef(wave[k:k+270],truth[15:-15])[0,1] for k in range(31)];best=int(np.argmax(scores))
            axes[row,1].plot(np.arange(270)/30,wave[best:best+270],color=colors[name],alpha=.8,linewidth=.9,
                label=f'{name}: diagnostic lag {(best-15)*1000/30:.0f} ms')
        identity=str(arrays['record_id'][validation[j]])
        axes[row,0].set_title(f'{identity}, start {arrays["start"][validation[j]]:.2f}s; native r={native[j]:.3f}')
        axes[row,1].set_title('DISPLAY-ONLY shape alignment; output files remain unshifted')
        for ax in axes[row]:ax.set_xlabel('Seconds');ax.grid(alpha=.2);ax.legend(fontsize=6,loc='upper right')
    fig.suptitle('Worst / median / best native-correlation cases — development data, not physiological timing validation',fontsize=11)
    fig.savefig(out/'waveform_examples.png',dpi=150);fig.savefig(out/'waveform_examples.svg')
    for name,value in analysis.items():
        m=value['counterfactual_metrics']
        print(json.dumps(clean(dict(name=name,weight_variation=value['mean_absolute_window_weight_deviation'],
            full=m['full']['UBFC']['subject_balanced'],constant_gate=m['constant_fit_descriptor']['UBFC']['subject_balanced'],
            no_residual=m['gate_without_temporal_residual']['UBFC']['subject_balanced']))),flush=True)


if __name__=='__main__':main()
