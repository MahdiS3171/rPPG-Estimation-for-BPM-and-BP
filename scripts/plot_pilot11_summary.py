"""Aggregate MAE figure; no participant points or pressure values."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output-prefix',required=True);args=p.parse_args()
    data=json.loads(Path(args.summary).read_text());models=data['bp']['models']
    names={'train_mean':'Training mean','HR_only':'HR only','HR_compact_geometry':'HR + compact-wave geometry',
        'HR_TCN_geometry':'HR + TCN-wave geometry','HR_joint_geometry':'HR + joint feature geometry','HR_independent_geometry':'HR + independent geometry'}
    chosen=[k for k in names if k in models];positions=np.arange(len(chosen));colors=['#6c757d','#1976b8','#15806c','#64a389','#7854a6','#c17d31'][:len(chosen)]
    fig,axes=plt.subplots(1,2,figsize=(11,4.9),sharey=True,layout='constrained')
    for i,(ax,target) in enumerate(zip(axes,('SBP','DBP'))):
        values=[models[k]['mae_mmHg'][i] for k in chosen];ax.barh(positions,values,color=colors,height=.65)
        limit=max(values)*1.4 if values else 1;ax.set_xlim(0,limit)
        for y,value,k in zip(positions,values,chosen):ax.text(value+limit*.015,y,f'{value:.2f}  (n={models[k]["subjects"]})',va='center',fontsize=9)
        ax.set_title(target,fontsize=12);ax.set_xlabel('Held-out session-mean MAE (mmHg)');ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(positions,[names[k] for k in chosen]);axes[0].invert_yaxis()
    fitting=sorted({data['bp']['development'][k]['subjects'] for k in chosen});evaluation=sorted({models[k]['subjects'] for k in chosen})
    counts=f'{fitting[0]} fitting people, {evaluation[0]} evaluation people' if len(fitting)==len(evaluation)==1 else 'Eligibility counts differ by comparator'
    fig.suptitle('Exploratory BP feasibility: '+counts,fontsize=12)
    unavailable=data['subjects']-data['bp']['coverage']['subjects_with_hr']
    fig.supxlabel(f'BP unavailable for {unavailable} of {data["subjects"]} recordings under fixed gates; legacy cuff timing. No clinical validation.',fontsize=9)
    prefix=Path(args.output_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(prefix.with_suffix('.png'),dpi=170);fig.savefig(prefix.with_suffix('.svg'));plt.close(fig)
    svg=prefix.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
