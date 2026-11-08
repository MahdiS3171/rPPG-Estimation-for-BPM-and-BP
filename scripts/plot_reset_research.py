"""Render aggregate-only Validation/internal-Test comparisons for the report."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('waveforms','features','test','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();w=json.loads((args.waveforms/'results.json').read_text());f=json.loads((args.features/'results.json').read_text());t=json.loads((args.test/'results.json').read_text())
    if not w['complete'] or not f['complete'] or not t['complete'] or not w['split_sha256']==f['split_sha256']==t['split_sha256']:raise ValueError('Need complete matched results')
    figure,axes=plt.subplots(2,3,figsize=(13,7.4),layout='constrained');colors=('#3280ac','#df8d38')
    families=('CHROM_WIN','morph_real_only','long_tcn__roi_consistency0_2');labels=('CHROM','Compact','Robust TCN')
    for column,(metric,title) in enumerate((('morphology_r','Aligned shape r'),('morphology_d1_r','Aligned derivative r'),('hr_mae','HR MAE (bpm)'))):
        ax=axes[0,column]
        for group,partition in enumerate(('Validation','Internal Test')):
            values=[];deviation=[]
            for name in families:
                if name=='CHROM_WIN':
                    table=w['controls']['summaries'][name] if group==0 else t['waveform_summaries'][name]
                    rows=[table['UBFC']['subject_balanced'][metric]]
                elif group==0:rows=[r['summary']['UBFC']['subject_balanced'][metric] for r in w['runs'] if r['job']['scope']=='full_train' and r['job']['configuration']['name']==name]
                else:rows=[v['UBFC']['subject_balanced'][metric] for key,v in t['waveform_summaries'].items() if key.startswith(name+'__seed')]
                values.append(np.mean(rows));deviation.append(np.std(rows))
            ax.bar(np.arange(3)+(group-.5)*.35,values,.35,yerr=deviation,capsize=3,label=partition,color=colors[group])
        ax.set_title(title+(' (lower is better)' if metric=='hr_mae' else ' (higher is better)'),fontsize=10)
        ax.set_xticks(range(3),labels);ax.set_ylim(0,1 if metric!='hr_mae' else 1.8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    modes=('independent','joint','shared_heads');labels=('Independent','Joint','Shared heads')
    for column,(key,title) in enumerate((('rhythm','Rhythm features'),('geometry','Pulse geometry'),('derivatives','Derivative features'))):
        ax=axes[1,column]
        for group,partition in enumerate(('Validation','Internal Test')):
            table=f['summaries'] if group==0 else t['feature_summaries'];values=[];deviation=[]
            for mode in modes:
                rows=[v['UBFC']['group_scaled_errors'][key] for name,v in table.items() if name.startswith(mode+'__seed')]
                values.append(np.mean(rows));deviation.append(np.std(rows))
            ax.bar(np.arange(3)+(group-.5)*.35,values,.35,yerr=deviation,capsize=3,color=colors[group])
        ax.set_title(title+' (lower is better)',fontsize=10);ax.set_xticks(range(3),labels,rotation=12)
        ax.set_ylabel('Train-SD-scaled feature error');ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=9)
    figure.suptitle('Public-data reset: waveform correction and three-part feature models',fontsize=14)
    figure.supxlabel('Six real subjects per evaluation partition. Bars average three seeds; whiskers show seed SD, not subject confidence intervals.\nAligned metrics allow reference shifts. Features use a 30-Hz filtered contact target. Historically exposed internal benchmark.',fontsize=9)
    args.out.mkdir(parents=True,exist_ok=True)
    figure.savefig(args.out/'reset_research_summary.png',dpi=160);figure.savefig(args.out/'reset_research_summary.svg')


if __name__=='__main__':main()
