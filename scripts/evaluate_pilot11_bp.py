"""Score the fixed small BP regressions after all label-free acquisitions."""
import argparse
from pathlib import Path
from rppg_lab.pilot11_bp import run_bp_study

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    result=run_bp_study(Path(args.run),Path('docs/waveform_research/pilot11_model_and_scope_lock_20261108.json'))
    print('Fixed BP feasibility evaluation complete:',len(result['models']),'models; private predictions retained locally')
