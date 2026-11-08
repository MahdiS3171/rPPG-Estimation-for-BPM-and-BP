"""Post-hoc timing diagnosis of saved development predictions; never repairs them."""
import argparse
import json
from pathlib import Path

import numpy as np

from rppg_lab.artifacts import file_sha256
from rppg_lab.datasets import load_ubfc_ground_truth, _prepare_multi_roi_reference
from rppg_lab.waveform_alignment import aligned_overlap, lag_candidates
from rppg_lab.waveform_v2_classical_study import correlation, finite_mean
from rppg_lab.waveform_v2_frozen_cache import clean
from scripts.study_two_source_waveform import install_guard


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('study',type=Path)
    args=parser.parse_args(); root=Path.cwd(); study=args.study.resolve()
    if (root/'outputs/two_source_waveform').resolve() not in study.parents:
        parser.error('Use a completed two_source_waveform study')
    report=json.loads((study/'results.json').read_text())
    ids={r['subject'] for r in report['window_rows']}
    install_guard(root,ids)
    if file_sha256(study/'locked_predictions.npz')!=report['prediction_archive_sha256']:
        raise ValueError('Saved predictions changed')
    refs={}
    for subject in ids:
        path=Path('UBFCData')/subject/'ground_truth.txt'
        if file_sha256(path)!=report['data_sha256'][str(path)]: raise ValueError('Reference changed')
        t,hr,ppg=load_ubfc_ground_truth(path)
        t,hr,ppg,_=_prepare_multi_roi_reference(t,hr,ppg)
        refs[subject]=(t,ppg)
    rows=[]
    with np.load(study/'locked_predictions.npz',allow_pickle=False) as archive:
        for row in report['window_rows']:
            t,ppg=refs[row['subject']]
            target=np.interp(row['start']+np.arange(300)/30,t,ppg)
            prediction=archive[row['subject']+'_'+str(row['start'])+'_'+row['method']]
            native=correlation(prediction,target)
            if abs(native-row['metrics']['native_r'])>2e-6:
                raise ValueError('Independent reference reconstruction does not reproduce native r')
            best=-np.inf; chosen=None
            for lag in lag_candidates(300,15):
                a,b=aligned_overlap(prediction,target,lag)
                value=correlation(a,b)
                if value>best: best=value; chosen=lag
            rows.append(dict(subject=row['subject'],method=row['method'],start=row['start'],
                native_r=native,lag_diagnostic_r=best,diagnostic_lag_ms=chosen*1000/30,
                search_boundary=abs(chosen)==15))
    summaries=[]
    for method in dict.fromkeys(r['method'] for r in rows):
        selected=[r for r in rows if r['method']==method]
        per_subject={s:{k:finite_mean([r[k] for r in selected if r['subject']==s])
            for k in ('native_r','lag_diagnostic_r','diagnostic_lag_ms')} for s in ids}
        summaries.append(dict(method=method,
            subject_balanced={k:finite_mean([v[k] for v in per_subject.values()]) for k in ('native_r','lag_diagnostic_r','diagnostic_lag_ms')},
            boundary_windows=sum(r['search_boundary'] for r in selected),windows=len(selected)))
    output=dict(status='POSTHOC_DEVELOPMENT_DIAGNOSTIC_ONLY',predictions_shifted=False,
        physical_delay_identified=False,reference_replay_tolerance=2e-6,summaries=summaries,rows=rows)
    with (study/'timing_diagnostics.json').open('x',encoding='utf-8') as stream:
        json.dump(clean(output),stream,indent=2,allow_nan=False)
    print(json.dumps(clean(summaries),indent=2))


if __name__=='__main__': main()
