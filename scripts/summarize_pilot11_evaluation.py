"""Export aggregate evidence only; private rows and traces stay in outputs/."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from rppg_lab.artifacts import file_sha256,write_json


def number(value):
    try:return float(value)
    except (TypeError,ValueError):return np.nan


def median(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else None


def rows(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))


def summarize(root):
    root=Path(root);scope_path=Path('docs/waveform_research/pilot11_model_and_scope_lock_20261108.json')
    scope=json.loads(scope_path.read_text());completed=[];quality=[];delays=[];waveforms=[];rate=[];learned=[];feature=[]
    for subject in scope['subjects']:
        folder=root/'evaluation'/f'{subject}_V01';receipt=folder/'evaluation_complete.json'
        record=json.loads(receipt.read_text())
        for name,digest in record['artifact_sha256'].items():
            if file_sha256(folder/name)!=digest:raise ValueError('Evaluation artifact changed')
        completed.append((record,file_sha256(receipt)))
        quality.extend(rows(folder/'timing/waveform_quality.csv'));delays.extend(rows(folder/'timing/delay_windows.csv'))
        diagnostic=json.loads((folder/'inference/reference_free_diagnostics.json').read_text())
        for name,target in (('waveforms',waveforms),('rate_agreement',rate),('feature_support',feature)):
            target.extend(dict(r,private_subject=subject) for r in diagnostic[name])
        learned.extend(dict(r,private_subject=subject) for r in rows(folder/'learned_face_timing_sensitivity.csv'))
    independent=[r for r in delays if r['policy']=='B' and r['face_roi']=='combined' and r['hand_roi']=='hand_palm' and r['method']=='GREEN' and r['primary_nonoverlapping']=='True' and r['cuff_phase']=='candidate_cuff_free']
    strict=Counter(r['subject_id'] for r in independent if r['status']=='strict')
    reasons=Counter(part for r in independent for part in r['reasons'].split('|') if part)
    primary=dict(attempted_independent_candidate_windows=len(independent),computable=sum(r['computable']=='True' for r in independent),
        status_counts=dict(Counter(r['status'] for r in independent)),subjects_with_at_least_two_strict=sum(n>=2 for n in strict.values()),
        subjects_with_at_least_one_strict=len(strict),verified_cuff_free_windows=sum(r['timing_verified']=='True' for r in independent),
        failure_reason_counts=dict(reasons),physical_timing_calibrated=False,physical_uncertainty_bound_ms=None)
    for key in ('delay_xcorr_sec','split_difference_sec','phase_xcorr_difference_sec','cross_method_range_sec','cross_roi_range_sec'):
        per_subject=[median([number(r.get(key))*1000 for r in independent if r['subject_id']==s and r['computable']=='True']) for s in scope['subjects']]
        primary['subject_median_'+key.replace('_sec','_ms')]=median([number(v) for v in per_subject])
    primary['agreement_diagnostic_availability']={}
    for key in ('split_difference_sec','phase_xcorr_difference_sec','cross_method_range_sec','cross_roi_range_sec'):
        values=np.asarray([number(r.get(key)) for r in independent]);finite=np.isfinite(values)
        primary['agreement_diagnostic_availability'][key]=dict(attempted=len(values),unavailable=int((~finite).sum()),
            measured_within_33_334_ms=int((finite&(values<=.033334)).sum()),measured_exceeds_33_334_ms=int((finite&(values>.033334)).sum()))
    phase_quality=[]
    for roi in ('combined','hand_palm'):
        for phase in ('candidate_cuff_free','cuff_active','cuff_boundary_guard','crosses_boundary'):
            group=[r for r in quality if r['policy']=='B' and r['roi']==roi and r['method']=='GREEN' and r['cuff_phase']==phase]
            phase_quality.append(dict(roi=roi,phase=phase,attempted_overlapping=len(group),status_counts=dict(Counter(r['status'] for r in group)),
                median_snr_db=median([number(r['snr_db']) for r in group]),median_periodicity=median([number(r['periodicity']) for r in group])))
    policies=[]
    for policy in ('A','B','C'):
        group=[r for r in delays if r['policy']==policy and r['face_roi']=='combined' and r['hand_roi']=='hand_palm' and r['method']=='GREEN' and r['primary_nonoverlapping']=='True' and r['cuff_phase']=='candidate_cuff_free']
        policies.append(dict(policy=policy,guard_sec={'A':0,'B':5,'C':10}[policy],attempted=len(group),computable=sum(r['computable']=='True' for r in group),
            status_counts=dict(Counter(r['status'] for r in group))))
    output=[]
    for model,fs in sorted(set((r['model'],r['sample_rate_hz']) for r in waveforms)):
        group=[r for r in waveforms if (r['model'],r['sample_rate_hz'])==(model,fs)]
        output.append(dict(model=model,sample_rate_hz=fs,subjects=len(group),
            **{k:sum(r[k] for r in group) for k in ('attempted','finite','nonflat','candidate_quality_windows','hr_comparison_windows')},
            subject_median_abs_hr_difference_from_GREEN_bpm=median([number(r['median_abs_hr_difference_from_GREEN_bpm']) for r in group])))
    rate_results=[]
    for model in sorted(set(r['model'] for r in rate)):
        group=[r for r in rate if r['model']==model and r['candidate_quality_window']]
        per_subject=[];lags=[]
        for subject in scope['subjects']:
            part=[r for r in group if r['private_subject']==subject]
            per_subject.append(median([number(r['same_clock_correlation']) for r in part]))
            lags.append(median([abs(number(r.get('relative_xcorr_lag_sec')))*1000 for r in part]))
        rate_results.append(dict(model=model,attempted_candidate_quality_windows=len(group),
            subject_median_same_clock_correlation=median([number(v) for v in per_subject]),
            subject_median_abs_relative_lag_ms=median([number(v) for v in lags]),physiological_accuracy_available=False))
    # Group differences measure sensitivity to processing/seed/rate, not timing truth.
    learned_groups=defaultdict(list)
    for r in learned:
        if r['computable']=='True':learned_groups[r['private_subject'],r['sample_rate_hz'],r['window_index']].append(r)
    spreads=[]
    for group in learned_groups.values():
        predictions=[number(r['delay_xcorr_sec']) for r in group if r['model']!='face_GREEN']
        if len(predictions)>=2:spreads.append(1000*float(np.ptp(predictions)))
    bp=json.loads((root/'bp/bp_summary.json').read_text())
    replay=json.loads((root/'bp/numerical_verification.json').read_text())
    bandwidth=json.loads((root/'native_output_audit/aggregate.json').read_text())
    if bandwidth['input_receipt_sha256']!=sorted(h for _,h in completed):raise ValueError('Bandwidth audit does not cover the exact completed cohort')
    result=dict(date='2026-11-08',scope_sha256=file_sha256(scope_path),evaluation_receipt_sha256=sorted(h for _,h in completed),
        aggregation_source_sha256=file_sha256(__file__),metadata_snapshot_sha256=file_sha256(root/'private_metadata/metadata_snapshot.xlsx'),
        subjects=len(completed),models=15,recordings=len(completed),
        source_frames=sum(r['timing']['camera']['source_frames'] for r,_ in completed),
        source_seconds=sum(r['timing']['camera']['source_seconds'] for r,_ in completed),
        repeated_decoded_pixel_frames=sum(r['timing']['camera']['repeated_decoded_pixel_frames'] for r,_ in completed),
        every_frame_native_acquisition=True,contact_waveform_reference_available=False,
        primary_timing=primary,guard_sensitivity=policies,quality_by_phase=phase_quality,waveform_support=output,
        paired_30_60_processing=rate_results,learned_face_timing=dict(computable_rows=sum(r['computable']=='True' for r in learned),
            attempted_rows=len(learned),median_same_window_cross_model_lag_range_ms=median(spreads),
            lag_range_population='All computable windows and phases, six frozen waveform checkpoints',
            calibrated=False,hand_neural_model_validated=False),
        bp=bp,numerical_verification=replay,native_output_bandwidth=bandwidth,private_artifacts_published=False,limitations=['No local contact waveform truth','Unmeasured rolling shutter/exposure timing',
            'Legacy unverified cuff event times','Six development and five evaluation people at most','30-Hz training targets; native60 is processing sensitivity only'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    write_json(Path(args.output),summarize(Path(args.run)))
    print('Aggregate Pilot11 evidence exported; no private rows or traces included')
