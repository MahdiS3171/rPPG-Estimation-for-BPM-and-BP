"""Gate 07 fixed-protocol optical comparison. No decoded reference values or models for rPPG."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import argparse
import shutil
import time
import tracemalloc
from collections import Counter

import numpy as np

from rppg_lab.waveform_v2_scamps import ScampsRecording,discover_scamps,source_sha256
from rppg_lab.waveform_v2_scamps_roi import ROI_NAMES,ScampsROIConfig,create_scamps_face_detector
from rppg_lab.waveform_v2_scamps_inputs import prepare_scamps_v1_inputs
from rppg_lab.waveform_v2_optical_audit import collect_optical_evidence,SAMPLE_INDICES
from rppg_lab.waveform_v2_optical_quality import OpticalQualityConfig,temporal_descriptors
from rppg_lab.waveform_v2_micro_rois import evaluate_candidate,weight_change_diagnostics
from rppg_lab.waveform_v2_quality_timing import timing_cases
from scripts.prepare_waveform_v2_scamps_inputs import RSSBudget

SPEC_COMMIT='0d0fb28ffb27ad0abf2d478367f0027114de1ae6'


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    return value


def write_json(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(clean(value),f,indent=2,allow_nan=False);f.write('\n')


def describe(values):
    values=np.asarray(values,dtype=float)
    good=values[np.isfinite(values)]
    return dict(intended_observations=values.size,finite_observations=good.size,
        missing_observations=values.size-good.size,minimum=float(good.min()) if good.size else None,
        median=float(np.median(good)) if good.size else None,mean=float(good.mean()) if good.size else None,
        maximum=float(good.max()) if good.size else None)


def review_template(review_id,frame_index,roi):
    fields=dict(reviewer_id=None,skin_fraction_bin=None,hair_fraction_bin=None,background_fraction_bin=None,
        shadow_fraction_bin=None,glare_fraction_bin=None,occlusion_severity=None,anatomical_confidence=None,
        uncertainty_reason=None,notes=None)
    return dict(annotation_id=f'{review_id}_f{frame_index:04d}_{roi}',review_id=review_id,
        original_frame_index=frame_index,roi_name=roi,status='PENDING_HUMAN_REVIEW',
        reviewer_1=fields.copy(),reviewer_2=fields.copy(),adjudication=None)


def contact_sheet(contacts,review_id,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,5,figsize=(15,11),constrained_layout=True)
    colors=('gold','cyan','magenta')
    for col,index in enumerate(SAMPLE_INDICES):
        item=contacts[index];frame=item['rgb'];masks=item['masks']
        axes[0,col].imshow(frame)
        axes[0,col].set_title(f'Frame {index}: original context',fontsize=10)
        for r,name in enumerate(ROI_NAMES):
            mask=None if masks is None else masks[name]
            if mask is not None and mask.any():
                axes[0,col].contour(mask,levels=[.5],colors=[colors[r]],linewidths=.8)
                yy,xx=np.nonzero(mask)
                y0,y1=max(0,int(yy.min())-12),min(frame.shape[0],int(yy.max())+13)
                x0,x1=max(0,int(xx.min())-12),min(frame.shape[1],int(xx.max())+13)
                axes[r+1,col].imshow(frame[y0:y1,x0:x1])
                axes[r+1,col].contour(mask[y0:y1,x0:x1],levels=[.5],colors=[colors[r]],linewidths=1.)
            else:
                axes[r+1,col].imshow(frame)
                axes[r+1,col].text(.5,.5,'Geometry unavailable',ha='center',transform=axes[r+1,col].transAxes,color='white')
            axes[r+1,col].set_title(f'{name}: frame {index}',fontsize=10)
    for ax in axes.flat:ax.axis('off')
    fig.suptitle(f'SCAMPS review {review_id} / original V1 polygons\n'
        'Full context plus ROI insets; automated scores and Candidate B are hidden\n'
        'McDuff et al., 2022 / R-UDA non-commercial research. Annotation status: PENDING.',fontsize=12)
    fig.savefig(output/f'{review_id}.png',dpi=125)
    plt.close(fig)


def summarize_recording(recording,evidence,candidate,previous_hashes):
    baseline=evidence['baseline'];n=len(baseline.frame_indices)
    old=prepare_scamps_v1_inputs(baseline,(0,300))
    hashes=[hashlib.sha256(v.tobytes()).hexdigest() for v in old['x']]
    if hashes!=previous_hashes:raise AssertionError('Baseline A changed relative to Gate 06')
    diagnostics=weight_change_diagnostics(evidence['patch_rgb'],candidate['provisional_weights'])
    rows={}
    for r,name in enumerate(ROI_NAMES):
        provisional_metrics={}
        for k in ('chroma_compatible_fraction','clipped_fraction','brightness','spatial_chroma_mad'):
            values=evidence['patch_metrics'][k][r]
            active=candidate['provisional_weights'][r]>0
            safe=np.where(active,values,0)
            weighted=np.sum(candidate['provisional_weights'][r]*safe,axis=0)
            weighted[~candidate['provisional_valid'][r]]=np.nan
            provisional_metrics[k]=describe(weighted)
        centers=np.column_stack((evidence['roi_metrics']['centroid_x'][r],evidence['roi_metrics']['centroid_y'][r]))
        rows[name]=dict(intended_frames=n,A_acquisition_valid_frames=int(baseline.roi_valid[r].sum()),
            A_acquisition_failure_reasons=dict(Counter(v for v in baseline.failure_reason[r] if v)),
            A_optical_indicators={k:describe(v[r]) for k,v in evidence['roi_metrics'].items()},
            anatomical_skin_confidence=dict(status='UNKNOWN',independent_annotations=0,confidence=None),
            hair_or_background_fraction_confirmed=None,
            B_provisional_available_frames=int(candidate['provisional_valid'][r].sum()),
            B_trusted_available_frames=int(candidate['roi_valid'][r].sum()),
            B_whole_ROI_block_reasons=candidate['whole_roi_reason'][r].tolist(),
            B_planned_weights=candidate['planned_weights'][r].tolist(),
            B_eligible_patch_counts=candidate['eligible_patches'][r].sum(-1).tolist(),
            B_provisional_optical_indicators=provisional_metrics,
            A_temporal_indicators=temporal_descriptors(baseline.rgb[r],centers),
            B_provisional_temporal_indicators=temporal_descriptors(candidate['provisional_rgb'][r]),
            weight_stability=dict(provisional_l1_step=describe(diagnostics['weight_l1_step'][r]),
                changing_weight_rgb_term=describe(diagnostics['weight_change_rgb_term'][r]),
                block_boundary_indices=[300]),
            physiological_pulse_quality='UNVERIFIED',trusted_abstention_frames=n-int(candidate['roi_valid'][r].sum()))
    return dict(filename=recording.path.name,source_sha256=recording.info['source_sha256'],
        decoded_datasets=['Xsub'],reference_values_decoded=False,
        quality_schema=evidence['quality_schema'],baseline_input_hashes=hashes,baseline_hashes_match_gate06=True,
        roi=rows,allow_training=False)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scamps-root',required=True,type=Path)
    ap.add_argument('--out-dir',required=True,type=Path)
    ap.add_argument('--chunk-frames',type=int,default=32)
    args=ap.parse_args(argv)
    paths=discover_scamps(args.scamps_root)
    if [p.name for p in paths]!=[f'P{i:06d}.mat' for i in range(1,11)]:
        ap.error('Only the ten already-admitted example filenames are allowed')
    output=args.out_dir.resolve();source=args.scamps_root.resolve()
    if output.exists():ap.error('Refusing existing output directory')
    if output==source or source in output.parents:ap.error('Output must be outside source recordings')
    if type(args.chunk_frames) is not int or not 1<=args.chunk_frames<=32:ap.error('Use chunks of 1-32 frames')
    config=OpticalQualityConfig()
    old=json.loads(Path('docs/waveform_v2/scamps_gate06/roi_coverage_and_inputs.json').read_text())
    lookup={r['filename']:r for r in old['recordings']}
    output.mkdir(parents=True,exist_ok=False)
    contacts=output/'contact_sheets';contacts.mkdir();arrays=output/'retained_arrays';arrays.mkdir()
    shutil.copyfile('docs/waveform_v2/scamps/LICENSE_SCAMPS.txt',output/'LICENSE_SCAMPS.txt')
    rows=[];reviews=[];mapping=[];runtimes=[]
    codefiles=['waveform_v2_optical_quality','waveform_v2_micro_rois','waveform_v2_quality_timing','waveform_v2_optical_audit']
    hashes={name:source_sha256(Path('rppg_lab')/(name+'.py')) for name in codefiles}
    with RSSBudget(1024**3) as memory:
        for path in paths:
            print(f'{path.name}: same-frame optical A/B comparison; no decoded PPG',flush=True)
            recording=ScampsRecording(source,path.name,max_window_frames=args.chunk_frames,
                expected_sha256=lookup[path.name]['source_sha256'])
            detector=create_scamps_face_detector(ScampsROIConfig(chunk_frames=args.chunk_frames,array_budget_bytes=192*1024**2))
            tracemalloc.start();started=time.perf_counter()
            try:
                evidence=collect_optical_evidence(recording,detector,chunk_frames=args.chunk_frames,resource_check=memory.check)
                _,peak=tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop();detector.close()
            choose_start=time.perf_counter()
            candidate=evaluate_candidate(evidence['patch_rgb'],evidence['patch_valid'],evidence['patch_metrics'],evidence['roi_metrics'],config=config)
            selection_seconds=time.perf_counter()-choose_start
            row=summarize_recording(recording,evidence,candidate,[w['input_sha256'] for w in lookup[path.name]['windows']])
            rows.append(row)
            review_id='R'+hashlib.sha256((path.name+recording.info['source_sha256']).encode()).hexdigest()[:8]
            contact_sheet(evidence['contacts'],review_id,contacts)
            mapping.append(dict(review_id=review_id,filename=path.name,source_sha256=recording.info['source_sha256'],
                frame_indices=SAMPLE_INDICES,contact_sheet=f'contact_sheets/{review_id}.png',
                contact_sheet_sha256=source_sha256(contacts/f'{review_id}.png')))
            reviews.extend(review_template(review_id,i,name) for i in SAMPLE_INDICES for name in ROI_NAMES)
            retained=dict(baseline_rgb=evidence['baseline'].rgb,baseline_acquisition_valid=evidence['baseline'].roi_valid,
                baseline_failure_reason=evidence['baseline'].failure_reason,frame_indices=evidence['baseline'].frame_indices,
                patch_rgb=evidence['patch_rgb'],patch_acquisition_valid=evidence['patch_valid'],
                **{f'patch_metric_{k}':v for k,v in evidence['patch_metrics'].items()},
                **{f'roi_metric_{k}':v for k,v in evidence['roi_metrics'].items()},
                **{f'candidate_{k}':v for k,v in candidate.items() if isinstance(v,np.ndarray)})
            np.savez_compressed(arrays/f'{path.stem}.npz',**retained)
            runtimes.append(dict(filename=path.name,**evidence['runtime'],
                B_selection_and_mix_seconds=selection_seconds,recording_wall_seconds=time.perf_counter()-started,
                traced_collection_peak_bytes=peak,process_peak_so_far_bytes=memory.peak))
            if source_sha256(path)!=lookup[path.name]['source_sha256']:raise AssertionError('Source changed')
            print(f"{path.name}: A valid {[v['A_acquisition_valid_frames'] for v in row['roi'].values()]}; "
                  f"B provisional {[v['B_provisional_available_frames'] for v in row['roi'].values()]}; trusted [0,0,0] (independent skin review pending)",flush=True)
            del evidence,candidate,retained,detector,recording
            memory.check()
        memory_report=dict(array_budget_bytes=192*1024**2,rss_budget_bytes=memory.limit,
            observed_process_peak_bytes=memory.peak,rss_budget_exceeded=memory.exceeded,rss_sampling_interval_s=.002,
            runtime=runtimes,limitations=['Tracemalloc excludes some native allocations','RSS sampling can miss brief native peaks'])
    comparison=dict(scope='exploratory_engineering_not_independent_quality_validation',specification_commit=SPEC_COMMIT,
        specification_file_sha256=source_sha256('docs/WAVEFORM_V2_OPTICAL_QUALITY_GATE07_SPEC.md'),
        fixed_config=asdict(config),code_sha256=hashes,recordings=rows,
        intended_ROI_frames=18000,human_annotations_completed=0,confirmed_skin_labels_available=False,
        physiological_pulse_quality='UNVERIFIED',mask_temporal_mapping='UNVERIFIED',segmentation_masks_used=False,
        allow_training=False,source_ppg_values_decoded=False)
    write_json(output/'optical_quality_comparison.json',comparison)
    write_json(output/'contact_review_index.json',dict(role='coordinator_only_hide_during_blinded_review',recordings=mapping))
    write_json(output/'human_annotation_template.json',dict(status='PENDING_HUMAN_REVIEW',annotations=reviews,completed=0))
    write_json(output/'memory_runtime.json',memory_report)
    write_json(output/'synthetic_timing.json',dict(scope='synthetic_only',cases=timing_cases(30.)+timing_cases(60.)))
    pair=[r for r in rows if r['filename'] in ('P000001.mat','P000005.mat')]
    write_json(output/'documented_forehead_comparison.json',dict(scope='documented_engineering_examples_not_validation_truth',
        fixed_parameters=True,ppg_or_prediction_scores_used=False,
        examples=[dict(filename=r['filename'],source_sha256=r['source_sha256'],**r['roi']['forehead']) for r in pair],
        trusted_outcome='Whole ROI abstains when independent skin confidence is UNKNOWN; compatible patches cannot rescue it'))
    print(json.dumps(dict(recordings=len(rows),human_annotations=0,allow_training=False,rss_peak_bytes=memory_report['observed_process_peak_bytes'])),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
