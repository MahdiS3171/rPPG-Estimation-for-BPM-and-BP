"""Orchestration and provenance for the isolated Pilot-6 feasibility experiment."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import pickle
import platform
from pathlib import Path
import shutil
import sys

import numpy as np

from .artifacts import file_sha256, provenance, write_json
from .config import PipelineConfig
from .extraction_cache import FaceROIExtraction
from .pilot_analysis import analyze_signals, validate_timing, finite_stat, exploratory_bp
from .pilot_metadata import read_tables, normalize_tables, match_video, align_references, fetch_google_workbook
from .pilot_video import inspect_video, extract_raw, traces_from_arrays
from .processing import shared_grid
from .types import SharedTimebase
from .waveform_visualization import write_csv


def checkpoint_audit(path, manifest=None, enable=False):
    """State/config verification is separate from permission to execute inference."""
    import torch
    from .waveform_training import checkpoint_dataset_config, reconstruct_waveform_model
    from .waveform_inference import FrozenWaveformExtractor
    path = Path(path)
    info = dict(path=str(path.resolve()),status="unavailable",face_inference="blocked",hand_inference="unsupported",
                hand_reason="Existing frozen API requires three named facial ROIs. No verified hand feature-to-ROI mapping; no synthetic facial slots substituted.")
    extractor = None
    if not path.is_file():
        info["reason"] = "checkpoint_missing"
        return info,None
    info["sha256"] = file_sha256(path)
    try:
        checkpoint = torch.load(path,map_location="cpu",weights_only=True)
        checkpoint_dataset_config(checkpoint)
        model = reconstruct_waveform_model(checkpoint)
        if checkpoint.get("waveform_v1_status") != "frozen_v1" or checkpoint.get("freeze_decision") != "FREEZE_WAVEFORM_V1":
            raise ValueError("Checkpoint is not a frozen identity")
        info.update(status="state_and_contract_compatible",model_class=checkpoint["model_class"],
                    model_config=checkpoint["model_config"],roi_order=checkpoint["roi_names"],
                    feature_order=checkpoint["channel_ordering"],fs=checkpoint["fs"],
                    samples=checkpoint["window_length_samples"],selected_epoch=checkpoint["selected_epoch"],
                    source_candidate_sha256=checkpoint["frozen_from_checkpoint_sha256"],
                    extraction_config=checkpoint["extraction_config"],
                    normalization="Per-window channel standardization; window-local classical priors; existing prepare_window API only",
                    output="signed independently normalized waveform windows; no continuous stitching",
                    eval_mode=not model.training,strict_state_load=True)
        del model
        manifest = Path(manifest) if manifest else path.with_name("waveform_v1_freeze_manifest.json")
        info["freeze_manifest_path"] = str(manifest.resolve())
        if not manifest.is_file():
            info.update(reason="required_original_freeze_manifest_missing",face_inference="blocked")
        else:
            info["freeze_manifest_sha256"] = file_sha256(manifest)
            extractor = FrozenWaveformExtractor(path,manifest)
            info.update(face_inference="ready" if enable else "disabled_by_classical_only",reason=None)
    except (ValueError,KeyError,RuntimeError,TypeError,OSError,EOFError,pickle.UnpicklingError) as exc:
        info.update(status="incompatible",reason=f"{type(exc).__name__}: {exc}")
        extractor = None
    return info,extractor if enable else None


def frozen_recording(checkpoint, manifest, traces, raw, output, enabled):
    if not enabled:
        return dict(status="not_requested")
    info,extractor = checkpoint_audit(checkpoint,manifest,True)
    if extractor is None:
        return dict(status="blocked",reason=info.get("reason"))
    raw_identity = json.loads(str(raw["identity_json"].item()))
    if raw_identity["max_dimension"]:
        return dict(status="blocked",reason="spatially_resized_pilot_RGB_does_not_meet_frozen_preprocessing_contract; use --max-dimension 0")
    expected = extractor.extraction_config.to_dict()
    for key in ("skin_mask","skin_cr_range","skin_cb_range","min_pixels","detector_backend",
                "face_model","min_detection_confidence","min_tracking_confidence"):
        actual = raw_identity["config"][key]
        if json.dumps(actual) != json.dumps(expected[key]):
            return dict(status="blocked",reason="frozen_extraction_contract_mismatch:"+key)
    source = raw["timestamps"]
    if len(source) < 2:
        return dict(status="blocked",reason="insufficient_source_timestamps")
    fs = 1/np.median(np.diff(source))
    if fs > extractor.fs_target/0.95:
        return dict(status="blocked",reason="native_fps_exceeds_frozen_30Hz_contract; no_validated_antialiasing_adaptation")
    grid = shared_grid(source,extractor.fs_target)
    if len(grid) < extractor.window_length_samples:
        return dict(status="unavailable",reason="insufficient_support_for_frozen_window",windows=0)
    names = extractor.roi_names
    extraction = FaceROIExtraction(SharedTimebase(grid,source),{n:traces[n] for n in names})
    waveforms,times,rows = [],[],[]
    for start in range(0,len(grid)-extractor.window_length_samples+1,round(5*extractor.fs_target)):
        try:
            prediction = extractor.infer_window(extraction,float(grid[start]))
            waveforms.append(prediction["waveform"]); times.append(prediction["timestamps"])
            rows.append(dict(start_sec=float(grid[start]),status="inferred",phase_validated=False,
                             roi_attention=prediction["roi_attention"].tolist(),prior_weights=prediction["prior_weights"].tolist()))
        except ValueError as exc:
            waveforms.append(np.full(extractor.window_length_samples,np.nan))
            times.append(grid[start:start+extractor.window_length_samples])
            rows.append(dict(start_sec=float(grid[start]),status="invalid",reason=str(exc)))
    np.savez_compressed(output/"frozen_face_windows.npz",waveform=np.asarray(waveforms),timestamps=np.asarray(times))
    write_json(output/"frozen_windows.json",rows)
    return dict(status="inferred",windows=len(rows),successful=sum(r["status"] == "inferred" for r in rows),
                hand="unsupported",timing_use="not_validated_not_used")


def model_phase_probe(extractor):
    """Analytical common-input shift test; no reference or model modification."""
    from .types import RGBTrace
    from .bp import estimate_delay
    fs = extractor.fs_target
    t = np.arange(extractor.window_length_samples)/fs
    signals = []
    for lag in (0,.1):
        x = np.sin(2*np.pi*1.2*(t-lag))+.2*np.sin(2*np.pi*2.4*(t-lag))
        rgb = np.column_stack((130+.8*x,110+x,90+.3*x))
        traces = {n:RGBTrace(t,rgb,n,np.ones(len(t),bool)) for n in extractor.roi_names}
        prediction = extractor.infer_window(FaceROIExtraction(SharedTimebase(t,t),traces),0.)
        signals.append(prediction["waveform"])
    return dict(status="synthetic_only",known_shift_sec=.1,
                result=estimate_delay(*signals,fs,.3,1.2),
                limitation="Synthetic common-input shift equivariance does not validate learned physiological phase or hand generalization")


def session_features(identity,quality,delays,references,truncated):
    features = dict(subject_id=identity["subject_id"],recording_id=identity["recording_id"],
                    session_id=identity["session_id"],source_sheet="02_Sessions_last_two_summary",
                    source_row=identity["matched_row"]["source_row"] if identity.get("matched_row") else None,
                    prediction_kind="observed_cuff_session_summary",association_eligible=False,
                    association="session_only; cuff times unverified; cuff interference not excluded",
                    observed_cuff_sbp_mmHg=None,observed_cuff_dbp_mmHg=None)
    valid_refs = [r for r in references if r["observed_cuff_sbp_mmHg"] is not None and r["observed_cuff_dbp_mmHg"] is not None]
    if len(valid_refs) >= 2:
        last = sorted(valid_refs,key=lambda r:r["measurement_index"])[-2:]
        features.update(observed_cuff_sbp_mmHg=float(np.mean([r["observed_cuff_sbp_mmHg"] for r in last])),
                        observed_cuff_dbp_mmHg=float(np.mean([r["observed_cuff_dbp_mmHg"] for r in last])),
                        observed_cuff_hr_bpm=finite_stat([r.get("hr_reference_bpm") if r.get("hr_reference_bpm") is not None else np.nan for r in last],np.mean),
                        reference_summary="last_two_recorded_cuff_readings_mean_per_workbook_protocol")
    counts = {}
    for site,roi in (("face","combined"),("hand","hand_palm")):
        selected = [r for r in quality if r["roi"] == roi and r["method"] == "GREEN" and
                    r["window_sec"] == 10 and r["primary_nonoverlapping"] and r["status"] == "good"]
        counts[site] = len(selected)
        features[site+"_primary_good_windows"] = len(selected)
        features[site+"_hr_bpm"] = finite_stat([r["hr_bpm"] for r in selected])
        features[site+"_snr_db"] = finite_stat([r["snr_db"] for r in selected])
        hr_ref = features.get("observed_cuff_hr_bpm")
        hr_est = features[site+"_hr_bpm"]
        features[site+"_minus_session_cuff_hr_bpm"] = hr_est-hr_ref if hr_est is not None and hr_ref is not None else None
        for key in ("rise_time_sec","width50_sec","ibi_std_sec"):
            source_key = {"rise_time_sec":"upstroke_mean_sec", "width50_sec":"width50_mean_sec"}.get(key,key)
            features[site+"_"+key] = finite_stat([r.get("feature_"+source_key,np.nan) for r in selected])
    stable = [r for r in delays if r["face_roi"] == "combined" and r["hand_roi"] == "hand_palm" and
              r["method"] == "GREEN" and r["status"] == "candidate" and r["primary_nonoverlapping"]]
    scatter = finite_stat([r["delay_xcorr_sec"] for r in stable],np.std)
    floor = finite_stat([r["delay_uncertainty_floor_sec"] for r in stable])
    delay_stable = len(stable) >= 3 and scatter is not None and floor is not None and scatter <= floor
    features["delay_xcorr_sec"] = finite_stat([r["delay_xcorr_sec"] for r in stable]) if delay_stable else None
    features["delay_candidate_primary_windows"] = len(stable)
    features["delay_stability_std_sec"] = scatter
    features["delay_usable_as_exploratory_feature"] = delay_stable
    matched = identity.get("matched_row")
    excluded = matched and str(matched["normalized"].get("manual_exclude","")).casefold() == "yes"
    features["association_eligible"] = bool(not truncated and min(counts.values()) >= 3 and valid_refs and
                                             identity["match_status"] == "matched_session" and not excluded)
    features["eligibility_reason"] = "eligible_session_descriptive_only" if features["association_eligible"] else "partial_recording_or_insufficient_quality_or_reference_or_match_or_manual_exclude"
    return features


def process_one(job):
    import torch
    torch.set_num_threads(1)
    video,identity,records,settings = job
    out = Path(settings["out_dir"])/identity["recording_id"]
    out.mkdir(parents=True,exist_ok=True)
    summary = dict(subject_id=identity["subject_id"],recording_id=identity["recording_id"],status="failed")
    try:
        audit = inspect_video(video)
        dt = audit.get("median_frame_interval_sec")
        if not dt:
            raise ValueError(audit.get("pts_error","No validated frame clock"))
        fs = float(round(1/dt,6))
        matched = identity.get("matched_row")
        side = matched["normalized"].get("hand_side","") if matched else ""
        # Mirroring is undocumented. Lock the detected hand identity and retain
        # metadata disagreement; do not reject all detections by an unverified side.
        cfg = PipelineConfig(sample_rate=fs,expected_hand="any")
        raw = extract_raw(video,out,cfg,max_frames=settings["max_frames"],max_dimension=settings["max_dimension"],
                          reuse=settings["reuse"],private_overlays=settings["private_overlays"])
        traces = traces_from_arrays(raw)
        # Independent decoder timestamp agreement; remove only a constant origin.
        backend,source = raw["backend_timestamps"],raw["timestamps"]
        error = float(np.max(np.abs((backend-backend[0])-(source-source[0]))))
        timing_verified = bool(np.isfinite(error) and error < dt/2 and np.all(np.diff(backend) > 0))
        quality,delays,signals = analyze_signals(traces,cfg,out,timing_verified=timing_verified)
        frozen = frozen_recording(settings["checkpoint"],settings["freeze_manifest"],traces,raw,out,settings["frozen"])
        refs = align_references(identity,records,float(source[-1]))
        truncated = len(source) < audit["timing"]["sample_count"]
        features = session_features(identity,quality,delays,refs,truncated)
        summary.update(status="processed_partial" if truncated else "processed",decoded_frames=len(source),
                       input_sha256=json.loads(str(raw["identity_json"].item()))["input_sha256"],
                       raw_rgb_sha256=file_sha256(out/"raw_rgb.npz"),
                       classical_waveforms_sha256=file_sha256(out/"classical_waveforms.npz"),
                       analyzed_duration_sec=float(source[-1]-source[0]),full_duration_sec=audit.get("pts_duration_sec"),
                       backend_pts_max_relative_error_sec=error,timing_clock_verified=timing_verified,
                       face_valid_fraction=float(traces["combined"].validity_mask.mean()),
                       hand_valid_fraction=float(traces["hand_palm"].validity_mask.mean()),
                       metadata_hand_side=side,detected_hand_identities=sorted(set(raw["hand_identity"].tolist())),
                       handedness_note="Detector convention versus recorded side; mirroring unverified",
                       quality=signals,frozen=frozen,session_features=features,bp_references=refs,
                       delay_summary=json.loads((out/"delay_summary.json").read_text(encoding="utf-8")),
                       optical_delay_primary_candidates=sum(r["status"] == "candidate" and r["primary_nonoverlapping"] and
                                                            r["face_roi"] == "combined" and r["hand_roi"] == "hand_palm" and r["method"] == "GREEN" for r in delays))
        write_csv(out/"bp_reference_alignment.csv",refs)
        write_json(out/"pilot_summary.json",summary)
        write_json(out/"analysis_config.json",cfg.to_dict())
    except Exception as exc:
        import traceback
        summary.update(reason=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        write_json(out/"failure.json",summary)
    return summary


def write_report(path,manifest,summaries,bp=None):
    checkpoint = manifest["checkpoint"]
    lines = ["# Pilot-6 dual-ROI feasibility run", "", f"Run: `{manifest['output']}`", "",
             "BP accuracy and physiological PTT are not validated. No contact PPG/ECG reference is available.", "",
             f"Checkpoint SHA256: `{checkpoint.get('sha256','unavailable')}`.",
             f"Frozen inference: {checkpoint['face_inference']}; {checkpoint.get('reason') or ''}.", "",
             "| Recording | Status | Frames | Face valid | Hand valid | Clock verified |",
             "|---|---|---:|---:|---:|---|"]
    for r in summaries:
        lines.append(f"| {r['recording_id']} | {r['status']} | {r.get('decoded_frames','')} | {r.get('face_valid_fraction','')} | {r.get('hand_valid_fraction','')} | {r.get('timing_clock_verified','')} |")
        if r.get("reason"):
            lines.extend(["",f"Failure {r['recording_id']}: {r['reason']}",""])
    def number(value):
        return f"{value:.3f}" if isinstance(value,(int,float)) and np.isfinite(value) else "unavailable"
    lines.extend(["", "Classical GREEN descriptive results (quality labels are engineering screens):", "",
                  "| Recording | Face good/uncertain/invalid | Hand good/uncertain/invalid | Primary face HR | Primary hand HR | Primary delay candidates |",
                  "|---|---|---|---:|---:|---:|"])
    for r in summaries:
        quality = r.get("quality",[])
        face = next((q for q in quality if q["roi"] == "combined" and q["method"] == "GREEN"),{})
        hand = next((q for q in quality if q["roi"] == "hand_palm" and q["method"] == "GREEN"),{})
        counts = lambda q: "/".join(str(q.get(k,"")) for k in ("good","uncertain","invalid"))
        lines.append(f"| {r['recording_id']} | {counts(face)} | {counts(hand)} | {number(face.get('primary_hr_median_bpm'))} | {number(hand.get('primary_hr_median_bpm'))} | {r.get('optical_delay_primary_candidates','')} |")
    if bp:
        lines.extend(["",f"BP baseline status: **{bp['status']}**. BP accuracy is not validated.",bp.get("reason",bp.get("protocol",""))])
        if bp.get("metrics"):
            lines.extend(["", "Exploratory subject-held-out session associations, not clinical accuracy:", "",
                          "| Ablation | Model | Subjects | SBP MAE (mmHg) | DBP MAE (mmHg) |",
                          "|---|---|---:|---:|---:|"])
            for m in bp["metrics"]:
                lines.append(f"| {m['ablation']} | {m['model']} | {m['subjects']} | {number(m['sbp_subject_balanced_mae_mmHg'])} | {number(m['dbp_subject_balanced_mae_mmHg'])} |")
    lines.extend(["", "Metadata headers and all worksheets: `metadata_validation.json`; original and cached raw tables remain local in `private_metadata/`.",
                  "Cuff observations remain measurement rows in `bp_reference_alignment.csv`. Session last-two means are descriptive associations, never continuous labels.",
                  "Default mask images contain no participant pixels. Spatial resizing affects classical extraction; Frozen V1 requires native resolution and its original preprocessing contract.",
                  "Timing uses MP4 presentation timestamps and matched paired filter segments. Positive lag means hand lags face. Sub-frame correlation interpolation is not acquired temporal resolution.",
                  "10s/5s-hop windows are overlapping diagnostics; primary summaries use chronological nonoverlapping windows. 30s windows provide longer spectral support.",
                  "Quality: original validity >=90%, complete finite filtered window, spectral SNR >=0 dB, periodicity >=0.5, clipping <=10%, illumination CV <=0.05. These are unvalidated engineering screens.",
                  "Filter: 0.7–4 Hz, order 3 Butterworth forward/backward SOS; 1s edge rejection; gaps <=0.15s interpolated, longer gaps retained. HR band 0.7–3.5 Hz.",
                  "Delay uncertainty floor is max(native median/max window interval, half-window disagreement/2, phase-versus-xcorr discrepancy). It is a diagnostic lower bound, not a confidence interval. Rolling shutter/readout, exposure, algorithm polarity and model phase are not calibrated.",
                  "No arterial PWV is computed. Method agreement, model allocations, and coarse waveform features provide post-hoc diagnostics, not causal or physiological validation."])
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(args):
    import torch
    torch.set_num_threads(1)
    output = Path(args.out_dir)
    previous_manifest = None
    if args.reuse and (output/"experiment_manifest.json").is_file():
        previous_manifest = json.loads((output/"experiment_manifest.json").read_text(encoding="utf-8"))
    if output.exists() and not args.reuse:
        raise FileExistsError("Output exists; use a new --out-dir or --reuse for matching intermediates")
    output.mkdir(parents=True,exist_ok=True)
    mapping = json.loads(Path(args.metadata_mapping).read_text(encoding="utf-8"))
    private = output/"private_metadata"
    private.mkdir(exist_ok=True)
    metadata_path = args.metadata_file
    access = []
    if not metadata_path and args.google_sheet_id:
        try:
            metadata_path = fetch_google_workbook(args.google_sheet_id,private,os.environ.get(args.google_token_env))
            access.append(dict(status="success",read_only=True))
        except Exception as exc:
            access.append(dict(status="failed",error_type=type(exc).__name__,detail=str(exc)))
    tables,records,issues = {},[],[]
    if metadata_path:
        metadata_path = Path(metadata_path)
        try:
            tables = read_tables(metadata_path,mapping.get("header_rows"),
                                 mapping.get("csv_sheet_name") if metadata_path.suffix.lower() == ".csv" else None)
            records,issues = normalize_tables(tables,mapping)
            target = private/("source"+metadata_path.suffix.lower())
            if metadata_path.resolve() != target.resolve():
                shutil.copyfile(metadata_path,target)
            write_json(private/"raw_and_normalized.json",dict(tables=tables,records=records))
        except (OSError,ValueError,ImportError,TypeError) as exc:
            tables,records = {},[]
            issues.append(dict(reason="metadata_import_failed",error_type=type(exc).__name__,detail=str(exc)))
    else:
        issues.append(dict(reason="metadata_unavailable",detail="No readable local metadata or authorized workbook response"))
    metadata_validation = dict(source_file=str(metadata_path) if metadata_path else None,
            source_sha256=file_sha256(metadata_path) if metadata_path and metadata_path.is_file() else None,access=access,
            sheets={name:dict(headers=t["headers"],header_row=t["header_row"],
                              duplicate_headers=t["duplicate_headers"],source_nonempty_rows=len(t["rows"])) for name,t in tables.items()},
            normalized_records=len(records),issues=issues,
            missing_field_counts={key:sum(key in r["missing_fields"] for r in records) for key in {k for r in records for k in r["missing_fields"]}},
            raw_values="private_metadata/raw_and_normalized.json plus original source workbook; no source modification")
    write_json(output/"metadata_validation.json",metadata_validation)
    write_json(output/"resolved_metadata_mapping.json",mapping)
    ckpt,extractor = checkpoint_audit(args.checkpoint,args.freeze_manifest,args.frozen)
    write_json(output/"checkpoint_provenance.json",ckpt)
    timing = validate_timing(seed=args.seed)
    write_json(output/"synthetic_timing_validation.json",timing)
    if extractor is not None:
        try:
            write_json(output/"model_phase_probe.json",model_phase_probe(extractor))
        except (ValueError,RuntimeError) as exc:
            write_json(output/"model_phase_probe.json",dict(status="failed",reason=str(exc),physiological_phase_validated=False))
    else:
        write_json(output/"model_phase_probe.json",dict(status="blocked",reason=ckpt.get("reason") or "frozen_not_requested"))
    paths = sorted(p for p in Path(args.dataset_root).rglob("*") if p.suffix.lower() in (".mp4",".mov",".avi",".mkv"))
    if args.recording:
        paths = [p for p in paths if p.stem.casefold() == Path(args.recording).stem.casefold()]
        if len(paths) != 1:
            raise ValueError("Single recording selector must match exactly one discovered video")
    identities = [match_video(p,records) for p in paths]
    for identity in identities:
        if not identity["recording_id"] or not identity["subject_id"]:
            raise ValueError("Video identity must be resolved by filename convention or explicit mapping")
    ids = [i["recording_id"] for i in identities]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate recording identities; matching is ambiguous")
    dataset_audit = []
    for p,i in zip(paths,identities):
        try:
            audit = inspect_video(p)
        except Exception as exc:
            audit = dict(filename=p.name,pts_source="unavailable",error_type=type(exc).__name__,detail=str(exc))
        dataset_audit.append(dict(**audit,subject_id=i["subject_id"],recording_id=i["recording_id"],match_status=i["match_status"]))
    write_json(output/"dataset_audit.json",dataset_audit)
    write_json(private/"resolved_sessions.json",identities)
    write_csv(output/"video_session_manifest.csv",[dict(video=p.name,subject_id=i["subject_id"],recording_id=i["recording_id"],session_id=i["session_id"],
              match_status=i["match_status"],match_source="exact session ID/video stem; no row-order matching") for p,i in zip(paths,identities)])
    settings = vars(args).copy()
    settings["out_dir"] = str(output.resolve())
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),output=str(output.resolve()),
                    seed=args.seed,command=sys.argv,settings=settings,checkpoint=ckpt,
                    metadata_sha256=metadata_validation["source_sha256"],mapping_sha256=file_sha256(args.metadata_mapping),
                    provenance=provenance(),synthetic_timing_max_error_sec=timing["max_absolute_error_sec"],
                    python_version=sys.version,platform=platform.platform(),logical_cpu_count=os.cpu_count(),
                    torch_threads=torch.get_num_threads(),
                    environment={d:importlib.metadata.version(d) for d in
                                 ("numpy","scipy","pandas","openpyxl","opencv-python","mediapipe","torch","scikit-learn")})
    if previous_manifest is not None:
        old = previous_manifest
        if any(old.get(k) != manifest.get(k) for k in ("metadata_sha256","mapping_sha256")):
            raise ValueError("Metadata/mapping changed; use new output directory")
        raw_modules = ("pilot_video","pilot_detection","detection","region_masks","roi","pipeline","video","types","config")
        for name in raw_modules:
            key = str(Path("rppg_lab")/(name+".py"))
            if old["provenance"]["source_sha256"].get(key) != manifest["provenance"]["source_sha256"].get(key):
                raise ValueError(f"Raw extraction code changed ({name}); use a new output directory")
        for name in ("numpy","opencv-python","mediapipe"):
            if old["environment"].get(name) != manifest["environment"].get(name):
                raise ValueError(f"Raw extraction library changed ({name}); use a new output directory")
        history = output/"invocations"
        history.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        write_json(history/(stamp+"_previous_manifest.json"),old)
    write_json(output/"experiment_manifest.json",manifest)
    audit_only = args.dry_run or args.metadata_only
    summaries = []
    if not audit_only:
        jobs = [(str(p),i,records,settings) for p,i in zip(paths,identities)]
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                summaries = list(pool.map(process_one,jobs))
        else:
            summaries = [process_one(job) for job in jobs]
    references = [ref for p,i,audit in zip(paths,identities,dataset_audit)
                  for ref in align_references(i,records,audit.get("pts_duration_sec") or audit.get("nominal_duration_sec") or 0.)]
    write_csv(output/"bp_reference_alignment.csv",references)
    features = [s["session_features"] for s in summaries if s.get("session_features")]
    write_csv(output/"bp_session_features.csv",features)
    bp = exploratory_bp(features,args.seed)
    write_json(output/"bp_feasibility.json",bp)
    write_csv(output/"exploratory_bp_metrics.csv",bp.get("metrics",[]))
    write_csv(output/"exploratory_bp_predictions.csv",bp["predictions"])
    write_json(output/"cross_recording_summary.json",summaries)
    write_report(output/"research_report.md",manifest,summaries,bp)
    print(json.dumps(dict(output=str(output),recordings=len(paths),processed=sum(s["status"].startswith("processed") for s in summaries),
                          failures=sum(s["status"] == "failed" for s in summaries),metadata_sheets=list(tables),
                          cuff_readings=len(references),bp_status=bp["status"])),flush=True)
    return 1 if any(s["status"] == "failed" for s in summaries) else 0
