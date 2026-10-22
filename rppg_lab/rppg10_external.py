"""Orchestration of the immutable Waveform v1 external HR evaluation.

Software smoke tests and ECG/structure QC do not consume the external dataset.
Only a complete, unlimited attempt can be REAL EXTERNAL. A persistent receipt
is reserved before the first model call and refuses another official attempt.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json

import cv2
import numpy as np
import pandas as pd
import torch

from .artifacts import file_sha256, json_safe, provenance, reserve_run, write_json
from .classical import DEFAULT_PRIORS
from .config import PipelineConfig
from .extraction_cache import DEFAULT_WAVEFORM_ROIS
from .rppg10 import (DATA_DOI, ECG_FS, EXPECTED_IDS, PAPER, ROI_MAP, SYNC_BOUND_SEC,
                     detect_r_peaks, ecg_reference, load_ecg, load_metadata,
                     load_subject_crops, scan_dataset, validate_roi_map)
from .rppg10_reporting import (CHROM, MODEL, aggregate_metrics, matched_comparisons,
                               plot_ecg_qc, plot_overview, plot_subject,
                               spectral_diagnostics, subgroup_metrics, temporal_metrics)
from .waveform_baselines import classical_window_predictions
from .waveform_inference import FrozenWaveformExtractor

OFFICIAL_SHA256 = "d6b2d1eeebc0153197a2a12da736b957b32d72ae00b22631e9afd0b2097131ff"
DEFAULT_CHECKPOINT = "checkpoints/waveform_v1_official_ubfc/waveform_v1_frozen.pt"
PROTOCOL_VERSION = "waveform_v1_rppg10_external_hr_v1"
METHODS = [MODEL, *[f"{r}/{p}" for r in DEFAULT_WAVEFORM_ROIS for p in DEFAULT_PRIORS],
           *[f"{p} multi-ROI average" for p in DEFAULT_PRIORS]]


def load_official_extractor(checkpoint):
    """Verify official bytes before deserialization/reconstruction/inference."""
    checkpoint = Path(checkpoint)
    if file_sha256(checkpoint) != OFFICIAL_SHA256:
        raise ValueError(f"Official frozen checkpoint SHA256 required: {OFFICIAL_SHA256}")
    adjacent = checkpoint.with_name("waveform_v1_freeze_manifest.json")
    manifest = json.loads(adjacent.read_text(encoding="utf-8"))
    if (manifest.get("freeze_decision") != "FREEZE_WAVEFORM_V1"
            or manifest.get("model_class") != "MultiROIPriorResidualWaveformNet"
            or manifest.get("fs") != 30. or manifest.get("win_sec") != 10.
            or manifest.get("window_length_samples") != 300):
        raise ValueError("Official freeze manifest architecture/fs/window contract mismatch")
    extractor = FrozenWaveformExtractor(checkpoint, adjacent, device="cpu")
    if (extractor.fs_target != 30. or extractor.window_length_samples != 300
            or extractor.roi_names != DEFAULT_WAVEFORM_ROIS or extractor.prior_names != tuple(DEFAULT_PRIORS)
            or extractor.channel_names != ("RGB_R", "RGB_G", "RGB_B", *DEFAULT_PRIORS)):
        raise ValueError("External adapter requires unchanged official ROI/prior/channel/30-Hz/10-second contract")
    return extractor, manifest


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ecg-fs", type=float, default=ECG_FS, help="1000 Hz verified in paper; never inferred from duration")
    p.add_argument("--metadata-path", help="Defaults to root/Subject Data.xlsx when present")
    p.add_argument("--roi-map", help='JSON object in frozen order, e.g. {"forehead":"Forehead","left_cheek":"Cheek2","right_cheek":"Cheek1"}')
    p.add_argument("--cache-dir", help="Optional local cache of raw RGB only, keyed by content/version")
    p.add_argument("--window-sec", type=float, default=10.)
    p.add_argument("--stride-sec", type=float, default=5.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--validate-only", action="store_true", help="Decode/ECG QC only; no frozen-model inference")
    p.add_argument("--max-subjects", type=int, default=0, help="Debug only; positive value forces SMOKE")
    p.add_argument("--smoke", action="store_true", help="Synthetic/debug data; never official external performance")
    return p


def validate_options(args):
    if args.window_sec != 10.:
        raise ValueError("window-sec must equal the unchanged frozen 10-second window (300 samples at 30 Hz)")
    if (not np.isfinite(args.stride_sec) or args.stride_sec <= 0
            or not np.isclose(args.stride_sec * 30, round(args.stride_sec * 30), atol=1e-8)):
        raise ValueError("stride-sec must be positive and lie on the frozen 30-Hz grid")
    if args.max_subjects < 0 or args.bootstrap_draws < 1 or args.threads < 1 or args.seed < 0:
        raise ValueError("Invalid max-subjects/bootstrap-draws/threads/seed")
    if not np.isfinite(args.ecg_fs) or not 100 <= args.ecg_fs <= 5000:
        raise ValueError("ecg-fs must be finite, documented and in [100,5000] Hz")
    smoke = args.smoke or args.max_subjects > 0
    if not smoke and args.ecg_fs != ECG_FS:
        raise ValueError("Official release is documented at 1000 Hz; another rate requires an explicitly labeled --smoke investigation")
    if not smoke and args.bootstrap_draws < 2000:
        raise ValueError("Official protocol requires at least 2000 subject bootstrap draws")
    mapping = validate_roi_map(json.loads(args.roi_map) if args.roi_map else None)
    if not smoke and mapping != ROI_MAP:
        raise ValueError("Official mapping is source-verified; an alternative mapping is SMOKE only")
    return smoke, mapping


def configuration(args, smoke, mapping):
    return dict(protocol_version=PROTOCOL_VERSION,
        result_type="VALIDATION_ONLY" if args.validate_only else "SMOKE" if smoke else "REAL EXTERNAL",
        seed=args.seed, ecg_fs_hz=args.ecg_fs, fs_hz=30., window_sec=10., window_samples=300,
        stride_sec=args.stride_sec, max_subjects=args.max_subjects, bootstrap_draws=args.bootstrap_draws,
        threads=args.threads, roi_order=list(DEFAULT_WAVEFORM_ROIS), roi_map=mapping,
        prior_order=list(DEFAULT_PRIORS), channel_order=["RGB_R", "RGB_G", "RGB_B", *DEFAULT_PRIORS],
        methods=METHODS, headline_baseline=CHROM, dataset_doi=DATA_DOI,
        sources=["https://data.mendeley.com/datasets/bx8982xgwt/1", "https://github.com/GRodrigues4/rPPG-10", PAPER],
        expected_usable_ids=list(EXPECTED_IDS), excluded_ids=["Subject_4"],
        timing=dict(source="nominal AVI FPS; independent common-start verification unavailable",
                    manual_sync_bound_sec=SYNC_BOUND_SEC, shifts_estimated=False,
                    label_sensitivity="all nominal and ±0.5-s ECG windows valid, max HR spread ≤3 bpm; nominal HR retained"),
        reference=dict(estimator="60 / median complete RR; both R peaks inside [start,end)", min_complete_rr=5,
                       rr_range_sec=[.30, 1.5], max_rr_cv=.20, max_rr_deviation_from_median=.30,
                       detector="abs_QRS_5_20Hz_v1; 300-ms refractory; 8-s noise blocks; height 5x median abs",
                       min_qrs_energy_fraction=.5, max_adc_rail_fraction=.05, max_flat_difference_fraction=.95),
        model_hr="unchanged signals.estimate_hr_welch(predicted waveform,30); finite/nonconstant guard",
        scheduling="all starts in the published 600 seconds and any longer decoded common recording; unsupported tails remain failures",
        coverage="finite prediction/all scheduled windows; accuracy and within3/5 use valid pairs only",
        subject_balancing="mean of scorable subject metrics; unscorable subject IDs disclosed; coverage includes all scheduled subjects",
        diagnostic_selection="per-subject median/worst ECG HR error plus subject-seeded random valid window; disclosed",
        quality="reference-independent spectral diagnostics and source support only; quality head omitted, uncalibrated",
        subgroup_bins=dict(illumination="midpoint of start/end lux; <800, 800–<1100, ≥1100; no continuous lighting claim",
                           uncertainty="participant bootstrap when n≥3; counts always shown; no fairness inference"),
        scientific_boundary="HR-only electrical reference; no optical waveform ground truth, morphology, BP or physiological timing validity")


def reserve_receipt(out, config, inventory, checkpoint_sha, manifest_sha, ledger_path=None):
    """Exclusive, immutable receipt. Status/completion is written separately."""
    serialized = json.dumps(json_safe(config), sort_keys=True, separators=(",", ":"))
    receipt = dict(protocol_version=PROTOCOL_VERSION, result_type=config["result_type"],
                   external_dataset_consumed=config["result_type"] == "REAL EXTERNAL",
                   reserved_at_utc=datetime.now(timezone.utc).isoformat(),
                   dataset_doi=DATA_DOI, dataset_identity_sha256=inventory["data_identity_sha256"],
                   checkpoint_sha256=checkpoint_sha, freeze_manifest_sha256=manifest_sha,
                   split="external full usable cohort; no train/validation/test partition",
                   immutable_configuration=config,
                   configuration_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
                   output=str(Path(out).resolve()),
                   policy="This attempt consumes rPPG-10 as independent evidence for frozen v1. Future tuning requires different independent evaluation evidence.")
    if receipt["external_dataset_consumed"]:
        if ledger_path is None:
            ledger_path = Path(__file__).resolve().parents[1] / "outputs" / f"rppg10_external_consumed_{OFFICIAL_SHA256}.json"
        ledger_path = Path(ledger_path)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        # Lock by frozen version/dataset release, not output path or byte variants.
        with ledger_path.open("x", encoding="utf-8") as f:
            json.dump(json_safe(receipt), f, indent=2, allow_nan=False)
    with (Path(out) / "external_evaluation_receipt.json").open("x", encoding="utf-8") as f:
        json.dump(json_safe(receipt), f, indent=2, allow_nan=False)
    return receipt


def _base_row(sid, start, reference, extraction_status):
    return dict(subject_id=sid, start_sec=float(start), end_sec=float(start+10), minute=int((start+5)//60),
                reference_hr_bpm=reference.get("hr_bpm", np.nan), reference_valid=reference.get("valid", False),
                reference_reason=reference.get("reason", "missing_ecg"), rr_count=reference.get("rr_count", 0),
                rr_cv=reference.get("rr_cv", np.nan), qrs_energy_fraction=reference.get("qrs_energy_fraction", np.nan),
                sync_hr_range_bpm=reference.get("sync_hr_range_bpm", np.nan), sync_bound_sec=SYNC_BOUND_SEC,
                extraction_status=extraction_status, valid_fraction=np.nan,
                residual_mean_abs=np.nan, residual_max_abs=np.nan,
                **{f"roi_weight_{r}": np.nan for r in DEFAULT_WAVEFORM_ROIS},
                **{f"roi_valid_{r}": False for r in DEFAULT_WAVEFORM_ROIS},
                **{f"prior_weight_{r}_{p}": np.nan for r in DEFAULT_WAVEFORM_ROIS for p in DEFAULT_PRIORS})


def evaluate_subject(extractor, extraction, peaks, sid, starts):
    """Use the existing frozen API and existing exact-tensor baselines only.

    ECG is accessed only after/alongside independent optical inference. Its
    labels never enter prepare_window, infer_window, support or attention.
    """
    rows, windows = [], {}
    for start in starts:
        support = extraction is not None and int(round(start*30)) + 300 <= len(extraction.shared.timestamps)
        reference = ecg_reference(peaks, start, start+10) if peaks is not None else dict(reason="missing_ecg", valid=False)
        row = _base_row(sid, start, reference, "ok" if support else "missing_or_unsupported_video")
        signals = {}
        if support:
            try:
                # Preparing again for the baseline is intentional: it invokes
                # the same exact window-local prior/support code as infer_window.
                batch = extractor.prepare_window(extraction, start)
                signals = dict(classical_window_predictions(batch["x"][0].numpy(), batch["roi_valid"][0].numpy(),
                    batch["prior_valid"][0].numpy(), extractor.roi_names, extractor.prior_names))
                prediction = extractor.infer_window(extraction, start)
                signals[MODEL] = prediction["waveform"]
                row.update(valid_fraction=float(np.mean(prediction["roi_source_valid_fraction"])),
                           residual_mean_abs=float(np.mean(abs(prediction["scaled_residuals"]))),
                           residual_max_abs=float(np.max(abs(prediction["scaled_residuals"]))),
                           **{f"roi_weight_{r}": float(prediction["roi_attention"][i]) for i, r in enumerate(extractor.roi_names)},
                           **{f"roi_valid_{r}": bool(prediction["roi_valid"][i]) for i, r in enumerate(extractor.roi_names)},
                           **{f"prior_weight_{r}_{p}": float(prediction["prior_weights"][i,k]) for i,r in enumerate(extractor.roi_names)
                              for k,p in enumerate(extractor.prior_names)})
                windows[float(start)] = dict(time=prediction["timestamps"], model=prediction["waveform"], chrom=signals[CHROM])
            except ValueError as exc:
                if "No usable ROI" not in str(exc):
                    raise  # Contract errors must fail, not silently become missing predictions.
                row["signal_failure_reason"] = "no_usable_roi"
        for method in METHODS:
            diagnostic = spectral_diagnostics(signals.get(method, np.full(300, np.nan)))
            pred = diagnostic["pred_hr_bpm"]
            error = pred - row["reference_hr_bpm"] if row["reference_valid"] and np.isfinite(pred) else np.nan
            rows.append(dict(row, method=method, **diagnostic, error_bpm=error, absolute_error_bpm=abs(error)))
    return rows, windows


def write_summary_markdown(out, summary, inventory):
    lines = ["# Frozen Waveform v1 / rPPG-10 external HR evaluation", "",
             f"**{summary['result_type']} — {summary['completion_status']}**", "",
             "ECG is an electrical HR reference. No ECG/optical waveform morphology, BP or transit-time validation is performed.", "",
             f"Checkpoint SHA256: `{OFFICIAL_SHA256}`. Configuration and data identity are locked in the receipt.", "",
             f"Expected participants: 26; discovered: {len(inventory['discovered_usable_ids'])}; missing: {inventory['missing_ids']}. Subject_4 excluded.", "",
             "| Method | Scorable subjects | Subject-balanced MAE [95% CI] | Window MAE | Window RMSE | Bias | Valid pairs | Finite coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in [MODEL, CHROM, "GREEN multi-ROI average", "POS_WIN multi-ROI average", "PBV multi-ROI average", "CHROM multi-ROI average", "OMIT multi-ROI average"]:
        r = summary["metrics"]["methods"][method]
        s, w, ci = r["subject_balanced"], r["window_weighted"], r["subject_balanced_ci95"]["mae_bpm"]
        interval = f"[{ci['lower_95']:.3f}, {ci['upper_95']:.3f}]" if ci else "unavailable"
        lines.append(f"| {method} | {r['scorable_subjects']}/{r['subjects']} | {s['mae_bpm']:.3f} {interval} | {w['mae_bpm']:.3f} | {w['rmse_bpm']:.3f} | {w['bias_bpm']:.3f} | {w['valid_pairs']}/{w['windows']} | {w['finite_prediction_coverage']:.3%} |")
    lines += ["", "All methods, percentiles, within-3/5 rates, missingness, both weighting conventions and subject bootstrap intervals are in external_summary.json and CSVs.", "",
              "Accuracy denominators contain valid ECG/prediction pairs only. Coverage includes every scheduled window, including missing subjects and unsupported tails. Subject-balanced accuracy is conditional on scorable subjects; their IDs/counts are disclosed.", "",
              "Windows overlap and are autocorrelated. Bootstrap resamples whole subjects (2000 draws by default, seed 42), not independent windows. Sparse strata do not support fairness conclusions.", "",
              "AVI clocks are nominal. ECG/video synchronization was manual (paper reports up to 500 ms). Nominal median-complete-RR labels are accepted only when ECG-only ±0.5-s checks are valid and HR spread is ≤3 bpm. No shifts are fitted to model agreement.", "",
              "The quality head is uncalibrated and omitted. Attention, priors, SNR, periodicity and residual magnitude are descriptive. ECG detector audits are in figures/*_ecg_qc.png; simple automatic beat detection still has limitations.", "",
              "Diagnostic windows are the median, worst and seeded random valid windows per subject; ECG-error selection is explicitly disclosed. Invalid gaps are never interpolated in HR plots.", "",
              "This external attempt consumes rPPG-10 as independent evidence for this frozen version. Future tuning on it needs different independent evaluation evidence. No model/preprocessing changes are permitted based on these outcomes."]
    (Path(out) / "external_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    smoke, mapping = validate_options(args)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    cfg = configuration(args, smoke, mapping)
    # Official hash/config checks happen before dataset inference is reachable.
    extractor, manifest = (None, None) if args.validate_only else load_official_extractor(args.checkpoint)
    config = extractor.extraction_config if extractor else PipelineConfig(regions=("face",), face_roi="forehead", face_rois=DEFAULT_WAVEFORM_ROIS)
    inventory = scan_dataset(args.dataset_root, mapping)
    selected = inventory["subjects"][:args.max_subjects] if args.max_subjects else inventory["subjects"]
    if smoke:
        selected = [r for r in selected if r["present"]]
    if not selected or not any(r["structure_usable"] for r in selected):
        raise ValueError("No structurally usable subjects; run cannot be labeled successful")
    metadata, metadata_qc = {}, dict(status="not_available")
    metadata_path = Path(args.metadata_path) if args.metadata_path else Path(args.dataset_root) / "Subject Data.xlsx"
    if metadata_path.exists():
        try:
            metadata, metadata_qc = load_metadata(metadata_path)
        except (ValueError, ImportError, OSError, IndexError) as exc:
            metadata_qc = dict(status="invalid_or_unreadable", reason=str(exc))
    out = reserve_run(args.out_dir)
    (out / "figures").mkdir()
    (out / "overview").mkdir()
    write_json(out / "dataset_inventory.json", inventory)
    source = provenance()
    for package in ("torch", "pandas", "openpyxl", "matplotlib"):
        try:
            source["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            source["versions"][package] = None
    frozen_hash = file_sha256(args.checkpoint) if extractor else None
    provenance_record = dict(configuration=cfg, software=source, metadata_qc=metadata_qc,
        checkpoint_sha256=frozen_hash, freeze_manifest=manifest,
        dataset_identity_sha256=inventory["data_identity_sha256"],
        dataset_root=str(Path(args.dataset_root).resolve()), cache_dir=str(args.cache_dir) if args.cache_dir else None,
        execution_started_at_utc=datetime.now(timezone.utc).isoformat(), client_timezone="Asia/Tehran")
    write_json(out / "config_and_provenance.json", provenance_record)
    receipt = None
    if extractor:
        receipt = reserve_receipt(out, cfg, inventory, frozen_hash,
                                  file_sha256(Path(args.checkpoint).with_name("waveform_v1_freeze_manifest.json")))
    all_rows, all_qc, diagnostic_selection = [], {}, []
    for record in selected:
        sid = record["subject_id"]
        print(f"{sid}: structure/video/ECG QC" + (" + frozen inference" if extractor else " only"), flush=True)
        extraction, peaks = None, None
        qc = dict(structure_usable=record["structure_usable"], warnings=list(record["qc"]))
        if all(name in record["files"] and record["files"][name]["bytes"] for name in DEFAULT_WAVEFORM_ROIS):
            try:
                extraction, qc["video"] = load_subject_crops(args.dataset_root, record, config, args.cache_dir)
            except (ValueError, OSError, cv2.error) as exc:
                qc["video_error"] = str(exc)
        if "ecg" in record["files"]:
            try:
                ecg, ecg_info = load_ecg(Path(args.dataset_root) / record["files"]["ecg"]["path"], args.ecg_fs)
                peaks = detect_r_peaks(ecg, args.ecg_fs)
                qc["ecg"] = dict(ecg_info, detector=peaks.qc)
                plot_ecg_qc(peaks, sid, out / "figures")
            except (ValueError, OSError) as exc:
                qc["ecg_error"] = str(exc)
        duration = len(extraction.shared.timestamps)/30 if extraction else 0
        # Published 600-s schedule exposes lost/missing tails in all coverage rates.
        scheduled_duration = duration if smoke else max(600., duration)
        count = max(0, int(np.floor((scheduled_duration - 10) / args.stride_sec + 1e-8)) + 1)
        starts = np.arange(count) * args.stride_sec
        refs = [ecg_reference(peaks, t, t+10) for t in starts] if peaks is not None else []
        qc["reference_window_counts"] = dict(Counter(r["reason"] for r in refs)) if peaks is not None else {"missing_ecg": len(starts)}
        qc["scheduled_windows"] = len(starts)
        qc["supported_video_windows"] = sum(int(round(t*30))+300 <= len(extraction.shared.timestamps) for t in starts) if extraction else 0
        if peaks and extraction:
            qc["ecg_minus_common_video_duration_sec"] = len(peaks.raw)/peaks.fs - duration
            if abs(qc["ecg_minus_common_video_duration_sec"]) > 1/30 + 1e-6:
                qc["warnings"].append("ecg_video_duration_mismatch; unsupported ECG/video spans remain invalid")
        all_qc[sid] = qc
        if extractor:
            rows, windows = evaluate_subject(extractor, extraction, peaks, sid, starts)
            all_rows.extend(rows)
            frame = pd.DataFrame(rows)
            if len(frame) and peaks is not None:
                diagnostic_selection.extend(plot_subject(frame, sid, windows, peaks, out / "figures", args.seed))
        write_json(out / "ecg_qc.json", all_qc)
    if args.validate_only:
        write_json(out / "validation_only_receipt.json", dict(result_type="VALIDATION_ONLY", inference_calls=0,
            dataset_consumed=False, data_identity_sha256=inventory["data_identity_sha256"], configuration=cfg))
        print(f"VALIDATION_ONLY saved: {out}", flush=True)
        return dict(result_type="VALIDATION_ONLY", qc=all_qc, inventory=inventory)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(out / "external_window_metrics.csv", index=False)
    subjects, metrics = aggregate_metrics(all_rows, [r["subject_id"] for r in selected], args.bootstrap_draws, args.seed)
    subjects.to_csv(out / "external_subject_metrics.csv", index=False)
    groups = subgroup_metrics(subjects, metadata, args.seed, args.bootstrap_draws)
    plot_overview(frame, subjects, groups, out / "overview")
    model = metrics["methods"][MODEL]
    completion = "COMPLETE" if model["window_weighted"]["valid_pairs"] else "COMPLETE_NO_VALID_HR_PAIRS"
    summary = dict(result_type=cfg["result_type"], completion_status=completion,
        official_external_evaluation=cfg["result_type"] == "REAL EXTERNAL",
        successful_official_evaluation=cfg["result_type"] == "REAL EXTERNAL" and model["scorable_subjects"] == 26,
        performance_acceptance_threshold="none; report outcomes, including failure; completion is not scientific success",
        evaluated_subject_ids=[r["subject_id"] for r in selected], missing_subject_ids=inventory["missing_ids"],
        metrics=metrics, matched_window_comparisons=matched_comparisons(frame),
        recording_minute_metrics=temporal_metrics(frame), descriptive_subgroups=groups,
        diagnostic_selection=diagnostic_selection, metadata_qc=metadata_qc,
        reference_rejection_counts=dict(sum((Counter(q["reference_window_counts"]) for q in all_qc.values()), Counter())),
        receipt_sha256=file_sha256(out / "external_evaluation_receipt.json"),
        limitations=["ECG provides no optical morphology ground truth", "nominal clocks and manual synchronization ±0.5 s",
                     "automatic ECG detector, no clinical adjudication", "young, small, sparse skin strata; no broad fairness inference",
                     "low-resolution crops differ from UBFC landmark masks; domain shift",
                     "source-valid frame means do not detect every tracking/occlusion/defocus failure",
                     "overlapping windows are autocorrelated; no clinical/BP/timing validity claim"])
    write_json(out / "external_summary.json", summary)
    write_summary_markdown(out, summary, inventory)
    if file_sha256(args.checkpoint) != frozen_hash:
        raise RuntimeError("Checkpoint changed during read-only evaluation")
    write_json(out / "completion_verification.json", dict(completion_status=completion,
        checkpoint_sha256_after=frozen_hash, immutable_configuration_sha256=receipt["configuration_sha256"],
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        artifact_sha256={str(p.relative_to(out)): file_sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}))
    print(f"{cfg['result_type']}: {completion}; saved {out}", flush=True)
    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(args)
