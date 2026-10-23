"""Versioned Pilot-6 facial experiment; never trains or evaluates a BP model."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys
import platform
from importlib.metadata import version
from datetime import datetime, timezone

import numpy as np

from .artifacts import file_sha256, write_json, provenance
from .pilot_metadata import read_tables, normalize_tables, match_video
from .pilot_video import inspect_video
from .waveform_inference import FrozenWaveformExtractor

OFFICIAL_MODEL_SHA256 = "d6b2d1eeebc0153197a2a12da736b957b32d72ae00b22631e9afd0b2097131ff"
OFFICIAL_MANIFEST_SHA256 = "da5ba397b3be389deec8821f8ea16944b5f5b93e2834bb79c3c147eb3d1f89dc"
DEFAULT_CHECKPOINT = "checkpoints/waveform_v1_official_ubfc/waveform_v1_frozen.pt"


def protected_hashes(checkpoint, manifest):
    paths = subprocess.check_output(["git", "ls-files", "rppg_lab", "scripts", "tests", "configs"],
                                    text=True).splitlines()
    paths += [str(checkpoint), str(manifest)]
    return {str(Path(p)): file_sha256(p) for p in paths if Path(p).is_file()}


def audit(videos, metadata_file, mapping_path, checkpoint, manifest, out):
    """Read-only audit always precedes acquisition and inference."""
    out = Path(out)
    from .detection import _choose_backend
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint)
    manifest = Path(manifest) if manifest else checkpoint.with_name("waveform_v1_freeze_manifest.json")
    # This exact unmodified API checks provenance, schema, strict state loading,
    # feature/ROI/prior order, extraction config, split and evaluation identities.
    extractor = FrozenWaveformExtractor(checkpoint, manifest)
    if extractor.model_sha256 != OFFICIAL_MODEL_SHA256 or file_sha256(manifest) != OFFICIAL_MANIFEST_SHA256:
        raise ValueError("This experiment is pinned to the restored official checkpoint AND original manifest")
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    tables = read_tables(metadata_file, mapping.get("header_rows"))
    records, issues = normalize_tables(tables, mapping)
    identities = []
    for video in videos:
        identity = match_video(video, records)
        identities.append({key: value for key, value in identity.items()
                           if key not in ("matched_row", "subject_row")})
    freeze = json.loads(manifest.read_text(encoding="utf-8"))
    source_changes, line_ending_changes = [], []
    for source, expected in freeze["training_source_provenance"]["source_sha256"].items():
        path = Path(source)
        if path.is_file() and file_sha256(path) != expected:
            lf = path.read_bytes().replace(b"\r\n", b"\n")
            if expected in {hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}:
                line_ending_changes.append(str(path))
            else:
                source_changes.append(str(path))
    runtime = provenance()
    runtime.update(python=sys.version, platform=platform.platform())
    runtime["versions"].update({name:version(name) for name in ("torch", "matplotlib", "openpyxl")})
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        model=extractor.metadata, checkpoint_path=str(checkpoint), manifest_path=str(manifest),
        manifest_sha256=file_sha256(manifest),
        compatibility_checks={name: "PASS" for name in (
            "frozen_status_and_decision", "checkpoint_manifest_hash", "official_identity_pin",
            "checkpoint_schema", "strict_state_dictionary_architecture", "roi_prior_channel_order",
            "source_candidate_and_epoch", "sample_rate_window_length", "model_extraction_configuration",
            "minimum_valid_fraction", "split_manifest_identity", "validation_locked_test_identity",
            "eval_mode")}, compatibility_failures=[],
        extraction_config=extractor.extraction_config.to_dict(),
        metadata=dict(path=str(metadata_file), sha256=file_sha256(metadata_file),
            mapping_path=str(mapping_path), mapping_sha256=file_sha256(mapping_path),
            worksheets={name: dict(headers=t["headers"], header_row=t["header_row"])
                        for name, t in tables.items()}, issues=issues, matches=identities),
        videos=[inspect_video(p) for p in videos],
        training_versions=freeze["training_source_provenance"]["versions"],
        historical_resolved_detector_backend="not recorded in freeze provenance",
        current_auto_detector_backend=_choose_backend(extractor.extraction_config.detector_backend),
        sources_different_from_training=source_changes,
        sources_with_only_line_ending_differences=line_ending_changes,
        runtime=runtime, protected_before=protected_hashes(checkpoint, manifest),
        boundaries=["No reference waveform; HR is a spectral estimate, not validated accuracy.",
            "No hand inference, physiological PTT or BP training/evaluation.",
            "Native geometry follows the frozen configuration; runtime/backend differences are reported.",
            "60-to-30 adaptation is an external input-domain experiment, not historical benchmark preprocessing."])
    write_json(out / "audit.json", report)
    return extractor, report


def native_extraction(arrays, extractor):
    from .extraction_cache import FaceROIExtraction
    from .processing import shared_grid
    from .types import RGBTrace, SharedTimebase
    t = arrays["camera_pts"]
    traces = {}
    for name in extractor.roi_names:
        quality = {key[len(name)+1:]: value for key, value in arrays.items()
                   if key.startswith(name + "_") and value.shape == t.shape}
        rgb = arrays[name + "_rgb"]
        traces[name] = RGBTrace(t.copy(), rgb.copy(), name, np.isfinite(rgb).all(axis=1), quality)
    return FaceROIExtraction(SharedTimebase(shared_grid(t, extractor.fs_target), t.copy()), traces)


def covered_seconds(intervals):
    """Union of half-open window intervals: overlap never creates more time."""
    total, end = 0.0, -np.inf
    for a, b in sorted(intervals):
        total += max(0., b - max(a, end))
        end = max(end, b)
    return float(total)


def waveform_diagnostics(waveform, fs):
    from scipy.integrate import trapezoid
    from .quality import signal_quality
    from .waveform_visualization import normalized_welch
    q = signal_quality(waveform, fs)
    spectrum = normalized_welch(waveform, fs)
    f, p = spectrum["frequency_hz"], spectrum["density"]
    band = (f >= .7) & (f <= 4)
    q["normalized_power_0_7_to_4_hz"] = float(trapezoid(p[band], f[band])) if band.any() else np.nan
    q["engineering_good"] = bool(np.isfinite(q["snr_db"]) and q["snr_db"] >= 0
                                  and np.isfinite(q["periodicity"]) and q["periodicity"] >= .5)
    q["quality_label"] = "good_engineering_only" if q["engineering_good"] else "uncertain"
    return q


def evaluate_native(arrays, extractor, out, timebase_config, recording_id):
    """All complete 10 s/5 s windows, including explicit failed window rows."""
    import torch
    from .frozen_timebase import adapt_native_face, validate_on_camera_clock
    from .processing import resample_trace
    from .window_priors import build_window_priors
    from .waveform_visualization import write_csv, coarse_feature_diagnostics
    from .pilot_frozen_plots import render_recording
    out = Path(out)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    raw = native_extraction(arrays, extractor)
    validation = validate_on_camera_clock(arrays["camera_pts"], extractor.extraction_config, timebase_config)
    write_json(out / "adapter_clock_validation.json", validation)
    if not validation["passed"]:
        raise ValueError("Synthetic adapter validation on this camera clock failed; inference stopped")
    extraction, adapter = adapt_native_face(raw, extractor.extraction_config, timebase_config)
    write_json(out / "timebase_adaptation.json", adapter)
    t, n = extraction.shared.timestamps, extractor.window_length_samples
    sampled = [resample_trace(extraction.traces[name], t, extractor.extraction_config.max_gap_sec)
               for name in extractor.roi_names]
    rgb = np.stack([values for values, _ in sampled])
    camera = arrays["camera_pts"]
    right = np.clip(np.searchsorted(camera, t), 0, len(camera)-1)
    exact = np.isclose(camera[right], t, atol=1e-8, rtol=0)
    left = np.where(exact, right, np.maximum(0, right-1))
    np.savez_compressed(out / "adapted_rgb.npz", camera_pts=arrays["camera_pts"], target_pts=t,
        rgb_30=rgb, roi_names=np.asarray(extractor.roi_names),
        target_interpolated=np.stack([flags for _, flags in sampled]),
        target_source_left_index=left, target_source_right_index=right,
        target_source_bracket_sec=camera[right]-camera[left],
        target_brackets_long_acquisition_interval=(camera[right]-camera[left]) > 1.5/60,
        filtered_at_camera_pts=np.stack([trace.values for trace in extraction.traces.values()]),
        effective_camera_valid=np.stack([trace.validity_mask for trace in extraction.traces.values()]))
    starts = list(range(0, len(t) - n + 1, int(5 * extractor.fs_target)))
    count, r, k = len(starts), len(extractor.roi_names), len(extractor.prior_names)
    waves = np.full((count, n), np.nan, np.float32)
    baseline = np.full((count, 3, n), np.nan, np.float32)
    baseline_roi = np.full((count, r, 3, n), np.nan, np.float32)
    attention, priors = np.full((count, r), np.nan), np.full((count, r, k), np.nan)
    source_fraction, valid_roi = np.full((count, r), np.nan), np.zeros((count, r), bool)
    valid_prior, residual = np.zeros((count, r, k), bool), np.full((count, r, n), np.nan)
    model_inputs = np.full((count, r, 9, n), np.nan, np.float32)
    timestamps = np.asarray([t[s:s+n] for s in starts])
    rows, feature_rows, allocation_rows = [], [], []
    methods = ("GREEN", "CHROM", "POS")
    state_before = {key: value.clone() for key, value in extractor._model.state_dict().items()}
    for wi, start in enumerate(starts):
        a, b = float(t[start]), float(t[start] + n / extractor.fs_target)
        common = dict(subject_id=recording_id.split("_")[0], recording_id=recording_id,
                      window_index=wi, start_sec=a, end_sec=b,
                      reference_waveform_available=False, bp_prediction_kind="unavailable_not_attempted")
        failure = None
        try:
            batch = extractor.prepare_window(extraction, a)
            prediction = extractor.infer_window(extraction, a)
            if prediction["waveform"].shape != (n,) or not np.isfinite(prediction["waveform"]).all():
                raise ValueError("Frozen output shape or finiteness failed")
            if not np.array_equal(prediction["timestamps"], t[start:start+n]):
                raise ValueError("Frozen output changed timestamps")
            for key in ("roi_attention", "prior_weights", "scaled_residuals"):
                if not np.isfinite(prediction[key]).all():
                    raise ValueError("Nonfinite frozen allocations/residuals")
            if not np.isclose(prediction["roi_attention"].sum(), 1, atol=1e-5):
                raise ValueError("Invalid attention normalization")
            if not np.allclose(prediction["prior_weights"].sum(axis=-1)[prediction["roi_valid"]], 1, atol=1e-5):
                raise ValueError("Invalid prior allocation normalization")
            waves[wi] = prediction["waveform"]
            attention[wi], priors[wi] = prediction["roi_attention"], prediction["prior_weights"]
            source_fraction[wi], valid_roi[wi] = prediction["roi_source_valid_fraction"], prediction["roi_valid"]
            valid_prior[wi], residual[wi] = prediction["prior_valid"], prediction["scaled_residuals"]
            model_inputs[wi] = batch["x"][0].numpy()
            for ri in np.flatnonzero(valid_roi[wi]):
                predictions, valid = build_window_priors(rgb[ri, start:start+n], extractor.fs_target, methods)
                baseline_roi[wi, ri, valid] = predictions[valid]
            for mi in range(3):
                supported = np.isfinite(baseline_roi[wi, :, mi]).all(axis=1)
                if supported.any():
                    baseline[wi, mi] = baseline_roi[wi, supported, mi].mean(axis=0)
            for ri, name in enumerate(extractor.roi_names):
                acq = (raw.traces[name].timestamps >= a) & (raw.traces[name].timestamps < b)
                for ki, prior in enumerate(extractor.prior_names):
                    allocation_rows.append(dict(**common, roi=name, prior=prior,
                        roi_valid=bool(valid_roi[wi, ri]), prior_valid=bool(valid_prior[wi, ri, ki]),
                        roi_attention=float(attention[wi, ri]), prior_weight=float(priors[wi, ri, ki]),
                        adapter_source_valid_fraction=float(source_fraction[wi, ri]),
                        acquisition_valid_fraction=float(raw.traces[name].validity_mask[acq].mean()),
                        interpretation="descriptive allocation; not causal or calibrated quality"))
            feature_rows.append(dict(**common, **coarse_feature_diagnostics(waves[wi], extractor.fs_target)))
        except ValueError as exc:
            failure = str(exc)
            # Do not keep any partially assigned successful prediction on failure.
            waves[wi], baseline[wi], baseline_roi[wi] = np.nan, np.nan, np.nan
            attention[wi], priors[wi], source_fraction[wi], residual[wi], model_inputs[wi] = np.nan, np.nan, np.nan, np.nan, np.nan
            valid_roi[wi], valid_prior[wi] = False, False
            allocation_rows = [row for row in allocation_rows if row["window_index"] != wi]
        for name, waveform in [("FrozenWaveformV1", waves[wi])] + list(zip(methods, baseline[wi])):
            valid = bool(np.isfinite(waveform).all())
            diagnostics = waveform_diagnostics(waveform, extractor.fs_target)
            if not valid:
                diagnostics.update(quality_label="invalid", engineering_good=False)
            row = dict(**common, method=name, inference_valid=valid,
                       failure_reason=failure or (None if valid else "classical_algorithm_failed"), **diagnostics)
            if name != "FrozenWaveformV1" and valid:
                row["signed_zero_lag_corr_with_frozen"] = float(np.corrcoef(waves[wi], waveform)[0, 1]) if np.std(waveform) > 1e-8 else None
            rows.append(row)
    unchanged = all(torch.equal(value, state_before[key]) for key, value in extractor._model.state_dict().items())
    if not unchanged or extractor._model.training:
        raise RuntimeError("Frozen state or evaluation mode changed during inference")
    np.savez_compressed(out / "waveform_windows.npz", timestamps=timestamps,
        waveform=waves, classical_waveforms=baseline, classical_roi_waveforms=baseline_roi,
        classical_methods=np.asarray(methods), roi_attention=attention, prior_weights=priors,
        roi_names=np.asarray(extractor.roi_names), prior_names=np.asarray(extractor.prior_names),
        roi_valid=valid_roi, prior_valid=valid_prior, adapter_source_valid_fraction=source_fraction,
        scaled_residuals=residual, model_inputs=model_inputs)
    write_csv(out / "window_metrics.csv", rows)
    write_csv(out / "allocations.csv", allocation_rows)
    write_csv(out / "morphology_diagnostics.csv", feature_rows)
    consistency = []
    hop = int(5 * extractor.fs_target)
    for wi in range(1, count):
        previous, current = waves[wi-1, hop:], waves[wi, :-hop]
        valid = bool(np.isfinite(previous).all() and np.isfinite(current).all()
                     and np.std(previous) > 1e-8 and np.std(current) > 1e-8)
        consistency.append(dict(recording_id=recording_id, previous_window_index=wi-1, window_index=wi,
            overlap_start_sec=float(timestamps[wi, 0]), overlap_duration_sec=5., valid=valid,
            signed_overlap_correlation=float(np.corrcoef(previous, current)[0,1]) if valid else None,
            failure_reason=None if valid else "invalid_or_flat_adjacent_window",
            interpretation="normalization/window-boundary sensitivity only; no reference accuracy"))
    write_csv(out / "adjacent_window_consistency.csv", consistency)
    summary = dict(recording_id=recording_id, frames=len(arrays["camera_pts"]),
        duration_sec=float(arrays["camera_pts"][-1] - arrays["camera_pts"][0]),
        native_shape=arrays["frame_shape"].tolist(), actual_detector_backend=str(arrays["actual_backend"].item()),
        camera_pts_preserved=adapter["camera_pts_preserved"], frozen_state_unchanged=unchanged,
        adapter_clock_validation_passed=validation["passed"],
        model_sha256=extractor.model_sha256, complete_windows=count,
        incomplete_tail_sec=float(t[-1] + 1/extractor.fs_target - (t[starts[-1]] + n/extractor.fs_target)) if starts else float(len(t)/extractor.fs_target),
        window_sec=n/extractor.fs_target, hop_sec=5, methods={},
        acquisition_valid_fraction={name:float(trace.validity_mask.mean()) for name,trace in raw.traces.items()},
        adjacent_overlap_median_signed_correlation=float(np.median([row["signed_overlap_correlation"] for row in consistency if row["valid"]]))
            if any(row["valid"] for row in consistency) else None,
        bp_model="not trained or evaluated", hand_inference="not attempted",
        phase="not validated for physiological timing", reference_waveform="unavailable")
    for name in ("FrozenWaveformV1", *methods):
        own = [row for row in rows if row["method"] == name]
        valid = [row for row in own if row["inference_valid"]]
        good = [row for row in valid if row["engineering_good"]]
        hrs = [row["hr_bpm"] for row in valid if np.isfinite(row["hr_bpm"])]
        summary["methods"][name] = dict(valid_windows=len(valid), invalid_windows=len(own)-len(valid),
            engineering_good_windows=len(good),
            usable_union_sec=covered_seconds([(row["start_sec"], row["end_sec"]) for row in valid]),
            good_union_sec=covered_seconds([(row["start_sec"], row["end_sec"]) for row in good]),
            hr_median_bpm=float(np.median(hrs)) if hrs else None,
            hr_min_bpm=float(min(hrs)) if hrs else None, hr_max_bpm=float(max(hrs)) if hrs else None)
    write_json(out / "summary.json", summary)
    render_recording(out, arrays, timestamps, waves, baseline, attention, priors, rows, extractor)
    return summary


def s004_decision(summary):
    """Engineering gate for cohort execution, not a physiological success test."""
    model = summary["methods"]["FrozenWaveformV1"]
    checks = dict(official_model=summary["model_sha256"] == OFFICIAL_MODEL_SHA256,
        state_unchanged=summary["frozen_state_unchanged"],
        native_clock_preserved=summary["camera_pts_preserved"],
        actual_clock_synthetic_validation=summary.get("adapter_clock_validation_passed", False),
        at_least_30_seconds_inferred=model["usable_union_sec"] >= 30,
        at_least_3_windows=model["valid_windows"] >= 3)
    return dict(decision="PROCESS_REMAINING_FIVE" if all(checks.values()) else "STOP_AND_INVESTIGATE",
        checks=checks, criteria="Prespecified technical feasibility gate; no waveform accuracy or BP criterion.",
        interpretation="Extending descriptive evaluation can characterize domain shift even when quality is uncertain.",
        observed_engineering_good_windows=model["engineering_good_windows"])


def _recording_job(video, out, checkpoint, manifest, timebase_config, reuse):
    from .pilot_frozen_extraction import extract_native_face
    extractor = FrozenWaveformExtractor(checkpoint, manifest)
    raw = extract_native_face(video, out, extractor.extraction_config, reuse)
    return evaluate_native(raw, extractor, out, timebase_config, Path(video).stem)


def write_comparative_report(out, summaries, failures, audit_report, decision, timebase_config=None):
    from .frozen_timebase import TimebaseConfig, filter_characteristics
    from .waveform_visualization import write_csv
    c = timebase_config or TimebaseConfig()
    fir = filter_characteristics(c)
    rows = []
    for summary in summaries:
        for method, metrics in summary["methods"].items():
            rows.append(dict(recording_id=summary["recording_id"], method=method,
                             frames=summary["frames"], duration_sec=summary["duration_sec"], **metrics))
    write_csv(out / "cross_recording_summary.csv", rows)
    write_json(out / "cross_recording_summary.json", dict(recordings=summaries, failures=failures, s004_decision=decision))
    lines = ["# Pilot-6 frozen facial evaluation with external timebase adaptation", "",
        f"Checkpoint SHA256: `{audit_report['model']['model_sha256']}`.",
        f"Manifest SHA256: `{audit_report['manifest_sha256']}`.", "",
        "The unchanged FrozenWaveformExtractor passed all recorded identity/configuration checks. "
        "Full-resolution three-ROI extraction replaces the earlier resized RGB cache. "
        "The versioned external adapter regularizes actual native PTS at 60 Hz, applies a centered "
        "anti-alias FIR, and retains filtered RGB on those PTS for the unchanged API's 30 Hz interpolation. "
        "V2 uses bounded natural cubic interpolation before/after the FIR; the API's final interpolation remains linear. "
        f"The FIR has {fir['taps']} symmetric taps, {fir['half_support_sec']:.3f} s half-support, unity DC, "
        f"and measured {fir['stopband_max_gain_db']:.2f} dB maximum stopband gain. "
        f"Full-support edges are excluded rather than padded. Short observation intervals ≤{1000*c.max_interpolation_gap_sec:.3f} ms may be interpolated "
        f"and flagged; longer holes split processing. The nominal {c.attenuation_db:g} dB design target is not a claim of {c.attenuation_db:g} dB rejection "
        "by the complete irregular-sampling system.", "",
        f"S004 execution gate: **{decision.get('decision', 'not_evaluated')}**. "
        "This gate establishes technical feasibility only; it does not validate pulse morphology or HR.", "",
        "| Recording | Frames | Model windows valid / total | Engineering good windows | Usable time union (s) | Good time union (s) | Median spectral HR (bpm) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for summary in summaries:
        m = summary["methods"]["FrozenWaveformV1"]
        hr = f"{m['hr_median_bpm']:.1f}" if m["hr_median_bpm"] is not None else "unavailable"
        lines.append(f"| {summary['recording_id']} | {summary['frames']} | {m['valid_windows']} / {summary['complete_windows']} | "
            f"{m['engineering_good_windows']} | {m['usable_union_sec']:.1f} | {m['good_union_sec']:.1f} | {hr} |")
    lines += ["", "Engineering good means SNR ≥ 0 dB and lag-periodicity ≥ 0.5 using the existing quality utility. "
        "Usable means the frozen API accepted the window and returned finite outputs. Neither label measures "
        "physiological accuracy. Time coverage is a union; overlapping windows are not independent observations.", "",
        "| Recording | Method | Valid | Engineering good | Median spectral HR (bpm) |", "|---|---|---:|---:|---:|"]
    for row in rows:
        hr = f"{row['hr_median_bpm']:.1f}" if row["hr_median_bpm"] is not None else "unavailable"
        lines.append(f"| {row['recording_id']} | {row['method']} | {row['valid_windows']} | {row['engineering_good_windows']} | {hr} |")
    lines += ["", "No continuous neural waveform was stitched. Each model prediction is an independent, "
        "signed 300-sample window at 30 Hz. Every scheduled complete window, including failure, is retained "
        "in window_metrics.csv; incomplete recording tails are reported in summary.json. Prior allocations "
        "and attention describe model operations and are not causal explanations or calibrated confidence.", "",
        "GREEN, CHROM and POS are the repository's window-local implementations, applied to exactly the same "
        "adapted RGB and supported facial ROIs. Each ROI's baseline is standardized using the unchanged prior "
        "builder, then averaged over successful ROIs. POS here is the existing single-window projection; "
        "the model's distinct POS_WIN prior is unchanged. Classical methods internally use the existing "
        "0.7–4 Hz zero-phase Butterworth filter. Neural output receives no additional filtering. "
        "Spectral HR searches 0.7–3.5 Hz. Agreement between methods is not ground-truth validation, "
        "and several classical signals also contribute to the model's priors.", "",
        "Limitations: no contact waveform reference; no learned phase validation; 30 Hz input spacing is "
        "33.33 ms; irregular acquisition and interpolation alter transfer functions. A centered FIR avoids "
        "its deterministic causal delay but cannot remove camera or neural timing uncertainty. The "
        "current MediaPipe runtime differs from training; the historical resolved backend was not recorded "
        "(current auto resolves to Tasks, recorded in audit and native_extraction.json), "
        "so extraction is configuration/geometry compliant rather than a claim of identical historical detector output. "
        "Frozen skin gating is disabled by contract: geometric ROIs do not guarantee semantic skin segmentation.", "",
        "No hand inference, inter-site physiological PTT, BP prediction, BP model training or BP accuracy "
        "evaluation was performed. Cuff measurements were not propagated to windows. Participant images "
        "were not saved; roi_geometry.png and roi_masks_examples.npz contain contributing masks only.", "",
        "Artifacts per recording: native_face_rgb.npz, native_extraction.json, adapted_rgb.npz, "
        "timebase_adaptation.json, waveform_windows.npz, window_metrics.csv, allocations.csv, "
        "morphology_diagnostics.csv, overview.png, prior_allocations.png, window_*.png and mask diagnostics.", "",
        "Next experiment: synchronized contact PPG/ECG plus camera timing/illumination controls, with "
        "simultaneous face and hand optical signals, before interpreting learned phase or collecting "
        "a larger BP training set. Add repeated recordings to quantify within-subject acquisition stability."]
    if failures:
        lines += ["", "Failures (retained):", *[f"- {f['recording_id']}: {f['error']}" for f in failures]]
    (out / "research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from .frozen_timebase import TimebaseConfig, filter_characteristics
    out = Path(args.out_dir)
    if out.exists() and not (args.reuse or args.reports_only):
        raise FileExistsError("Output directory exists; use a fresh namespace or explicit --reuse")
    out.mkdir(parents=True, exist_ok=True)
    videos = sorted(Path(args.dataset_root).glob("*.mp4"))
    if args.recording:
        videos = [p for p in videos if args.recording in (p.name, p.stem)]
    if not videos:
        raise ValueError("No exact requested recording found")
    cfg = TimebaseConfig(**json.loads(Path(args.adaptation_config).read_text(encoding="utf-8")))
    audit_out = out / "audit"
    index = 1
    while audit_out.exists():
        audit_out = out / f"audit_{index:02d}"
        index += 1
    extractor, report = audit(videos, args.metadata_file, args.metadata_mapping,
                             args.checkpoint, args.freeze_manifest, audit_out)
    write_json(audit_out / "adapter_filter.json", filter_characteristics(cfg))
    print(json.dumps(dict(audit=str(audit_out), compatibility="PASS", recordings=[p.stem for p in videos])), flush=True)
    if args.dry_run:
        return 0
    summaries, failures, decision = [], [], {}
    manifest = args.freeze_manifest or str(Path(args.checkpoint).with_name("waveform_v1_freeze_manifest.json"))
    def consume(video):
        directory = out / video.stem
        if args.reports_only:
            return json.loads((directory / "summary.json").read_text())
        return _recording_job(video, directory, args.checkpoint, manifest, cfg, args.reuse)
    s004 = next((p for p in videos if p.stem == "S004_V01"), None)
    if args.all_recordings and s004 is None:
        raise ValueError("Cohort execution requires S004_V01 as the first feasibility gate")
    if s004:
        try:
            summary = consume(s004)
            summaries.append(summary)
            decision = s004_decision(summary)
        except Exception as exc:
            failures.append(dict(recording_id=s004.stem, error=f"{type(exc).__name__}: {exc}"))
            decision = dict(decision="STOP_AND_INVESTIGATE", error=failures[-1]["error"])
        write_json(out / "s004_execution_decision.json", decision)
        print(json.dumps(decision), flush=True)
    remaining = [p for p in videos if p != s004]
    if args.all_recordings and decision.get("decision") != "PROCESS_REMAINING_FIVE":
        failures += [dict(recording_id=p.stem, error="not_attempted_S004_gate_failed") for p in remaining]
    elif remaining:
        if args.reports_only or args.workers == 1:
            for video in remaining:
                try:
                    summaries.append(consume(video))
                except Exception as exc:
                    failures.append(dict(recording_id=video.stem, error=f"{type(exc).__name__}: {exc}"))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                pending = {pool.submit(_recording_job, p, out/p.stem, args.checkpoint, manifest, cfg, args.reuse):p
                           for p in remaining}
                for future in as_completed(pending):
                    video = pending[future]
                    try:
                        summaries.append(future.result())
                    except Exception as exc:
                        failures.append(dict(recording_id=video.stem, error=f"{type(exc).__name__}: {exc}"))
    summaries.sort(key=lambda s:s["recording_id"])
    protected_after = protected_hashes(args.checkpoint, manifest)
    changes = [p for p, sha in report["protected_before"].items() if protected_after.get(p) != sha]
    write_json(out / "experiment_manifest.json", dict(audit=str(audit_out), invocation=vars(args),
        seed=42, adapter_config=cfg.__dict__, runtime=provenance(),
        checkpoint_sha256=extractor.model_sha256, freeze_manifest_sha256=file_sha256(manifest),
        protected_historical_files_unchanged=not changes, changed_protected_files=changes,
        recordings_completed=[s["recording_id"] for s in summaries], failures=failures,
        new_source_sha256={str(p):file_sha256(p) for p in [Path(__file__),
            Path(__file__).with_name("frozen_timebase.py"), Path(__file__).with_name("pilot_frozen_extraction.py"),
            Path(__file__).with_name("pilot_frozen_plots.py"), Path("scripts/evaluate_pilot6_frozen.py")]}))
    write_comparative_report(out, summaries, failures, report, decision, cfg)
    if changes:
        raise RuntimeError(f"Protected historical files changed: {changes}")
    return 1 if failures else 0
