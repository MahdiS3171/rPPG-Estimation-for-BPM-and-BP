"""Validation-only visual inspection of the immutable, official Waveform v1.

Inference, priors, targets and evaluation use the existing Part 3 interfaces.
Only diagnostic selection and plotting happen here; no training or test runner
is reachable. Signals retain native polarity. All shifts are discrete evaluation
slices, never a timing estimate, learned correction or beat-specific alignment.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, welch
import torch

from .artifacts import file_sha256, json_safe, provenance, reserve_run, write_json
from .datasets import UBFCMultiROIRPPGDataset, UBFCSubject
from .extraction_cache import (FaceROIExtraction, _validate, extraction_cache_identity)
from .signals import bandpass, estimate_hr_welch, pulse_wave_features, standardize_1d
from .splits import validate_waveform_split_manifest
from .types import RGBTrace, SharedTimebase
from .waveform_alignment import aligned_overlap
from .waveform_baselines import classical_window_predictions, target_sha256
from .waveform_inference import FrozenWaveformExtractor
from .waveform_metrics import (aggregate_waveform_metrics, forward_batch,
                               waveform_window_metrics)
from .waveform_training import checkpoint_dataset_config, split_manifest_hash

OFFICIAL_MODEL_SHA256 = "d6b2d1eeebc0153197a2a12da736b957b32d72ae00b22631e9afd0b2097131ff"
DEFAULT_CHECKPOINT = "checkpoints/waveform_v1_official_ubfc/waveform_v1_frozen.pt"
DEFAULT_SPLIT = "checkpoints/waveform_v1_official_ubfc/waveform_v1_split.json"
MODEL = "Waveform v1"
CHROM = "CHROM_WIN multi-ROI average"
SELECTION_NOTICE = "REFERENCE-BASED DIAGNOSTIC SELECTION"
ALIGNMENT_NOTICE = "Aligned for waveform-shape comparison only; not an inter-site timing measurement."
FEATURES = {
    "hr_bpm": "HR (bpm)",
    "ibi_mean_sec": "IBI (s)",
    "width50_mean_sec": "W50 (s)",
    "upstroke_mean_sec": "Rise time (s)",
    "fall_time_mean_sec": "Fall time (s)",
    "upstroke_slope_mean": "Normalized rise slope (z/s)",
    "ibi_std_sec": "Beat-to-beat IBI SD (s)",
    "ibi_cv": "Beat-to-beat IBI CV (ratio)",
}
LIMITATIONS = [
    "Visual similarity is not clinical BP validation; observed consistency is not BP accuracy.",
    "ECG and PPG are not interchangeable morphology targets.",
    "Alignment is for comparison, not delay estimation, PAT/PTT or face-hand timing.",
    "UBFC finger PPG is not an exact facial waveform ground truth; the sites can differ physiologically.",
    "Detailed notch/RI/APG morphology remains unvalidated and is not measured here.",
    "Attention/prior weights are descriptive allocations, not causal or physiological explanations; "
    "the quality head is not calibrated and is omitted.",
    "Reference-based best/worst examples must not influence checkpoint selection or model tuning.",
    "True physiology can change within a recording; consistency does not require constant features.",
]


def validation_subject_ids(manifest: dict, subject: str | None = None) -> list[str]:
    """Validate exact membership before any data/cache discovery or access."""
    validate_waveform_split_manifest(manifest)
    ids = sorted(manifest["validation_ids"])
    if subject is not None:
        if subject in manifest["test_ids"]:
            raise ValueError(f"Locked-test subject {subject!r} is forbidden; the test is already consumed")
        if subject not in ids:
            raise ValueError(f"{subject!r} is not an exact validation subject ID; allowed: {', '.join(ids)}")
        return [subject]
    return ids


def verify_frozen_identity(checkpoint_path, split_path, freeze_path=None):
    """Require exact official bytes, frozen status, adjacent manifest and split.

    There is deliberately no allow-candidate/override-hash option. An explicitly
    supplied manifest must be byte-identical to the mandatory adjacent copy.
    FrozenWaveformExtractor performs the established reconstruction/config checks.
    """
    checkpoint_path, split_path = Path(checkpoint_path), Path(split_path)
    adjacent = checkpoint_path.with_name("waveform_v1_freeze_manifest.json")
    freeze_path = Path(freeze_path) if freeze_path else adjacent
    for description, path in (("official frozen checkpoint", checkpoint_path),
                              ("adjacent freeze manifest", adjacent),
                              ("freeze manifest", freeze_path), ("saved split manifest", split_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}. Supply the existing official artifacts; "
                                    "this tool cannot train, freeze or simulate results.")
    if file_sha256(checkpoint_path) != OFFICIAL_MODEL_SHA256:
        raise ValueError("Checkpoint SHA256 differs from official frozen Waveform v1: " + OFFICIAL_MODEL_SHA256)
    if file_sha256(adjacent) != file_sha256(freeze_path):
        raise ValueError("Supplied freeze manifest differs from the mandatory adjacent freeze manifest")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    validation_subject_ids(manifest)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("waveform_v1_status") != "frozen_v1"
            or checkpoint.get("freeze_decision") != "FREEZE_WAVEFORM_V1"
            or freeze.get("status") != "frozen_v1"
            or freeze.get("freeze_decision") != "FREEZE_WAVEFORM_V1"
            or freeze.get("model_sha256") != OFFICIAL_MODEL_SHA256
            or checkpoint.get("selected_epoch") != 25 or freeze.get("selected_epoch") != 25):
        raise ValueError("Require the official frozen_v1 status, freeze decision and selected epoch 25")
    config = checkpoint_dataset_config(checkpoint, manifest)
    split_hash = file_sha256(split_path)
    if (checkpoint.get("split_manifest_file_sha256") != split_hash
            or freeze.get("split_manifest_file_sha256") != split_hash
            or freeze.get("split_manifest_sha256") != split_manifest_hash(manifest)
            or any(freeze.get(k) != manifest[k] for k in ("train_ids", "validation_ids", "test_ids"))):
        raise ValueError("Saved split file/partition identity differs from the frozen experiment")
    extractor = FrozenWaveformExtractor(checkpoint_path, freeze_path, device="cpu")
    return extractor, checkpoint, manifest, config, freeze_path


def validation_subject_paths(root, manifest: dict) -> list[UBFCSubject]:
    """Open only allowlisted direct children; never enumerate cohort folders.

    Reject redirected subject/video/reference paths so an alias cannot lead to
    a locked-test folder. Metadata and cache access have the same allowlist.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"UBFC root missing: {root}; provide existing UBFC validation data")
    subjects = []
    for sid in validation_subject_ids(manifest):
        if Path(sid).name != sid or sid in (".", ".."):
            raise ValueError("Validation ID must name a direct UBFC subject folder")
        folder = root / sid
        if folder.resolve() != folder or not folder.is_dir():
            raise FileNotFoundError(f"Missing or redirected validation folder: {folder}")
        video = next((folder / name for name in ("vid.avi", "video.avi", "video.mp4", "vid.mp4")
                      if (folder / name).is_file()), None)
        gt = folder / "ground_truth.txt"
        if video is None or not gt.is_file():
            raise FileNotFoundError(f"Validation subject {sid} needs a video and ground_truth.txt under {folder}")
        if video.resolve() != video or gt.resolve() != gt:
            raise ValueError(f"Redirected validation video/reference is forbidden: {sid}")
        subjects.append(UBFCSubject(sid, video, gt))
    return subjects


def require_phase1_cache(subject: UBFCSubject, config: dict, cache_dir) -> dict:
    """Preflight exact existing cache bytes using Phase 1 identity/validation.

    Reading and validating before constructing the dataset prevents its normal
    cache-miss/rebuild path. No video extraction is requested by this tool.
    The small raw arrays are reread by the unmodified dataset cache interface.
    """
    root = Path(cache_dir)
    identity = extraction_cache_identity(subject.video_path, config["extraction_config"], config["max_frames"])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = root / ("phase1_face_" + key + ".npz")
    if not path.is_file():
        raise FileNotFoundError(f"Missing matching Phase 1 extraction cache for validation {subject.subject_id}: "
                                f"{path}. Restore the original cache/environment; no extraction or synthetic fallback is run.")
    try:
        with np.load(path, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata_json"].item()))
            if metadata["identity"] != identity:
                raise ValueError("cache identity mismatch")
            source, grid = arrays["timestamps_original"], arrays["timestamps_uniform"]
            traces = {}
            for name in arrays["face_roi_names"].tolist():
                prefix = "face_roi_" + name + "_"
                quality = {k: arrays[prefix + k] for k in metadata["quality_keys"][name]}
                traces[name] = RGBTrace(source, arrays[prefix + "rgb"], metadata["trace_roi_names"][name],
                                        arrays[prefix + "valid"], quality)
            _validate(FaceROIExtraction(SharedTimebase(grid, source), traces), config["extraction_config"])
    except Exception as exc:
        raise ValueError(f"Invalid Phase 1 cache for validation {subject.subject_id}: {path}: {exc}") from exc
    return dict(subject_id=subject.subject_id, cache_path=str(path.resolve()), cache_sha256=file_sha256(path),
                video_path=str(subject.video_path), reference_path=str(subject.gt_path))


def signed_zscore(signal) -> np.ndarray:
    """Independent display scaling only: no polarity selection or gap filling."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or x.size < 2 or x.std() <= 1e-8:
        return np.full(x.shape, np.nan)
    return standardize_1d(x)


def diagnostic_score(metrics: dict) -> float:
    return .5 * metrics["wave_corr_aligned"] + .5 * metrics["d1_corr_same_lag"]


@dataclass
class VisualWindow:
    subject_id: str
    timestamps: np.ndarray
    reference: np.ndarray
    signals: dict[str, np.ndarray]
    metrics: dict[str, dict]
    roi_names: tuple[str, ...]
    prior_names: tuple[str, ...]
    diagnostics: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    data_source: str = "unverified_input"

    @property
    def start_sec(self):
        return float(self.timestamps[0])


def raw_plot_data(window: VisualWindow) -> dict:
    """Original clock, original sign, zero shift; return copies for audit/tests."""
    return dict(timestamps=window.timestamps.copy(), prediction=signed_zscore(window.signals[MODEL]),
                reference=signed_zscore(window.reference))


def waveform_overlap(window: VisualWindow, method=MODEL) -> dict:
    metrics = window.metrics[method]
    if not np.isfinite(metrics["selected_lag_samples"]):
        return dict(prediction=np.array([]), reference=np.array([]), timestamps=np.array([]),
                    reference_indices=np.array([], dtype=int), lag_samples=None)
    lag = int(metrics["selected_lag_samples"])
    p, t = aligned_overlap(window.signals[method], window.reference, lag)
    _, indices = aligned_overlap(np.arange(len(window.reference)), np.arange(len(window.reference)), lag)
    return dict(prediction=p.copy(), reference=t.copy(), timestamps=window.timestamps[indices].copy(),
                reference_indices=indices, lag_samples=lag)


def common_comparison(window: VisualWindow, methods: list[str], fs: float) -> dict:
    """Intersect reference indices AFTER each full-window evaluation lag.

    Unavailable methods do not shrink others' support. Zero-lag metric calls
    measure fixed, already aligned samples without optimizing another lag.
    Full official metrics remain separate in window.metrics and ranking tables.
    """
    overlaps = {name: waveform_overlap(window, name) for name in methods}
    usable = {name: data for name, data in overlaps.items() if len(data["reference_indices"]) >= 3}
    if not usable:
        return dict(timestamps=np.array([]), reference=np.array([]), methods={}, valid_samples=0)
    lo = max(data["reference_indices"][0] for data in usable.values())
    hi = min(data["reference_indices"][-1] for data in usable.values()) + 1
    reference = window.reference[lo:hi].copy()
    result = {}
    for name, data in usable.items():
        offset = lo - data["reference_indices"][0]
        p = data["prediction"][offset:offset + hi - lo].copy()
        result[name] = dict(prediction=p, metrics=waveform_window_metrics(
            p, reference, fs, max_lag_sec=0, coarse_diagnostics=False))
    return dict(timestamps=window.timestamps[lo:hi].copy(), reference=reference, methods=result,
                valid_samples=len(reference), reference_index_start=int(lo), reference_index_stop=int(hi))


def derivative_plot_data(window: VisualWindow, fs: float) -> dict:
    """Forward differences of the exact model evaluation overlap, at its lag.

    Multiplying differences by fs gives units per second without changing the
    established same-lag derivative correlation. Separate display arrays use
    independent waveform z-scores so arbitrary model/reference amplitude scales
    do not obscure inflections. Both use the very same samples and differences.
    """
    overlap = waveform_overlap(window)
    return dict(timestamps=overlap["timestamps"][1:], prediction=np.diff(overlap["prediction"]) * fs,
                reference=np.diff(overlap["reference"]) * fs, lag_samples=overlap["lag_samples"],
                prediction_z_per_sec=np.diff(signed_zscore(overlap["prediction"])) * fs,
                reference_z_per_sec=np.diff(signed_zscore(overlap["reference"])) * fs,
                correlation=window.metrics[MODEL]["d1_corr_same_lag"])


def reliable_beats(signal, fs: float) -> dict:
    """Conservative screening around the historical pulse_wave_features detector.

    Same 0.7--4 Hz bandpass, independent standardization, distance and
    prominence as signals.py. Extrema are candidate anchors, not validated
    physiological fiducials. Complete trough-to-trough cycles require one
    detected peak, 2+ samples on each side and duration in [1/3.5, 1/0.7] s.
    """
    x = np.asarray(signal, dtype=float)
    empty = dict(anchors=[], peaks=np.array([], dtype=int), reliable=False, ibi_reliable=False,
                 reason="insufficient reliable beats", filtered=np.array([]))
    if (not np.isfinite(fs) or fs <= 0 or x.ndim != 1 or len(x) < 3 * fs
            or not np.isfinite(x).all() or x.std() <= 1e-8):
        return empty
    z = standardize_1d(bandpass(x, fs))
    distance = max(1, int(.30 * fs))
    peaks, _ = find_peaks(z, distance=distance, prominence=max(.05, .2 * np.std(z)))
    troughs, _ = find_peaks(-z, distance=distance)
    anchors, complete = [], 0
    for p in peaks:
        left, right = troughs[troughs < p], troughs[troughs > p]
        if not len(left) or not len(right):
            continue
        l, r = int(left[-1]), int(right[0])
        complete += 1
        if (p - l >= 2 and r - p >= 2 and 1 / 3.5 <= (r - l) / fs <= 1 / .7
                and np.count_nonzero((peaks > l) & (peaks < r)) == 1):
            anchors.append((l, int(p), r))
    ibis = np.diff(peaks) / fs
    hr = estimate_hr_welch(x, fs).hr_bpm
    ibi_ok = bool(len(ibis) >= 3 and np.all((ibis >= 1 / 3.5) & (ibis <= 1 / .7))
                  and np.isfinite(hr) and abs(60 / ibis.mean() - hr) <= .15 * hr)
    reliable = bool(len(anchors) >= 3 and len(anchors) == complete and ibi_ok)
    return dict(anchors=anchors, peaks=peaks, reliable=reliable, ibi_reliable=ibi_ok, filtered=z,
                reason="ok" if reliable else "insufficient reliable beats or ambiguous cycles/IBI")


def beat_plot_data(window: VisualWindow, fs: float) -> dict:
    """Reference-only anchors and identical sample offsets for paired beats.

    Use real elapsed seconds from each reference trough. Right-pad with NaN
    for a pointwise median only where >=3 beats contribute. No interpolation,
    normalized-phase warping, per-beat shift, peak matching or amplitude fit.
    """
    overlap = waveform_overlap(window)
    detection = reliable_beats(overlap["reference"], fs)
    if not detection["reliable"]:
        return dict(reliable=False, reason="insufficient reliable beats", anchors=[], prediction=[], reference=[])
    p, t = signed_zscore(overlap["prediction"]), signed_zscore(overlap["reference"])
    anchors = detection["anchors"]
    pb, tb = [p[l:r + 1].copy() for l, _, r in anchors], [t[l:r + 1].copy() for l, _, r in anchors]

    def summarize(beats):
        array = np.full((len(beats), max(map(len, beats))), np.nan)
        for i, beat in enumerate(beats):
            array[i, :len(beat)] = beat
        count = np.isfinite(array).sum(axis=0)
        # Every column has at least one real sample; mask unsupported tails.
        quantiles = np.nanpercentile(array, [25, 50, 75], axis=0)
        quantiles[:, count < 3] = np.nan
        return quantiles, count

    pq, count = summarize(pb)
    tq, _ = summarize(tb)
    return dict(reliable=True, reason="ok", anchors=anchors, prediction=pb, reference=tb,
                prediction_quantiles=pq, reference_quantiles=tq, contributors=count,
                seconds=np.arange(pq.shape[1]) / fs, lag_samples=overlap["lag_samples"])


def normalized_welch(signal, fs: float) -> dict:
    """Use estimate_hr_welch's exact segment/zero-padding conventions.

    Display density has unit integral over 0--Nyquist; HR peak is obtained by
    the existing estimator (0.7--3.5 Hz), not a new harmonic selection rule.
    """
    x = np.asarray(signal, dtype=float)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("PSD sample rate must be positive and finite")
    if x.ndim != 1 or len(x) < 3 * fs or not np.isfinite(x).all() or x.std() <= 1e-8:
        return dict(frequency_hz=np.array([]), density=np.array([]), hr_peak_hz=float("nan"))
    nperseg = min(len(x), max(32, int(round(fs * 30))))
    nfft = int(2 ** np.ceil(np.log2(max(nperseg, 4 * nperseg))))
    f, psd = welch(x - x.mean(), fs=fs, nperseg=nperseg, nfft=nfft)
    power = trapezoid(psd, f)
    return dict(frequency_hz=f, density=psd / power if power > 0 else np.full_like(psd, np.nan),
                hr_peak_hz=float(estimate_hr_welch(x, fs).hr_hz), nperseg=nperseg, nfft=nfft)


def coarse_feature_diagnostics(signal, fs: float) -> dict:
    """Existing feature values with explicit conservative reliability gates.

    Features are computed independently on full windows, never after fitting
    to the other site. Normalized slope is the legacy filtered z-amplitude/sec.
    No new BP features, anatomical landmarks or second derivatives are added.
    """
    values = dict.fromkeys(FEATURES, float("nan"))
    detection = reliable_beats(signal, fs)
    reasons = dict.fromkeys(FEATURES, detection["reason"])
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or len(x) < 3 * fs or not np.isfinite(x).all() or x.std() <= 1e-8:
        return dict(values=values, reasons=reasons, reliable_beats=0)
    hr = float(estimate_hr_welch(x, fs).hr_bpm)
    if np.isfinite(hr) and hr > 0:
        values["hr_bpm"], reasons["hr_bpm"] = hr, "ok (Welch; independent of beat detector)"
    try:
        legacy = pulse_wave_features(x, fs)
    except (ValueError, IndexError, FloatingPointError):
        return dict(values=values, reasons=reasons, reliable_beats=len(detection["anchors"]))
    for key in FEATURES:
        allowed = detection["ibi_reliable"] if key in ("ibi_mean_sec", "ibi_std_sec", "ibi_cv") else detection["reliable"]
        if key in ("hr_bpm", "ibi_cv") or not allowed:
            continue
        value = float(legacy.get(key, np.nan))
        nonnegative = key == "ibi_std_sec"
        if np.isfinite(value) and (value >= 0 if nonnegative else value > 1e-8):
            values[key], reasons[key] = value, "ok"
    if np.isfinite(values["ibi_std_sec"]) and values["ibi_mean_sec"] > 1e-8:
        values["ibi_cv"] = values["ibi_std_sec"] / values["ibi_mean_sec"]
        reasons["ibi_cv"] = "ok"
    return dict(values=values, reasons=reasons, reliable_beats=len(detection["anchors"]))


def rank_windows(rows: list[dict]) -> list[dict]:
    """Higher exact morphology score first, then earlier start; invalid last."""
    ranked = []
    for sid in sorted({r["subject_id"] for r in rows}):
        group = sorted((dict(r) for r in rows if r["subject_id"] == sid), key=lambda r: (
            -r["morphology_score"] if np.isfinite(r["morphology_score"]) else float("inf"), r["start_sec"]))
        for i, row in enumerate(group):
            row["rank"] = i + 1 if np.isfinite(row["morphology_score"]) else None
        ranked.extend(group)
    return ranked


def select_windows(ranked: list[dict], subject_ids: list[str], selection=None, seed=42,
                   include_random=False, start_sec=None) -> list[dict]:
    """Upper middle rank for even N; per-subject seeded PCG64 random example."""
    if start_sec is not None and (len(subject_ids) != 1 or not np.isfinite(start_sec) or selection):
        raise ValueError("A finite --start-sec requires one --subject and no --selection")
    if selection is not None and selection not in ("best", "median", "worst", "random"):
        raise ValueError("Unknown window selection")
    result = []
    for sid in sorted(subject_ids):
        group = [r for r in ranked if r["subject_id"] == sid]
        if start_sec is not None:
            matches = [r for r in group if abs(r["start_sec"] - start_sec) <= 1e-6]
            if len(matches) != 1:
                raise ValueError(f"{sid}: --start-sec must match a retained validation window; available: "
                                 + ", ".join(f"{r['start_sec']:.6f}" for r in sorted(group, key=lambda r: r['start_sec'])))
            result.append(dict(subject_id=sid, start_sec=matches[0]["start_sec"], selection="specific",
                               selection_notice="User-specified existing validation window"))
            continue
        group = sorted((r for r in group if r["rank"] is not None), key=lambda r: r["rank"])
        if not group:
            raise ValueError(f"{sid}: no finite morphology scores for diagnostic ranking")
        named = dict(best=group[0], median=group[len(group) // 2], worst=group[-1])
        used = {r["start_sec"] for r in named.values()}
        pool = sorted((r for r in group if r["start_sec"] not in used), key=lambda r: r["start_sec"]) or sorted(
            group, key=lambda r: r["start_sec"])
        # A stable subject seed makes requests for one/all subjects agree.
        subject_seed = int.from_bytes(hashlib.sha256(sid.encode()).digest()[:4], "little")
        named["random"] = pool[int(np.random.default_rng(np.random.SeedSequence([seed, subject_seed])).integers(len(pool)))]
        labels = [selection] if selection else ["best", "median", "worst"]
        if include_random and "random" not in labels:
            labels.append("random")
        for label in labels:
            result.append(dict(subject_id=sid, start_sec=named[label]["start_sec"], selection=label,
                               selection_notice=SELECTION_NOTICE if label != "random" else
                               "Seeded random diagnostic; prefers a window distinct from best/median/worst"))
    return result


def nonoverlapping_keys(rows: list[dict]) -> set[tuple[str, float]]:
    """Greedy earliest windows on half-open support, before feature filtering."""
    chosen, ends = set(), {}
    for row in sorted(rows, key=lambda r: (r["subject_id"], r["start_sec"])):
        sid, start = row["subject_id"], row["start_sec"]
        if start >= ends.get(sid, -np.inf) - 1e-9:
            chosen.add((sid, start))
            ends[sid] = row["end_sec_exclusive"]
    return chosen


def descriptive_stats(values) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n_valid=0, median=None, mad=None, cv=None)
    median = float(np.median(x))
    # CV requires repeat observations and a strictly positive, non-near-zero
    # mean relative to native feature units and its observed spread.
    cv_ok = len(x) >= 2 and np.all(x >= 0) and x.mean() > max(1e-6, 1e-6 * x.std())
    return dict(n_valid=len(x), median=median, mad=float(np.median(abs(x - median))),
                cv=float(x.std(ddof=1) / x.mean()) if cv_ok else None)


def feature_tables(windows: list[VisualWindow], primary: set) -> tuple[list, list, list]:
    agreement = []
    for window in sorted(windows, key=lambda w: (w.subject_id, w.start_sec)):
        ref = window.features["reference"]
        for method in window.signals:
            prediction = window.features[method]
            for key in FEATURES:
                p, t = prediction["values"][key], ref["values"][key]
                valid = bool(np.isfinite(p) and np.isfinite(t))
                agreement.append(dict(subject_id=window.subject_id, start_sec=window.start_sec,
                    method=method, feature=key, prediction=p, reference=t,
                    difference=p - t if valid else np.nan, paired_valid=valid,
                    prediction_valid=bool(np.isfinite(p)), reference_valid=bool(np.isfinite(t)),
                    prediction_reason=prediction["reasons"][key], reference_reason=ref["reasons"][key],
                    primary_nonoverlap=(window.subject_id, window.start_sec) in primary))
    summaries, repeatability = [], []
    for sid in sorted({w.subject_id for w in windows}):
        for method in windows[0].signals:
            for key in FEATURES:
                group = [r for r in agreement if r["subject_id"] == sid and r["method"] == method and r["feature"] == key]
                paired = [r for r in group if r["paired_valid"]]
                summaries.append(dict(subject_id=sid, method=method, feature=key, windows=len(group),
                    paired_valid_windows=len(paired), valid_feature_fraction=len(paired) / len(group),
                    prediction_valid_fraction=sum(r["prediction_valid"] for r in group) / len(group),
                    reference_valid_fraction=sum(r["reference_valid"] for r in group) / len(group),
                    difference=descriptive_stats([r["difference"] for r in paired])))
                nonoverlap = [r for r in group if r["primary_nonoverlap"]]
                for source in ("prediction", "reference"):
                    # Reference is included once per subject/feature.
                    if source == "reference" and method != MODEL:
                        continue
                    stats = descriptive_stats([r[source] for r in nonoverlap])
                    repeatability.append(dict(subject_id=sid, method=method if source == "prediction" else "reference",
                        feature=key, eligible_nonoverlap_windows=len(nonoverlap),
                        valid_feature_fraction=stats["n_valid"] / len(nonoverlap) if nonoverlap else 0., **stats))
    return agreement, summaries, repeatability


def infer_validation_windows(extractor, dataset, checkpoint: dict, progress=print) -> list[VisualWindow]:
    """Use the established forward_batch; reference tensors never enter it."""
    cfg, fs = checkpoint["loss_config"], checkpoint["fs"]
    windows = []
    allowed = set(checkpoint["split_manifest"]["validation_ids"])
    if not {s.subject_id for s in dataset.subjects} <= allowed:
        raise ValueError("Refuse a dataset containing non-validation subjects before inference")
    for index in range(len(dataset)):
        item = dataset[index]
        if item["subject_id"] not in allowed:
            raise ValueError("Dataset item outside validation allowlist")
        batch = {key: item[key][None] for key in ("x", "roi_quality", "roi_valid", "prior_valid")}
        with torch.inference_mode():
            out = forward_batch(extractor._model, batch, extractor.device)
        # Prediction is complete before accessing this item's reference values.
        reference = item["y_ppg"].numpy().copy()
        hr = float(item["y_hr"])
        array = lambda value: value[0].detach().cpu().numpy().copy()
        signals = {MODEL: array(out["ppg"])}
        signals.update({name: value.copy() for name, value in classical_window_predictions(
            item["x"].numpy(), item["roi_valid"].numpy(), item["prior_valid"].numpy(),
            extractor.roi_names, extractor.prior_names) if name.endswith(" multi-ROI average")})
        metrics = {name: waveform_window_metrics(signal, reference, fs, cfg["max_lag_sec"],
            cfg["spectral_fmin_hz"], cfg["spectral_fmax_hz"], hr) for name, signal in signals.items()}
        rec_idx, start, end = dataset.samples[index]
        record = dataset.records[rec_idx]
        timestamps = record["timestamps"][start:end].copy()
        if item["subject_id"] != record["subject_id"] or timestamps[0] != float(item["start_sec"]):
            raise ValueError("Dataset window identity/grid mismatch")
        diagnostics = dict(x=item["x"].numpy().copy(), roi_valid=item["roi_valid"].numpy().copy(),
            roi_source_valid_fraction=item["roi_quality"].numpy().copy(), prior_valid=item["prior_valid"].numpy().copy(),
            roi_attention=array(out["roi_attention"]), prior_weights=array(out["prior_weights"]),
            scaled_residuals=array(extractor._model.residual_scale * out["residuals"]),
            raw_contact_ppg=record["ppg"][start:end].copy(), target_sha256=target_sha256(reference, hr),
            reference_hr_bpm=hr)
        features = {name: coarse_feature_diagnostics(signal, fs) for name, signal in signals.items()}
        features["reference"] = coarse_feature_diagnostics(reference, fs)
        windows.append(VisualWindow(item["subject_id"], timestamps, reference, signals, metrics,
                                    extractor.roi_names, extractor.prior_names, diagnostics, features,
                                    data_source="real_UBFC_validation"))
        if progress and (index == 0 or (index + 1) % 30 == 0 or index + 1 == len(dataset)):
            progress(f"Validation windows: {index + 1}/{len(dataset)} ({item['subject_id']})", flush=True)
    if not windows:
        raise ValueError("Validation dataset has no usable windows; inspect cache/reference support and exclusions")
    missing = allowed - {w.subject_id for w in windows}
    if missing:
        raise ValueError(f"Validation subjects have no usable windows: {sorted(missing)}; full report refused")
    return windows


def _number(value, digits=3):
    return f"{value:.{digits}f}" if value is not None and np.isfinite(value) else "NA"


def _style_axis(ax, title, xlabel, ylabel):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)


def render_window(window: VisualWindow, path, fs: float, selection: str, extra_baseline=None) -> dict:
    """Six readable panels in one required per-example PNG; headless backend."""
    methods = [MODEL, CHROM]
    if extra_baseline and extra_baseline not in methods:
        if extra_baseline not in window.signals:
            raise ValueError(f"Unknown checkpoint baseline {extra_baseline!r}")
        methods.append(extra_baseline)
    colors = {"reference": "#222222", MODEL: "#147db3", CHROM: "#dd8526"}
    if extra_baseline:
        colors.setdefault(extra_baseline, "#7f4ca5")
    fig = plt.figure(figsize=(20, 18), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(1.15, 1, 1))
    label = SELECTION_NOTICE if selection in ("best", "median", "worst") else "Validation diagnostic example"
    source = "Frozen Waveform v1" if window.data_source == "real_UBFC_validation" else "SYNTHETIC / UNVERIFIED INPUT (no official inference result)"
    fig.suptitle(f"{source} | {window.subject_id} | {selection} | start {window.start_sec:.6f} s\n"
                 f"{label} | morphology score {_number(diagnostic_score(window.metrics[MODEL]))}", fontsize=16)

    ax = fig.add_subplot(grid[0, 0])
    raw = raw_plot_data(window)
    ax.plot(raw["timestamps"], raw["reference"], color=colors["reference"], label="Finger PPG")
    ax.plot(raw["timestamps"], raw["prediction"], color=colors[MODEL], label="Facial Waveform v1", alpha=.85)
    _style_axis(ax, "1. Unaligned face rPPG vs finger PPG (original timestamps)\n"
                f"No shift; raw signed r={_number(window.metrics[MODEL]['wave_corr'])}; N={len(window.reference)}",
                "Original video timestamp (s)", "Independent z-score (native sign)")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(grid[0, 1])
    common = common_comparison(window, methods, fs)
    ax.plot(common["timestamps"], signed_zscore(common["reference"]), color=colors["reference"], label="Finger PPG")
    lines = ["2. Aligned waveform comparison", ALIGNMENT_NOTICE,
             f"Common reference overlap N={common['valid_samples']}; r / nRMSE: full evaluation -> common plot"]
    for method in methods:
        m = window.metrics[method]
        shared = common["methods"].get(method)
        if shared:
            ax.plot(common["timestamps"], signed_zscore(shared["prediction"]), label=method, color=colors[method], alpha=.85)
        cm = shared["metrics"] if shared else {}
        lines += [f"{method}: lag={_number(m['selected_lag_ms'], 1)} ms; raw r={_number(m['wave_corr'])}; "
                  f"HR MAE={_number(m['hr_mae'], 2)} bpm",
                  f"   r={_number(m['wave_corr_aligned'])} -> {_number(cm.get('wave_corr'))}; "
                  f"nRMSE={_number(m['aligned_nrmse'])} -> {_number(cm.get('aligned_nrmse'))}"]
    _style_axis(ax, "\n".join(lines), "Reference video timestamp after comparison slicing (s)", "Independent z-score (native sign)")
    ax.title.set_fontsize(9)
    ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(grid[1, 0])
    derivative = derivative_plot_data(window, fs)
    ax.plot(derivative["timestamps"], derivative["reference_z_per_sec"], color=colors["reference"], label="d(reference z)/dt")
    ax.plot(derivative["timestamps"], derivative["prediction_z_per_sec"], color=colors[MODEL], label="d(prediction z)/dt", alpha=.85)
    _style_axis(ax, "3. First derivative on exact model evaluation overlap\n"
                f"Same waveform lag={derivative['lag_samples']} samples; d1 r={_number(derivative['correlation'])}; "
                f"N differences={len(derivative['prediction'])}",
                "Reference video timestamp (s); forward difference at right sample", "Independent waveform z-score / s")
    ax.legend(loc="upper right", fontsize=9)

    ax = fig.add_subplot(grid[1, 1])
    beats = beat_plot_data(window, fs)
    _style_axis(ax, "4. Reference-anchored pulse shapes; global model lag only\n"
                "Real beat durations; window z-scores; no per-beat shift, warp or amplitude fit",
                "Elapsed time from candidate reference trough (s)", "Window-standardized amplitude (z)")
    ax.title.set_fontsize(11)
    if not beats["reliable"]:
        ax.text(.5, .5, "insufficient reliable beats", ha="center", va="center", transform=ax.transAxes, fontsize=14)
    else:
        for key, color in (("reference", colors["reference"]), ("prediction", colors[MODEL])):
            for beat in beats[key]:
                ax.plot(np.arange(len(beat)) / fs, beat, color=color, alpha=.13, linewidth=.8)
            q1, median, q3 = beats[key + "_quantiles"]
            ax.plot(beats["seconds"], median, color=color, linewidth=2, label=f"{key}: median, {len(beats[key])} beats")
            ax.fill_between(beats["seconds"], q1, q3, color=color, alpha=.15, label=f"{key}: IQR (>=3 contributors)")
        ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[2, 0])
    for name, signal in [("reference", window.reference)] + [(m, window.signals[m]) for m in methods]:
        spectrum = normalized_welch(signal, fs)
        f, psd, peak = spectrum["frequency_hz"], spectrum["density"], spectrum["hr_peak_hz"]
        ax.plot(f, psd, color=colors[name], label=f"{name}: HR peak {_number(peak, 2)} Hz")
        if np.isfinite(peak) and len(f):
            ax.plot(peak, np.interp(peak, f, psd), marker="o", color=colors[name], markersize=5)
    _style_axis(ax, "5. Normalized Welch PSD on original full windows\n"
                f"fs={fs:g} Hz; unit integral over 0-Nyquist; harmonics need not match across sites",
                "Frequency (Hz)", "Normalized PSD density (1/Hz)")
    ax.title.set_fontsize(11)
    ax.set_xlim(.5, min(8, fs / 2))
    ax.legend(fontsize=8)

    inner = grid[2, 1].subgridspec(2, 2, height_ratios=(1, 1.2))
    d = window.diagnostics
    ax = fig.add_subplot(inner[0, 0])
    y = np.arange(len(window.roi_names))
    ax.barh(y, d["roi_attention"], color=colors[MODEL])
    ax.set_yticks(y, window.roi_names)
    ax.set_xlim(0, 1)
    ax.set_title("6. Descriptive model allocations\nROI attention", fontsize=11)
    ax.set_xlabel("Allocation weight")
    for i, (valid, fraction) in enumerate(zip(d["roi_valid"], d["roi_source_valid_fraction"])):
        ax.text(.99, i, f"valid={bool(valid)}, source={fraction:.2f}", ha="right", va="center", fontsize=8)
    ax = fig.add_subplot(inner[0, 1])
    residuals = np.abs(d["scaled_residuals"])
    ax.barh(y, residuals.mean(axis=-1), color="#88709d")
    ax.set_yticks(y, window.roi_names)
    ax.set_xlabel("Mean absolute scaled residual (native units)", fontsize=8)
    ax.set_title("Scaled residual magnitude\nLabels: maximum absolute", fontsize=10)
    for i, maximum in enumerate(residuals.max(axis=-1)):
        ax.text(0, i, f"max {maximum:.3f}", va="center", fontsize=8)
    ax = fig.add_subplot(inner[1, :])
    weights, valid = d["prior_weights"], d["prior_valid"] & d["roi_valid"][:, None]
    ax.imshow(weights, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(np.arange(len(window.prior_names)), window.prior_names, fontsize=9)
    ax.set_yticks(y, window.roi_names, fontsize=9)
    ax.set_title("Prior weights in checkpoint order (x = invalid)\n"
                 "Descriptive, not causal/physiological; uncalibrated quality head omitted", fontsize=9)
    for r in range(len(window.roi_names)):
        for k in range(len(window.prior_names)):
            ax.text(k, r, f"{weights[r, k]:.3f}" + (" x" if not valid[r, k] else ""),
                    ha="center", va="center", fontsize=9, color="white" if weights[r, k] > .5 else "black")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return dict(common_plot_valid_samples=common["valid_samples"],
                common_plot_reference_index_start=common.get("reference_index_start"),
                common_plot_reference_index_stop=common.get("reference_index_stop"),
                derivative_valid_differences=len(derivative["prediction"]),
                reliable_reference_beats=len(beats["anchors"]), beat_summary_available=beats["reliable"],
                lag_samples_by_method={name: window.metrics[name]["selected_lag_samples"] for name in methods},
                common_plot_metrics={name: data["metrics"] for name, data in common["methods"].items()})


def _finish_overview(fig, path):
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_overviews(windows, ranked, agreement, repeatability, out):
    """All validation windows; stability uses fixed non-overlapping repeats."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted({w.subject_id for w in windows})
    colors = dict(zip(ids, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    for ax, key, label in zip(axes.flat,
            ("wave_corr_aligned", "d1_corr_same_lag", "morphology_score", "hr_mae"),
            ("Aligned signed r", "Same-lag derivative r", "Morphology score", "HR MAE (bpm)")):
        values = [[r[key] for r in ranked if r["subject_id"] == sid and np.isfinite(r[key])] for sid in ids]
        ax.boxplot([v if v else [np.nan] for v in values])
        ax.set_xticks(np.arange(len(ids)) + 1, ids, rotation=30)
        _style_axis(ax, label, "Validation subject", label)
    fig.suptitle("Frozen Waveform v1: validation-window distributions (overlapping windows; descriptive only)")
    _finish_overview(fig, out / "subject_metric_distributions.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for ax, key, label in zip(axes, ("morphology_score", "d1_corr_same_lag"), ("Morphology score", "Same-lag derivative r")):
        all_values = []
        for sid in ids:
            pairs = []
            for w in windows:
                if w.subject_id != sid:
                    continue
                a = diagnostic_score(w.metrics[CHROM]) if key == "morphology_score" else w.metrics[CHROM][key]
                b = diagnostic_score(w.metrics[MODEL]) if key == "morphology_score" else w.metrics[MODEL][key]
                if np.isfinite([a, b]).all():
                    pairs.append((a, b))
            if pairs:
                a, b = np.asarray(pairs).T
                ax.scatter(a, b, s=18, alpha=.6, color=colors[sid], label=sid)
                all_values.extend(pairs)
        if all_values:
            lo, hi = np.min(all_values), np.max(all_values)
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="equal diagnostic metric")
        _style_axis(ax, label, "CHROM_WIN multi-ROI average", "Frozen Waveform v1")
        ax.legend(fontsize=8)
    fig.suptitle("Matched validation windows; independent established evaluation lag for each method")
    _finish_overview(fig, out / "classical_vs_frozen_scatter.png")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), layout="constrained")
    for ax, (key, label) in zip(axes.flat, FEATURES.items()):
        all_values = []
        counts = []
        for method, color in ((MODEL, "#147db3"), (CHROM, "#dd8526")):
            group = [r for r in agreement if r["feature"] == key and r["method"] == method]
            paired = [r for r in group if r["paired_valid"]]
            ax.scatter([r["reference"] for r in paired], [r["prediction"] for r in paired], s=14, alpha=.4,
                       color=color, label=f"{method}: {len(paired)}/{len(group)} paired valid")
            all_values.extend([r[k] for r in paired for k in ("prediction", "reference")])
            counts.append(len(paired))
        if all_values:
            lo, hi = min(all_values), max(all_values)
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=.8)
        _style_axis(ax, label, "Finger PPG", "Facial waveform")
        ax.legend(fontsize=7)
        if not any(counts):
            ax.text(.5, .5, "No paired reliable feature estimates", ha="center", transform=ax.transAxes, fontsize=8)
    fig.suptitle("Cross-site feature agreement diagnostics\ny=x is an anatomical comparison aid, not a clinical target")
    _finish_overview(fig, out / "morphology_feature_scatter.png")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), layout="constrained")
    for ax, (key, label) in zip(axes.flat, FEATURES.items()):
        for offset, method, color in ((-.2, "reference", "#222222"), (0, MODEL, "#147db3"), (.2, CHROM, "#dd8526")):
            group = [next((r for r in repeatability if r["subject_id"] == sid and r["method"] == method
                           and r["feature"] == key), {}) for sid in ids]
            medians = [r.get("median") if r.get("median") is not None else np.nan for r in group]
            mads = [r.get("mad") if r.get("mad") is not None else np.nan for r in group]
            ax.errorbar(np.arange(len(ids)) + offset, medians, yerr=mads, fmt="o", capsize=3,
                        color=color, markersize=4, label=method)
        ax.set_xticks(np.arange(len(ids)), ids, rotation=40, fontsize=8)
        _style_axis(ax, label, "Validation subject", "Median +/- unscaled MAD")
        ax.legend(fontsize=7)
    fig.suptitle("Observed within-subject consistency on non-overlapping windows\n"
                 "Earliest-window greedy selection before feature filtering; physiology can change")
    _finish_overview(fig, out / "stability_summary.png")

    for sid in ids:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), layout="constrained")
        for ax, (key, label) in zip(axes.flat, FEATURES.items()):
            for method, color in ((MODEL, "#147db3"), (CHROM, "#dd8526"), ("reference", "#222222")):
                group = sorted((r for r in agreement if r["subject_id"] == sid and r["feature"] == key
                                and r["method"] == (MODEL if method == "reference" else method)),
                               key=lambda r: r["start_sec"])
                source = "reference" if method == "reference" else "prediction"
                ax.plot([r["start_sec"] for r in group], [r[source] for r in group], color=color,
                        alpha=.45, linewidth=1, label=method)
                primary = [r for r in group if r["primary_nonoverlap"]]
                ax.scatter([r["start_sec"] for r in primary], [r[source] for r in primary], color=color, s=20)
            _style_axis(ax, label, "Window start on original clock (s)", label)
            ax.legend(fontsize=7)
        fig.suptitle(f"{sid}: within-subject feature trajectories\n"
                     "Lines: overlapping diagnostic windows; dots: primary non-overlapping repeats; gaps: invalid features")
        _finish_overview(fig, out / f"{sid}_feature_trajectories.png")


def write_csv(path, rows):
    """Stable column order, exact float repr, empty missing values, strict JSON."""
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            safe = json_safe(row)
            writer.writerow({k: json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)
                             if isinstance(v, (dict, list)) else v for k, v in safe.items()})


def window_metric_rows(windows, fs):
    rows = []
    for w in sorted(windows, key=lambda w: (w.subject_id, w.start_sec)):
        metrics = w.metrics[MODEL]
        n = len(w.reference)
        lag = metrics["selected_lag_samples"]
        rows.append(dict(subject_id=w.subject_id, start_sec=w.start_sec,
            end_sec_exclusive=float(w.timestamps[-1] + 1 / fs), fs=fs,
            **{k: v for k, v in metrics.items() if not isinstance(v, dict)},
            morphology_score=diagnostic_score(metrics),
            raw_sample_count=n, valid_reference_samples=int(np.isfinite(w.reference).sum()),
            valid_prediction_samples=int(np.isfinite(w.signals[MODEL]).sum()),
            aligned_sample_count=n - abs(int(lag)) if np.isfinite(lag) else 0,
            target_sha256=w.diagnostics["target_sha256"],
            baseline_metrics={name: dict(m, morphology_score=diagnostic_score(m),
                aligned_sample_count=n - abs(int(m["selected_lag_samples"])) if np.isfinite(m["selected_lag_samples"]) else 0,
                valid_prediction_samples=int(np.isfinite(w.signals[name]).sum()))
                for name, m in w.metrics.items() if name != MODEL},
            roi_valid=w.diagnostics["roi_valid"], prior_valid=w.diagnostics["prior_valid"],
            roi_attention=w.diagnostics["roi_attention"], prior_weights=w.diagnostics["prior_weights"],
            scaled_residual_mean_abs=np.abs(w.diagnostics["scaled_residuals"]).mean(axis=-1),
            scaled_residual_max_abs=np.abs(w.diagnostics["scaled_residuals"]).max(axis=-1)))
    return rows


def save_window_arrays(window, path):
    """Small selected-window audit artifact; never concatenate independent windows."""
    arrays = {k: v for k, v in window.diagnostics.items() if isinstance(v, np.ndarray)}
    arrays.update(timestamps=window.timestamps, reference_evaluation=window.reference,
                  method_names=np.asarray(list(window.signals)), signals=np.stack(list(window.signals.values())),
                  roi_names=np.asarray(window.roi_names), prior_names=np.asarray(window.prior_names),
                  data_source=np.asarray(window.data_source))
    np.savez_compressed(path, **arrays)


def write_summary_markdown(path, summary):
    m = summary["metric_aggregates"][MODEL]["subject_balanced_metrics"]
    comparison = summary["model_vs_chrom"]
    lines = ["# Frozen Waveform v1 visual validation", "",
        "Source: **real UBFC validation data and the identity-checked official frozen checkpoint**.", "",
        f"Validation subjects: {', '.join(summary['validation_subject_ids'])}.", "",
        f"Ranked {summary['validation_windows']} validation windows; rendered {len(summary['selected_windows'])} examples. "
        f"Primary non-overlapping windows: {summary['nonoverlap_windows_by_subject']}.", "",
        f"{SELECTION_NOTICE}: highest, upper-middle and lowest exact morphology scores. "
        "These examples cannot be used for checkpoint selection or model tuning.", "",
        "## Identity and access", "",
        f"- Checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        f"- Freeze manifest SHA256: `{summary['freeze_manifest_sha256']}`",
        f"- Split manifest file SHA256: `{summary['split_manifest_file_sha256']}`",
        f"- Git commit: `{summary['provenance']['git_commit']}`; dirty: {summary['provenance']['git_dirty']}",
        f"- Created UTC: {summary['provenance']['created_at_utc']}; fs: {summary['fs']} Hz",
        "- Only explicit validation video/reference/cache paths were accessed; see data_access in JSON.",
        "- Locked test was already consumed; no locked-test data, saved evaluation or references were opened by this report.",
        "- No checkpoint, model, training configuration or partition was modified.", "",
        "## Metrics and comparison", "",
        f"Equal-subject means: aligned r={_number(m['wave_corr_aligned'], 6)}, "
        f"same-lag d1 r={_number(m['d1_corr_same_lag'], 6)}, "
        f"morphology={_number(.5 * (m['wave_corr_aligned'] + m['d1_corr_same_lag']), 6)}, "
        f"HR MAE={_number(m['hr_mae'], 6)} bpm.", "",
        f"Matched-window morphology versus CHROM_WIN: model higher in {comparison['model_higher']} windows, "
        f"lower in {comparison['model_lower']}, equal in {comparison['equal']}; "
        f"{comparison['paired_valid']} valid pairs out of {summary['validation_windows']}.", "",
        "Metrics are waveform_window_metrics unchanged; alignment maximizes signed waveform Pearson r with the "
        "checkpoint max lag. Positive lag means prediction lags reference. Ties visit zero, smaller absolute lag, "
        "then negative before positive. Derivatives are forward differences at that same lag; "
        "the derivative plot independently z-scores each waveform on its evaluation overlap before differentiation.", "",
        "Aligned panel methods use their own full-window evaluation lag and a common intersection of reference samples. "
        "Panel metrics explicitly distinguish full evaluation overlaps from the common plot overlap. "
        "Raw r, HR, PSD and feature diagnostics use full original windows. Each signal is scaled independently "
        "for display without polarity flips. Raw amplitudes/inputs are preserved in selected NPZ artifacts.", "",
        f"**{ALIGNMENT_NOTICE}**", "",
        "## Rendered windows", "", "| Subject | Selection | Start (s) | Common N | Reliable reference beats | Figure |",
        "|---|---|---:|---:|---:|---|"]
    for selected in summary["selected_windows"]:
        lines.append(f"| {selected['subject_id']} | {selected['selection']} | {selected['start_sec']:.6f} | "
                     f"{selected['common_plot_valid_samples']} | {selected['reliable_reference_beats']} | "
                     f"[PNG]({selected['figure']}) |")
    lines += ["", "Each PNG contains original-clock signals, aligned signals, same-lag derivatives, "
              "reference-anchored individual/median pulses with IQR, normalized Welch PSD and descriptive allocations.", "",
              "## Cross-site feature agreement diagnostics", "",
              "No anatomical identity or clinical target is assumed. Features use the historical pulse_wave_features "
              "0.7-4 Hz processing, with conservative beat/IBI gates; independent HR uses original-window Welch. "
              "Normalized rise slope is filtered window z-amplitude/s. Missing/ambiguous estimates stay null/empty, never zero. "
              "The y=x scatter line is an anatomical comparison aid only.", "",
              "| Feature | Model paired valid | CHROM_WIN paired valid | Model median signed difference |",
              "|---|---:|---:|---:|"]
    for key, label in FEATURES.items():
        model = summary["feature_agreement_overall"][MODEL][key]
        classical = summary["feature_agreement_overall"][CHROM][key]
        lines.append(f"| {label} | {model['paired_valid_windows']}/{model['windows']} "
                     f"({model['valid_feature_fraction']:.1%}) | {classical['paired_valid_windows']}/{classical['windows']} "
                     f"({classical['valid_feature_fraction']:.1%}) | {_number(model['difference']['median'], 6)} |")
    lines += ["", "Per-subject feature summaries, every valid fraction and all methods are in summary.json and "
              "feature_subject_summary.csv. Across-window differences are descriptive and correlated due to overlap; "
              "they are not clinical errors.", "", "## Observed consistency", "",
              "Primary repeats are earliest non-overlapping half-open windows, selected before examining feature validity. "
              "repeatability.csv reports per-subject/method/feature median, unscaled MAD, valid count/fraction and sample CV "
              "only for non-near-zero positive means with at least two valid observations. Missing means insufficient evidence. "
              "Overview stability panels compare model, CHROM_WIN and finger reference; subject trajectories mark primary "
              "repeats and retain missing gaps. Other classical averages remain in the tables. "
              "No reference-stability or model-quality threshold selects these repeats.", "",
              "## Descriptive review flags", ""]
    for flag in summary["review_flags"]:
        lines.append(f"- {flag}")
    lines += ["", "These flags locate cases for inspection; they do not diagnose anatomical peak recovery. "
              "Individual reference pulses, rises/falls, extra oscillations and spectral structure should be judged "
              "alongside the six panels. High aggregate correlation alone does not validate detailed pulse morphology.", "",
              "## What do these plots establish, and what do they NOT establish?", "",
              "They establish reproducible signed signal comparisons, coarse cross-site feature agreement, "
              "observed within-subject consistency and model allocations on the existing validation split.", ""]
    lines.extend(f"- {limitation}" for limitation in LIMITATIONS)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_visual_validation(*, ubfc_root, checkpoint_path=DEFAULT_CHECKPOINT, split_path=DEFAULT_SPLIT,
                          freeze_path=None, cache_dir="cache_roi_multi_phase1", out_dir="outputs/waveform_visual_v1",
                          subject=None, selection=None, start_sec=None, include_random=False, seed=42,
                          baseline=None, progress=print) -> dict:
    """Produce a complete validation ranking and requested diagnostic examples.

    All validation subjects are scored even for a single requested figure; this
    keeps the report/ranking universe explicit and its subject summaries intact.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    extractor, checkpoint, manifest, config, freeze_path = verify_frozen_identity(checkpoint_path, split_path, freeze_path)
    requested = validation_subject_ids(manifest, subject)
    if start_sec is not None and (subject is None or selection or not np.isfinite(start_sec)):
        raise ValueError("A finite --start-sec requires --subject and cannot be combined with --selection")
    if start_sec is not None and include_random:
        raise ValueError("--include-random cannot be combined with a single --start-sec selection")
    if selection not in (None, "best", "median", "worst", "random"):
        raise ValueError("Invalid selection")
    if baseline and baseline not in extractor.prior_names:
        raise ValueError(f"--baseline must use the checkpoint's actual priors: {extractor.prior_names}")
    if not Path(cache_dir).is_dir():
        raise FileNotFoundError(f"Phase 1 cache directory missing: {cache_dir}; restore the existing extraction cache")
    if Path(out_dir).exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {out_dir}; choose a fresh --out-dir")
    subjects = validation_subject_paths(ubfc_root, manifest)
    access = [require_phase1_cache(s, config, cache_dir) for s in subjects]
    dataset = UBFCMultiROIRPPGDataset(subjects, cache_dir=cache_dir, **config)
    for attr in ("roi_names", "prior_names", "channel_names"):
        if tuple(getattr(dataset, attr)) != tuple(getattr(extractor, attr)):
            raise ValueError(f"Frozen dataset {attr} mismatch")
    windows = infer_validation_windows(extractor, dataset, checkpoint, progress)
    fs = checkpoint["fs"]
    ranked = rank_windows(window_metric_rows(windows, fs))
    primary = nonoverlapping_keys(ranked)
    for row in ranked:
        row["primary_nonoverlap"] = (row["subject_id"], row["start_sec"]) in primary
        row["selection_notice"] = SELECTION_NOTICE
    selected = select_windows(ranked, requested, selection, seed, include_random, start_sec)
    agreement, subject_features, repeatability = feature_tables(windows, primary)
    output = reserve_run(out_dir)
    (output / "figures").mkdir()
    by_key = {(w.subject_id, w.start_sec): w for w in windows}
    for example in selected:
        window = by_key[example["subject_id"], example["start_sec"]]
        stem = f"{window.subject_id}_{example['selection']}"
        filename = f"figures/{stem}.png"
        example.update(render_window(window, output / filename, fs, example["selection"],
                                     f"{baseline} multi-ROI average" if baseline else None))
        example.update(figure=filename, arrays=f"figures/{stem}.npz",
                       morphology_score=diagnostic_score(window.metrics[MODEL]), raw_sample_count=len(window.reference))
        save_window_arrays(window, output / example["arrays"])
        if progress:
            progress(f"Saved {output / filename}", flush=True)
    render_overviews(windows, ranked, agreement, repeatability, output / "overview")
    aggregate = {}
    feature_overall = {}
    for method in windows[0].signals:
        aggregate[method] = aggregate_waveform_metrics([
            dict(w.metrics[method], subject_id=w.subject_id, start_sec=w.start_sec) for w in windows])
        feature_overall[method] = {}
        for feature in FEATURES:
            group = [r for r in agreement if r["method"] == method and r["feature"] == feature]
            paired = [r for r in group if r["paired_valid"]]
            feature_overall[method][feature] = dict(windows=len(group), paired_valid_windows=len(paired),
                valid_feature_fraction=len(paired) / len(group), difference=descriptive_stats([r["difference"] for r in paired]))
    deltas = [diagnostic_score(w.metrics[MODEL]) - diagnostic_score(w.metrics[CHROM]) for w in windows]
    deltas = np.asarray(deltas)
    comparison = dict(paired_valid=int(np.isfinite(deltas).sum()), model_higher=int((deltas > 0).sum()),
                      model_lower=int((deltas < 0).sum()), equal=int((deltas == 0).sum()))
    flags = [f"{r['subject_id']} start={r['start_sec']:.6f} s: "
             f"morphology={_number(r['morphology_score'])}, aligned r={_number(r['wave_corr_aligned'])}, "
             f"same-lag d1 r={_number(r['d1_corr_same_lag'])}, raw r={_number(r['wave_corr'])}."
             for sid in sorted(manifest["validation_ids"])
             for r in sorted((r for r in ranked if r["subject_id"] == sid and r["rank"] is not None),
                             key=lambda r: r["rank"])[-1:]]
    boundary = sum(abs(w.metrics[MODEL]["selected_lag_samples"]) == round(checkpoint["loss_config"]["max_lag_sec"] * fs)
                   for w in windows)
    flags += [f"Model alignment reached the configured lag boundary in {boundary}/{len(windows)} windows; "
              "lag ambiguity is not a delay measurement.",
              f"Model morphology was lower than CHROM_WIN in {comparison['model_lower']}/{comparison['paired_valid']} paired windows."]
    # Reference hashes document only the allowlisted data read for this report.
    for item in access:
        item["reference_sha256"] = file_sha256(item["reference_path"])
    summary = dict(report_version="waveform_v1_visual_validation_part4_5a_v1", data_source="real_UBFC_validation",
        waveform_v1_status="frozen_v1", selected_epoch=checkpoint["selected_epoch"],
        checkpoint_path=str(Path(checkpoint_path).resolve()), checkpoint_sha256=file_sha256(checkpoint_path),
        freeze_manifest_path=str(freeze_path.resolve()), freeze_manifest_sha256=file_sha256(freeze_path),
        split_manifest_path=str(Path(split_path).resolve()), split_manifest_file_sha256=file_sha256(split_path),
        split_manifest_canonical_sha256=split_manifest_hash(manifest), provenance=provenance(),
        fs=fs, window_length_samples=checkpoint["window_length_samples"], stride_sec=checkpoint["stride_sec"],
        roi_ordering=extractor.roi_names, prior_ordering=extractor.prior_names, channel_ordering=extractor.channel_names,
        alignment=dict(max_lag_sec=checkpoint["loss_config"]["max_lag_sec"],
            selection="waveform_window_metrics: maximize signed waveform Pearson r; positive lag = prediction lags reference",
            ties="zero, smaller absolute lag, negative before positive", derivative="forward np.diff at waveform-selected lag; times fs for /s",
            comparison="common reference-index intersection after each method's full evaluation lag; no new lag optimization",
            target="dataset contact-time duplicate averaging/interpolation on exact retained video grid; independent standardize_1d",
            raw="original timestamps; independent signed z-score; zero shift", notice=ALIGNMENT_NOTICE),
        seed=seed, random_generator="NumPy PCG64; SeedSequence(seed, SHA256(subject_id) first 4 bytes little-endian)",
        selection_notice=SELECTION_NOTICE, ranking_rule="descending 0.5*wave_corr_aligned+0.5*d1_corr_same_lag, then start_sec; invalid last",
        median_rule="upper middle ranked valid window (zero-based n//2)",
        validation_subject_ids=sorted(manifest["validation_ids"]), requested_figure_subject_ids=requested,
        validation_windows=len(windows), selected_windows=selected, metric_aggregates=aggregate,
        model_vs_chrom=comparison, feature_agreement_overall=feature_overall,
        feature_agreement_per_subject=subject_features, repeatability=repeatability,
        repeatability_rule="earliest greedy non-overlapping half-open windows selected before feature validity; median and unscaled MAD",
        feature_reliability_rule="historical peak/trough detector; >=3 unambiguous complete beats, 2 samples per slope, "
            "0.7-3.5 Hz cycle/IBI bounds, >=3 IBI intervals, peak-rate vs Welch within 15%; engineering screening only",
        nonoverlap_windows_by_subject={sid: sum(s == sid for s, _ in primary) for sid in sorted(manifest["validation_ids"])},
        exclusions=dataset.exclusions, reference_diagnostics=dataset.reference_diagnostics,
        data_access=access, locked_test_evaluated=False, locked_test_data_opened=False,
        model_modified=False, review_flags=flags, limitations=LIMITATIONS)
    if summary["checkpoint_sha256"] != OFFICIAL_MODEL_SHA256:
        raise ValueError("Official checkpoint changed during visualization; refusing to issue a report")
    write_csv(output / "window_metrics.csv", sorted(ranked, key=lambda r: (r["subject_id"], r["start_sec"])))
    write_json(output / "window_rankings.json", ranked)
    write_csv(output / "feature_agreement.csv", agreement)
    write_csv(output / "feature_subject_summary.csv", subject_features)
    write_csv(output / "repeatability.csv", repeatability)
    write_json(output / "summary.json", summary)
    write_summary_markdown(output / "summary.md", summary)
    return json_safe(summary)
