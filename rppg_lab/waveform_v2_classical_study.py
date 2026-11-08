"""Fixed Gate 08 classical predictions, label-free descriptors and separate scoring.

No neural execution, subject adaptation, reference-driven selection or sign repair.
The existing V1 input/prior/fusion functions remain the prediction implementation.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import warnings

import numpy as np
from scipy.signal import welch
from scipy.stats import spearmanr

from .classical import DEFAULT_PRIORS, METHOD_FUNCS, bandpass_filter
from .quality import signal_quality
from .signals import estimate_hr_welch, standardize_1d
from .waveform_alignment import aligned_overlap, lag_candidates
from .waveform_baselines import classical_window_predictions
from .waveform_v2_scamps_inputs import prepare_scamps_v1_inputs
from .waveform_v2_scamps_roi import ROI_NAMES, ScampsROITraces
from .waveform_v2_timing_contract import audit_synthetic_timing_transfer, synthetic_pulse

METHODS = tuple(DEFAULT_PRIORS)
CONDITIONS = (*ROI_NAMES, "equal_roi_average")
STARTS = (0, 300)
FS = 30.0
FLAT_EPS = 1e-10
QUALITY_KEYS = ("spectral_concentration", "snr_db", "periodicity",
                "half_hr_difference_bpm", "adjacent_window_hr_difference_bpm",
                "inter_roi_hr_range_bpm", "clipping_fraction_mean", "rgb_step_mean")


def array_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def finite_mean(values):
    x = np.asarray(values, dtype=float)
    return float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else float("nan")


def usable(values):
    x = np.asarray(values, dtype=float)
    return x.ndim == 1 and len(x) >= 90 and np.isfinite(x).all() and np.std(x) > FLAT_EPS


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or not usable(a) or not usable(b):
        return float("nan")
    a, b = a-a.mean(), b-b.mean()
    return float(np.clip(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))


def make_traces(rgb, valid):
    """The legacy target slot is explicitly NOT LOADED for prediction preparation."""
    rgb, valid = np.asarray(rgb), np.asarray(valid)
    if (rgb.shape != (3, 600, 3) or rgb.dtype.kind != "f" or valid.shape != (3, 600)
            or valid.dtype != bool):
        raise ValueError("Require three 600-frame RGB traces and explicit bool validity")
    provenance = dict(coordinate_kind="documented_nominal_synthetic_frame_grid",
                      nominal_rate_hz=FS, capture_timestamps=None, physical_clock_verified=False,
                      segmentation_used=False, subject_adaptation_enabled=False,
                      allow_training=False, reference_values_decoded=False)
    indices = np.arange(600)
    return ScampsROITraces(rgb, valid, indices, indices.copy(), np.full(600, np.nan),
                          np.zeros((3, 600), dtype="i4"), np.zeros((3, 600)),
                          np.where(valid, "", "missing_source"), valid.any(0),
                          indices/FS, provenance)


def predict_recording(rgb, valid):
    """No reference argument: exact V1 priors and existing equal-ROI fusion."""
    batch = prepare_scamps_v1_inputs(make_traces(rgb, valid), STARTS)
    signals = np.full((2, 4, 6, 300), np.nan, dtype=np.float32)
    computational = np.zeros((2, 4, 6), dtype=bool)
    flat_input = np.zeros_like(computational)
    weights = np.zeros((2, 6, 3))
    for window, start in enumerate(STARTS):
        source_flat = []
        for r in range(3):
            values = rgb[r, start:start+300][valid[r, start:start+300]]
            source_flat.append(bool(len(values) and np.isfinite(values).all()
                                    and np.std(values, axis=0).max() <= FLAT_EPS))
        predictions = dict(classical_window_predictions(
            batch["x"][window], batch["roi_valid"][window], batch["prior_valid"][window],
            ROI_NAMES, METHODS))
        for k, method in enumerate(METHODS):
            keep = batch["roi_valid"][window] & batch["prior_valid"][window, :, k]
            if keep.any():
                weights[window, k, keep] = 1.0/keep.sum()
            for r, roi in enumerate(CONDITIONS):
                name = f"{roi}/{method}" if r < 3 else f"{method} multi-ROI average"
                signals[window, r, k] = predictions[name]
                computational[window, r, k] = bool(keep[r]) if r < 3 else bool(keep.any())
                flat_input[window, r, k] = source_flat[r] if r < 3 else bool(
                    keep.any() and np.asarray(source_flat)[keep].all())
    return dict(batch=batch, signals=signals, computational=computational,
                flat_input=flat_input, fusion_weights=weights)


def trace_quality(values):
    """All peaks/consistency are estimated from the prediction alone."""
    keys = ("dominant_hr_bpm", "snr_db", "spectral_peak_to_mean", "periodicity",
            "amplitude_std", "spectral_concentration", "half_hr_difference_bpm")
    out = dict.fromkeys(keys, float("nan"))
    if not usable(values):
        return out
    q = signal_quality(values, FS)
    out.update({k: q[k] for k in ("snr_db", "spectral_peak_to_mean", "periodicity", "amplitude_std")})
    out["dominant_hr_bpm"] = q["hr_bpm"]
    f, p = welch(np.asarray(values, float)-np.mean(values), fs=FS,
                 nperseg=min(len(values), 240))
    band = (f >= .7) & (f <= 3.5)
    if p[band].sum() > 0:
        own_peak = f[band][np.argmax(p[band])]
        out["spectral_concentration"] = float(p[band & (np.abs(f-own_peak) <= .15)].sum()/p[band].sum())
    halves = np.array_split(values, 2)
    if all(usable(v) for v in halves):
        out["half_hr_difference_bpm"] = abs(estimate_hr_welch(halves[0], FS).hr_bpm
                                              - estimate_hr_welch(halves[1], FS).hr_bpm)
    return out


def prediction_rows(filename, split, rgb, valid, clipping, prediction):
    """Create and freeze label-free observations for every intended cell."""
    if np.shape(clipping) != (3, 600):
        raise ValueError("Clipping must retain every ROI/frame")
    rows = []
    for w, start in enumerate(STARTS):
        optical = []
        for r in range(3):
            source = rgb[r, start:start+300]
            pairs = np.isfinite(source[1:]).all(-1) & np.isfinite(source[:-1]).all(-1)
            steps = np.abs(np.diff(source, axis=0)).mean(-1)/255
            optical.append(dict(source_missing_fraction=float(1-valid[r, start:start+300].mean()),
                                imputed_fraction=float(prediction["batch"]["interpolated"][w, r].mean()),
                                clipping_fraction_mean=finite_mean(clipping[r, start:start+300]),
                                rgb_step_mean=finite_mean(steps[pairs])))
        for r, roi in enumerate(CONDITIONS):
            for k, method in enumerate(METHODS):
                signal = prediction["signals"][w, r, k]
                computed = bool(prediction["computational"][w, r, k])
                flat = bool(prediction["flat_input"][w, r, k])
                eligible = computed and not flat and usable(signal)
                reason = ("NO_COMPUTABLE_PRIOR" if not computed else "FLAT_SOURCE_RGB" if flat
                          else "FLAT_OR_NONFINITE_PREDICTION" if not usable(signal) else "")
                quality = trace_quality(signal) if eligible else trace_quality(np.full(300, np.nan))
                if r < 3:
                    quality.update(optical[r])
                else:
                    weights = prediction["fusion_weights"][w, k]
                    for name in optical[0]:
                        quality[name] = (float(sum(weights[i]*optical[i][name] for i in range(3) if weights[i] > 0))
                                         if weights.any() else float("nan"))
                rows.append(dict(filename=filename, split=split, start=start, stop=start+300,
                                 roi=roi, method=method, prediction_sha256=array_sha256(signal),
                                 computational_success=computed, pulse_estimate_eligible=bool(eligible),
                                 prediction_failure_reason=reason, quality=quality,
                                 skin_label="UNKNOWN", physiological_validation="SYNTHETIC_ONLY_EXPLORATORY"))
    lookup = {(v["start"], v["roi"], v["method"]): v for v in rows}
    for row in rows:
        other = lookup[(300-row["start"], row["roi"], row["method"])]
        a, b = row["quality"]["dominant_hr_bpm"], other["quality"]["dominant_hr_bpm"]
        row["quality"]["adjacent_window_hr_difference_bpm"] = abs(a-b) if np.isfinite([a, b]).all() else float("nan")
        hrs = [lookup[(row["start"], roi, row["method"])]["quality"]["dominant_hr_bpm"] for roi in ROI_NAMES]
        row["quality"]["inter_roi_hr_range_bpm"] = float(np.ptp(hrs)) if np.isfinite(hrs).all() else float("nan")
    return rows


def score_waveform(prediction, reference, *, prediction_eligible=True):
    """Reference access ends here: it cannot alter saved predictions or quality."""
    p, raw = np.asarray(prediction, float), np.asarray(reference, float)
    if p.shape != (300,) or raw.shape != (300,):
        raise ValueError("Require identical complete 300-sample windows")
    pred_ok, ref_ok = bool(prediction_eligible and usable(p)), usable(raw)
    out = dict.fromkeys(("pred_hr_bpm", "reference_hr_bpm", "hr_error_bpm", "hr_abs_error_bpm",
                         "hr_squared_error", "zero_lag_r", "lag_aligned_r", "selected_lag_samples",
                         "selected_lag_ms", "aligned_overlap_samples", "matched_band_reference_zero_lag_r"), float("nan"))
    out.update(reference_usable=bool(ref_ok), prediction_usable=pred_ok, lag_at_search_boundary=False)
    if pred_ok:
        out["pred_hr_bpm"] = float(estimate_hr_welch(p, FS).hr_bpm)
    if ref_ok:
        out["reference_hr_bpm"] = float(estimate_hr_welch(raw, FS).hr_bpm)
    if np.isfinite([out["pred_hr_bpm"], out["reference_hr_bpm"]]).all():
        error = out["pred_hr_bpm"]-out["reference_hr_bpm"]
        out.update(hr_error_bpm=error, hr_abs_error_bpm=abs(error), hr_squared_error=error**2)
    if pred_ok and ref_ok:
        target = standardize_1d(raw)
        out["zero_lag_r"] = correlation(p, target)
        best, selected = -np.inf, 0
        for lag in lag_candidates(300, 15):
            pp, tt = aligned_overlap(p, target, lag)
            score = correlation(pp, tt)
            if score > best:
                best, selected = score, lag
        out.update(lag_aligned_r=float(best), selected_lag_samples=selected,
                   selected_lag_ms=1000*selected/FS, aligned_overlap_samples=300-abs(selected),
                   lag_at_search_boundary=abs(selected) == 15,
                   matched_band_reference_zero_lag_r=correlation(p, bandpass_filter(raw, FS)))
    return out


METRIC_KEYS = ("hr_abs_error_bpm", "hr_squared_error", "hr_error_bpm", "zero_lag_r",
               "lag_aligned_r", "selected_lag_ms", "matched_band_reference_zero_lag_r")


def summarize_group(rows):
    counts = {k: sum(np.isfinite(r["metrics"][k]) for r in rows) for k in METRIC_KEYS}
    means = {k: finite_mean([r["metrics"][k] for r in rows]) for k in METRIC_KEYS}
    return dict(intended_cells=len(rows), computational_cells=sum(r["computational_success"] for r in rows),
                eligible_cells=sum(r["pulse_estimate_eligible"] for r in rows),
                finite_cells=counts, **means, hr_rmse_bpm=float(np.sqrt(means["hr_squared_error"])),
                prediction_failures=dict(Counter(r["prediction_failure_reason"] for r in rows if r["prediction_failure_reason"])),
                reference_unusable_cells=sum(not r["metrics"]["reference_usable"] for r in rows),
                lag_boundary_cells=sum(r["metrics"]["lag_at_search_boundary"] for r in rows))


def aggregate(rows):
    """Condition-wise equal recording weights, including metric-specific counts."""
    result = []
    for split in dict.fromkeys(r["split"] for r in rows):
        for roi in CONDITIONS:
            for method in METHODS:
                group = [r for r in rows if (r["split"], r["roi"], r["method"]) == (split, roi, method)]
                recordings = {name: summarize_group([r for r in group if r["filename"] == name])
                              for name in dict.fromkeys(r["filename"] for r in group)}
                means = {k: finite_mean([v[k] for v in recordings.values()]) for k in METRIC_KEYS}
                result.append(dict(split=split, roi=roi, method=method, per_recording=recordings,
                                   pooled=summarize_group(group), recording_balanced=dict(
                                       **means, hr_rmse_bpm=float(np.sqrt(means["hr_squared_error"])),
                                       intended_recordings=len(recordings), finite_recordings={
                                           k: sum(np.isfinite(v[k]) for v in recordings.values()) for k in METRIC_KEYS})))
    return result


def quality_associations(rows):
    """Descriptive Train-only recording means; no indicator fitting or cutoffs."""
    out = []
    for roi in CONDITIONS:
        for method in METHODS:
            group = [r for r in rows if r["split"] == "Train" and r["roi"] == roi and r["method"] == method]
            for key in QUALITY_KEYS:
                for metric in ("hr_abs_error_bpm", "zero_lag_r"):
                    pairs = [(finite_mean([r["quality"][key] for r in group if r["filename"] == name]),
                              finite_mean([r["metrics"][metric] for r in group if r["filename"] == name]))
                             for name in dict.fromkeys(r["filename"] for r in group)]
                    good = np.asarray([p for p in pairs if np.isfinite(p).all()], float).reshape(-1, 2)
                    defined = len(good) >= 3 and all(np.ptp(good[:, j]) > 0 for j in (0, 1))
                    rho = float(spearmanr(good[:, 0], good[:, 1]).statistic) if defined else float("nan")
                    out.append(dict(roi=roi, method=method, indicator=key, metric=metric,
                                    finite_recording_pairs=len(good), spearman_rho=rho,
                                    status="DESCRIPTIVE_ONLY" if defined else "UNDEFINED_TIES_OR_SUPPORT"))
    return out


def control_fixtures():
    t = np.arange(600)/FS
    pulse = np.sin(2*np.pi*1.2*t)
    base = np.array([160., 120., 100.])
    rgb = np.broadcast_to(base+pulse[:, None]*[.8, 1.2, .5], (3, 600, 3)).copy()
    valid = np.ones((3, 600), bool)
    constant = np.broadcast_to(base, rgb.shape).copy()
    drift = constant+2*np.sin(2*np.pi*.15*t)[None, :, None]
    noise = constant+np.random.default_rng(808).normal(0, 2, rgb.shape)
    illumination = constant+2*pulse[None, :, None]
    impulses = rgb.copy(); impulses[:, [90, 210, 390, 510]] = 255
    missing = rgb.copy(); missing_valid = valid.copy()
    missing_valid[0, :300] = False; missing_valid[:, 300:] = False
    missing[~missing_valid] = np.nan
    rng = np.random.default_rng(809)
    permuted = np.stack([r[rng.permutation(600)] for r in rgb])
    for name, values, flags in (
        ("sinusoidal_positive", rgb, valid), ("constant_nonpulsatile", constant, valid),
        ("slow_illumination_drift", drift, valid), ("independent_noise", noise, valid),
        ("coherent_pulse_band_illumination", illumination, valid),
        ("clipped_impulses", impulses, valid), ("missing_windows", missing, missing_valid),
        ("time_permuted", permuted, valid)):
        yield name, values.copy(), flags.copy(), pulse.copy()


def run_controls():
    rows = []
    for name, rgb, valid, reference in control_fixtures():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # constant PCA fixture
            prediction = predict_recording(rgb, valid)
        clipping = np.any((rgb <= 1) | (rgb >= 254), axis=-1).astype(float)
        clipping[~np.isfinite(rgb).all(-1)] = np.nan
        items = prediction_rows(name, "Control", rgb, valid, clipping, prediction)
        for row in items:
            w, r, k = STARTS.index(row["start"]), CONDITIONS.index(row["roi"]), METHODS.index(row["method"])
            row["metrics"] = score_waveform(prediction["signals"][w, r, k], reference[row["start"]:row["stop"]],
                                             prediction_eligible=row["pulse_estimate_eligible"])
        rows.extend(items)
    return rows


def timing_controls():
    t = np.arange(600)/FS
    face, hand = synthetic_pulse(t), synthetic_pulse(t, delay_ms=45)
    rgb = lambda pulse: np.broadcast_to([160., 120., 100.]+pulse[:, None]*[.8, 1.2, .5], (3, 600, 3)).copy()
    valid = np.ones((3, 600), bool)
    fp, hp = predict_recording(rgb(face), valid), predict_recording(rgb(hand), valid)
    results = []
    for k, method in enumerate(METHODS):
        windowed = fp["signals"][:, 0, k].reshape(-1)
        full = standardize_1d(METHOD_FUNCS[method](rgb(face)[0], FS))
        boundary = np.zeros(600, bool)
        boundary[:15] = True; boundary[-15:] = True; boundary[285:315] = True
        delta = np.abs(windowed-full)
        report = audit_synthetic_timing_transfer(
            t, source_face=face, source_hand=hand, reconstructed_face=windowed,
            reconstructed_hand=hp["signals"][:, 0, k].reshape(-1), known_source_delay_ms=45)
        results.append(dict(case="window_local_classical", method=method, **report,
                            full_vs_window_zero_lag_r=correlation(full, windowed),
                            boundary_max_absolute_difference=float(delta[boundary].max()),
                            interior_max_absolute_difference=float(delta[~boundary].max()),
                            difference_units="standardized_output_includes_window_statistics_and_filter_edges"))
    delays = (0, 120, 120)
    fusion_face = np.stack([rgb(synthetic_pulse(t, delay_ms=d))[0] for d in delays])
    fusion_hand = np.stack([rgb(synthetic_pulse(t, delay_ms=d+45))[0] for d in delays])
    f, h = predict_recording(fusion_face, valid), predict_recording(fusion_hand, valid)
    report = audit_synthetic_timing_transfer(
        t, source_face=face, source_hand=hand, reconstructed_face=f["signals"][:, 3, 0].reshape(-1),
        reconstructed_hand=h["signals"][:, 3, 0].reshape(-1), known_source_delay_ms=45)
    results.append(dict(case="existing_equal_roi_fusion_heterogeneous_phases", method="GREEN",
                        roi_delays_ms=delays, weights=[1/3]*3, **report))
    return results
