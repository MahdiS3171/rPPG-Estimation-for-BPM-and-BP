"""Window and participant-balanced waveform evaluation, independent of scripts.

Undefined metrics are NaN in memory (JSON artifacts encode them as null).
Nonfinite samples are never removed to compress time or invent adjacent edges.
"""
from __future__ import annotations

from dataclasses import asdict
import numpy as np
import torch

from .losses import waveform_spectral_band, waveform_spectral_distance
from .metrics import regression_metrics
from .signals import estimate_hr_welch, pulse_wave_features
from .waveform_alignment import aligned_overlap, lag_candidates

COARSE_FEATURES = ("width50_mean_sec", "upstroke_mean_sec", "fall_time_mean_sec",
                   "area_mean", "ibi_mean_sec", "ibi_std_sec")
WINDOW_METRICS = ("wave_corr", "wave_corr_aligned", "selected_lag_samples", "selected_lag_ms",
                  "d1_corr_same_lag", "aligned_nrmse", "spectral_distance", "spectral_similarity",
                  "hr_mae", "hr_error_bpm", *(f"{key}_abs_error" for key in COARSE_FEATURES))


def finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _corr(pred, target) -> float:
    if len(pred) < 2 or not np.isfinite(pred).all() or not np.isfinite(target).all():
        return float("nan")
    p, t = pred - pred.mean(), target - target.mean()
    denominator = np.linalg.norm(p) * np.linalg.norm(t)
    return float(np.clip(np.dot(p, t) / denominator, -1, 1)) if denominator > 0 else float("nan")


def _usable(x) -> bool:
    return len(x) >= 3 and np.isfinite(x).all() and np.std(x) > 0


def waveform_window_metrics(pred, target, fs: float, max_lag_sec: float = 0.5,
                            spectral_fmin_hz: float = 0.7, spectral_fmax_hz: float = 8.0,
                            reference_hr_bpm: float | None = None, coarse_diagnostics: bool = True) -> dict:
    """Lag maximizes signed waveform correlation; derivative uses that same lag.

    Training selects a joint waveform/derivative lag; evaluation deliberately
    selects waveform alone. Positive lag means prediction lags reference.
    Spectra/HR/coarse features use the original full window, not cropped edges.
    """
    p, t = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    if p.ndim != 1 or p.shape != t.shape:
        raise ValueError("Waveform windows must have matching one-dimensional shapes")
    waveform_spectral_band(fs, spectral_fmin_hz, spectral_fmax_hz)
    if not np.isfinite(max_lag_sec) or max_lag_sec < 0:
        raise ValueError("max_lag_sec must be finite and nonnegative")
    result = dict.fromkeys(WINDOW_METRICS, float("nan"))
    result.update(pred_hr_bpm=float("nan"), reference_hr_bpm=float("nan"))
    if p.size == 0:
        return result
    result["wave_corr"] = _corr(p, t)
    best, selected = -np.inf, None
    # Gapped/nonfinite windows stay unavailable; no sample deletion or DTW.
    if _usable(p) and _usable(t):
        for lag in lag_candidates(len(p), int(round(max_lag_sec * fs))):
            pp, tt = aligned_overlap(p, t, lag)
            corr = _corr(pp, tt)
            if np.isfinite(corr) and corr > best:
                best, selected = corr, lag
        if selected is not None:
            pp, tt = aligned_overlap(p, t, selected)
            result.update(wave_corr_aligned=best, selected_lag_samples=selected,
                          selected_lag_ms=1000 * selected / fs,
                          d1_corr_same_lag=_corr(np.diff(pp), np.diff(tt)),
                          aligned_nrmse=float(np.sqrt(np.mean(
                              ((pp - pp.mean()) / pp.std() - (tt - tt.mean()) / tt.std()) ** 2))))
        distance = float(waveform_spectral_distance(torch.tensor(p[None]), torch.tensor(t[None]), fs,
                                                    spectral_fmin_hz, spectral_fmax_hz)[0])
        result.update(spectral_distance=distance, spectral_similarity=1 - distance)
    if _usable(p):
        result["pred_hr_bpm"] = float(estimate_hr_welch(p, fs).hr_bpm)
    reference = (float(reference_hr_bpm) if reference_hr_bpm is not None
                 else float(estimate_hr_welch(t, fs).hr_bpm) if _usable(t) else float("nan"))
    result["reference_hr_bpm"] = reference
    if np.isfinite(result["pred_hr_bpm"]) and np.isfinite(reference):
        error = result["pred_hr_bpm"] - reference
        result.update(hr_error_bpm=error, hr_mae=abs(error))
    if coarse_diagnostics and _usable(p) and _usable(t):
        # Reuse the established coarse pulse extractor with its historical
        # filter conventions; this is diagnostic only, never model selection.
        try:
            fp, ft = pulse_wave_features(p, fs), pulse_wave_features(t, fs)
        except (ValueError, IndexError, FloatingPointError):
            fp, ft = {}, {}
        result["coarse_prediction"] = {k: float(fp.get(k, np.nan)) for k in COARSE_FEATURES}
        result["coarse_reference"] = {k: float(ft.get(k, np.nan)) for k in COARSE_FEATURES}
        for key in COARSE_FEATURES:
            result[f"{key}_abs_error"] = abs(fp.get(key, np.nan) - ft.get(key, np.nan))
    return result


def aggregate_waveform_metrics(rows: list[dict]) -> dict:
    """Finite means within subject, then equally weighted subject means.

    Each metric has its own finite counts. A valid window for the combined
    morphology summary has finite aligned waveform and same-lag derivative
    correlations. Subjects missing a metric are excluded only from that metric.
    """
    if not rows:
        raise ValueError("No evaluation windows")

    def summarize(group):
        counts = {k: int(sum(np.isfinite(row.get(k, np.nan)) for row in group)) for k in WINDOW_METRICS}
        return {**{k: finite_mean([row.get(k, np.nan) for row in group]) for k in WINDOW_METRICS},
                "windows": len(group), "finite_window_counts": counts,
                "valid_windows": int(sum(np.isfinite(row.get("wave_corr_aligned", np.nan))
                                     and np.isfinite(row.get("d1_corr_same_lag", np.nan)) for row in group))}

    groups = {}
    for row in rows:
        sid = row.get("subject_id")
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError("Each evaluation row requires subject_id")
        groups.setdefault(sid, []).append(row)
    subjects = {sid: summarize(group) for sid, group in sorted(groups.items())}
    window = summarize(rows)
    balanced = {k: finite_mean([s[k] for s in subjects.values()]) for k in WINDOW_METRICS}
    balanced["subjects"] = len(subjects)
    balanced["finite_subject_counts"] = {
        k: int(sum(np.isfinite(s[k]) for s in subjects.values())) for k in WINDOW_METRICS}
    return dict(window_metrics=window, subject_balanced_metrics=balanced, per_subject=subjects,
                valid_windows_per_subject={sid: s["valid_windows"] for sid, s in subjects.items()})


def forward_batch(model, batch: dict, device: torch.device) -> dict:
    return model(batch["x"].to(device), roi_quality=batch["roi_quality"].to(device),
                 roi_valid=batch["roi_valid"].to(device), prior_valid=batch["prior_valid"].to(device),
                 return_dict=True)


def evaluate_waveform_model(model, loader, device: torch.device, fs: float,
                            max_lag_sec: float = 0.5, spectral_fmin_hz: float = 0.7,
                            spectral_fmax_hz: float = 8.0, include_windows: bool = False) -> dict:
    """Full metric suite and descriptive gates, conditional on valid ROI support."""
    model.eval()
    rows = []
    attention = np.zeros(model.num_rois)
    weights = np.zeros((model.num_rois, model.num_priors))
    valid_count = np.zeros(model.num_rois)
    residual_sum = np.zeros(model.num_rois)
    residual_samples = np.zeros(model.num_rois)
    residual_max = 0.0
    with torch.no_grad():
        for batch in loader:
            out = forward_batch(model, batch, device)
            pred = out["ppg"].cpu().numpy()
            target = batch["y_ppg"].cpu().numpy()
            valid = batch["roi_valid"].cpu().numpy()
            attention += out["roi_attention"].cpu().numpy().sum(axis=0)
            weights += out["prior_weights"].cpu().numpy().sum(axis=0)
            valid_count += valid.sum(axis=0)
            scaled = (model.residual_scale * out["residuals"]).abs().cpu().numpy()
            residual_max = max(residual_max, float(scaled.max()))
            residual_sum += (scaled * valid[..., None]).sum(axis=(0, 2))
            residual_samples += valid.sum(axis=0) * scaled.shape[-1]
            for i in range(len(pred)):
                row = waveform_window_metrics(pred[i], target[i], fs, max_lag_sec,
                    spectral_fmin_hz, spectral_fmax_hz, float(batch["y_hr"][i]))
                row.update(subject_id=batch["subject_id"][i], start_sec=float(batch["start_sec"][i]))
                rows.append(row)
    result = aggregate_waveform_metrics(rows)
    windows = len(rows)
    result.update(windows=windows,
        # Keep Phase 2 window-level keys for existing analysis callers.
        hr_metrics=asdict(regression_metrics([r["reference_hr_bpm"] for r in rows],
                                            [r["pred_hr_bpm"] for r in rows])),
        wave_corr=result["window_metrics"]["wave_corr"],
        wave_corr_aligned=result["window_metrics"]["wave_corr_aligned"],
        roi_attention_mean={name: float(attention[r] / windows) for r, name in enumerate(model.roi_names)},
        roi_valid_fraction={name: float(valid_count[r] / windows) for r, name in enumerate(model.roi_names)},
        prior_weights_mean={name: {prior: float(weights[r, k] / valid_count[r]) if valid_count[r] else None
                                  for k, prior in enumerate(model.prior_names)}
                            for r, name in enumerate(model.roi_names)},
        residual_contribution_mean_abs={name: float(residual_sum[r] / residual_samples[r])
                                       if residual_samples[r] else float("nan")
                                       for r, name in enumerate(model.roi_names)},
        residual_contribution_max_abs=residual_max)
    if include_windows:
        result["window_results"] = rows
    return result
