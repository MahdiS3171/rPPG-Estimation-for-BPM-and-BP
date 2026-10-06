"""Inspectable signal indicators, never medical confidence."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from .processing import valid_runs
from .signals import estimate_hr_welch
from .types import RGBTrace, PhysiologicalSignal


def finite_mean(values) -> float:
    x = np.asarray(values, dtype=float)
    good = np.isfinite(x)
    return float(x[good].mean()) if good.any() else float("nan")


def signal_quality(values: np.ndarray, fs: float, fmin: float = 0.7, fmax: float = 3.5) -> dict:
    x = np.asarray(values, dtype=float)
    out = {"hr_bpm": float("nan"), "snr_db": float("nan"), "spectral_peak_to_mean": float("nan"),
           "periodicity": float("nan"), "amplitude_std": float("nan")}
    if x.ndim != 1 or len(x) < 3 * fs or not np.isfinite(x).all() or np.std(x) < 1e-10:
        return out
    est = estimate_hr_welch(x, fs, fmin=fmin, fmax=fmax)
    f, p = welch(x - x.mean(), fs=fs, nperseg=min(len(x), int(8 * fs)))
    p = p[(f >= fmin) & (f <= fmax)]
    lag = int(round(fs / est.hr_hz)) if np.isfinite(est.hr_hz) and est.hr_hz > 0 else 0
    periodicity = float("nan")
    if 0 < lag < len(x) - 2 and np.std(x[:-lag]) > 0 and np.std(x[lag:]) > 0:
        periodicity = float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
    out.update(hr_bpm=est.hr_bpm, snr_db=est.snr_db,
               spectral_peak_to_mean=float(p.max() / p.mean()) if len(p) and p.mean() > 0 else float("nan"),
               periodicity=periodicity, amplitude_std=float(np.std(x)))
    return out


def region_quality(trace: RGBTrace, signal: PhysiologicalSignal, imputed: np.ndarray,
                   min_valid_fraction: float, min_snr_db: float, fmin: float, fmax: float) -> dict:
    runs = valid_runs(np.isfinite(signal.values))
    longest = max(runs, key=lambda ab: ab[1] - ab[0]) if runs else (0, 0)
    start, end = longest
    out = signal_quality(signal.values[start:end], signal.sample_rate, fmin, fmax)
    out.update(valid_frame_fraction=float(np.mean(trace.validity_mask)),
               missing_frame_fraction=float(np.mean(~trace.validity_mask)),
               signal_valid_fraction=float(np.mean(np.isfinite(signal.values))),
               interpolated_grid_fraction=float(np.mean(imputed)),
               longest_signal_start_sec=float(signal.timestamps[start]) if end > start else None,
               longest_signal_end_sec=float(signal.timestamps[end - 1]) if end > start else None,
               hr_support="longest_contiguous_usable_segment")
    for name, values in trace.quality.items():
        out[name + "_mean"] = finite_mean(values)
    brightness = np.asarray(trace.quality.get("brightness", []), dtype=float)
    out["illumination_std"] = float(np.std(brightness[np.isfinite(brightness)])) if np.isfinite(brightness).any() else float("nan")
    reasons = []
    if not trace.validity_mask.any():
        reasons.append(trace.roi_name + "_not_detected")
    if out["valid_frame_fraction"] < min_valid_fraction:
        reasons.append("insufficient_valid_frames")
    if not np.isfinite(out["hr_bpm"]):
        reasons.append("insufficient_duration_or_flat_signal")
    if not np.isfinite(out["snr_db"]) or out["snr_db"] < min_snr_db:
        reasons.append("low_signal_quality")
    out["usable"] = not reasons
    out["reasons"] = reasons
    return out
