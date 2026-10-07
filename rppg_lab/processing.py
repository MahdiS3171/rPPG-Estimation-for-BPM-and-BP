"""Timing-aware processing. No sorting, extrapolation or hidden gap filling."""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence
import numpy as np
from scipy.signal import butter, sosfiltfilt
from .types import RGBTrace, PhysiologicalSignal, RegionResult, RegionObservation, validate_timestamps
from .config import PipelineConfig
from .classical import METHOD_FUNCS, DEFAULT_PRIORS


def valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, np.asarray(valid, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def shared_grid(timestamps: np.ndarray, sample_rate: float) -> np.ndarray:
    t = validate_timestamps(timestamps)
    if not np.isfinite(sample_rate) or sample_rate <= 0 or not len(t):
        raise ValueError("Need timestamps and a positive finite sample rate")
    count = int(np.floor((t[-1] - t[0]) * sample_rate + 1e-8)) + 1
    return t[0] + np.arange(count, dtype=np.float64) / sample_rate


def resample_trace(trace: RGBTrace, grid: np.ndarray, max_gap_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation within bounded gaps only, with an imputation mask.

    This primitive does not antialias. The video pipeline rejects material
    downsampling; it also records the original acquisition interval separately.
    """
    grid = validate_timestamps(grid)
    if not np.isfinite(max_gap_sec) or max_gap_sec <= 0:
        raise ValueError("max_gap_sec must be positive")
    out = np.full((len(grid), 3), np.nan)
    imputed = np.zeros(len(grid), dtype=bool)
    good = trace.validity_mask & np.isfinite(trace.values).all(axis=1)
    t, x = trace.timestamps[good], trace.values[good]
    if not len(t):
        return out, imputed
    right = np.searchsorted(t, grid, side="left")
    nearest = np.clip(right, 0, len(t) - 1)
    exact = np.isclose(t[nearest], grid, atol=1e-8, rtol=0)
    out[exact] = x[nearest[exact]]
    between = (~exact) & (right > 0) & (right < len(t))
    idx = np.flatnonzero(between)
    l, r = right[idx] - 1, right[idx]
    permitted = (t[r] - t[l]) <= max_gap_sec + 1e-8
    idx, l, r = idx[permitted], l[permitted], r[permitted]
    a = (grid[idx] - t[l]) / (t[r] - t[l])
    out[idx] = x[l] * (1 - a[:, None]) + x[r] * a[:, None]
    imputed[idx] = True
    return out, imputed


def zero_phase_bandpass(values: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    """Length-preserving offline SOS Butterworth. Invalid/short input raises."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or not 0 < low < high < fs / 2 or order < 1:
        raise ValueError("Invalid filter input, order or frequency bounds")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x - x.mean())


def extract_signal(rgb: np.ndarray, grid: np.ndarray, region: str, config: PipelineConfig,
                   allowed: np.ndarray | None = None) -> PhysiologicalSignal:
    """Adapt unchanged classical formulas, filter each valid segment explicitly.

    No filtering across missing spans. Edge samples are excluded, not shifted.
    PCA/ICA polarity and adaptive CHROM/POS phase are unvalidated for timing.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    grid = validate_timestamps(grid)
    if rgb.shape != (len(grid), 3):
        raise ValueError("RGB/grid length mismatch")
    valid = np.isfinite(rgb).all(axis=1)
    if allowed is not None:
        if np.asarray(allowed).shape != valid.shape:
            raise ValueError("allowed mask shape mismatch")
        valid &= allowed
    result = np.full(len(grid), np.nan)
    minimum = max(4, int(np.ceil(config.min_segment_sec * config.sample_rate)))
    edge = int(np.ceil(config.edge_guard_sec * config.sample_rate))
    segments, failures = [], []
    for start, end in valid_runs(valid):
        if end - start < minimum or end - start <= 2 * edge:
            failures.append({"start_index": start, "end_index": end, "reason": "insufficient_duration"})
            continue
        try:
            raw = METHOD_FUNCS[config.method](rgb[start:end], config.sample_rate, do_filter=False)
            if raw.shape != (end - start,) or not np.isfinite(raw).all():
                raise ValueError("Invalid classical algorithm output")
            filtered = zero_phase_bandpass(raw, config.sample_rate, config.filter_low_hz,
                                          config.filter_high_hz, config.filter_order)
        except (ValueError, np.linalg.LinAlgError) as exc:
            failures.append({"start_index": start, "end_index": end, "reason": "algorithm_or_filter_failure", "detail": str(exc)})
            continue
        a, b = start + edge, end - edge
        result[a:b] = filtered[edge:len(filtered) - edge] if edge else filtered
        segments.append({"start_index": start, "end_index": end, "retained_start_index": a, "retained_end_index": b})
    return PhysiologicalSignal(result, grid, config.sample_rate, region, config.method, preprocessing={
        "filter": "Butterworth SOS forward/backward", "filter_low_hz": config.filter_low_hz,
        "filter_high_hz": config.filter_high_hz, "filter_order": config.filter_order,
        "edge_guard_sec": config.edge_guard_sec, "segments": segments, "failures": failures,
        "polarity": "algorithm_native_unvalidated", "algorithm_internal_filter": False,
        "interpolation": "linear_bounded_gap_no_extrapolation", "max_gap_sec": config.max_gap_sec,
    })


def process_rgb_trace(trace: RGBTrace, grid: np.ndarray, config: PipelineConfig,
                      observations: list[RegionObservation] | None = None
                      ) -> tuple[RegionResult, np.ndarray, np.ndarray]:
    """Apply identical bounded interpolation, segment filtering and quality to any ROI.

    The caller constructs the shared grid once, independently of ROI validity.
    Returns the result, uniform RGB, and interpolation flags on that grid.
    """
    from .quality import region_quality
    uniform, imputed = resample_trace(trace, grid, config.max_gap_sec)
    signal = extract_signal(uniform, grid, trace.roi_name, config)
    quality = region_quality(trace, signal, imputed, config.min_valid_fraction,
                             config.min_snr_db, config.hr_min_hz, config.hr_max_hz)
    signal.quality = quality
    return RegionResult(trace, signal, quality, observations or []), uniform, imputed


def build_classical_priors(trace: RGBTrace, grid: np.ndarray, config: PipelineConfig,
                           methods: Sequence[str] = tuple(DEFAULT_PRIORS)) -> dict[str, PhysiologicalSignal]:
    """Build per-ROI priors using the existing classical methods and gap rules.

    Pass result.shared.timestamps: this helper never invents a per-ROI grid.
    Priors retain NaNs, segment failures and filter edge guards. They are
    algorithm-native waveforms, with no morphology or timing validation implied.
    """
    if isinstance(methods, str) or not set(methods) <= METHOD_FUNCS.keys():
        raise ValueError("methods must contain known classical method names")
    uniform, _ = resample_trace(trace, grid, config.max_gap_sec)
    return {name: extract_signal(uniform, grid, trace.roi_name, replace(config, method=name))
            for name in methods}
