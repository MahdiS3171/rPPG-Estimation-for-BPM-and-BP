"""Waveform v2 *synthetic-only* timing-transfer contract.

This does not establish physiological face/hand pulse transit time or BP accuracy.
It measures whether a processing stage adds a differential phase shift when
reference/known-shift synthetic waveforms are provided on one verified grid.

Do not use on real video until the camera timing and anatomical references have
been independently audited. No alignment is applied to fix the candidate outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

import numpy as np

CONTRACT_VERSION = "waveform_v2_synthetic_timing_transfer_v0.1"


@dataclass(frozen=True)
class LagEstimate:
    status: str
    reason: str
    delay_ms: float | None
    peak_correlation: float | None
    first_half_delay_ms: float | None
    second_half_delay_ms: float | None
    half_disagreement_ms: float | None
    source_sample_spacing_ms: float | None
    uncertainty_floor_ms: float | None

    def json(self) -> dict[str, Any]:
        return asdict(self)


def _refuse(reason: str, sample_spacing_ms: float | None = None,
            peak: float | None = None, halves: tuple[float | None, float | None] = (None, None)) -> LagEstimate:
    diff = None if any(x is None for x in halves) else abs(halves[0] - halves[1])
    return LagEstimate("abstain", reason, None, peak, halves[0], halves[1], diff,
                       sample_spacing_ms, sample_spacing_ms)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a - np.mean(a)
    b = b - np.mean(b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 1e-12 else float("nan")


def _best_grid_lag(face: np.ndarray, hand: np.ndarray, radius: int, dt_ms: float) -> tuple[float, float, bool, bool]:
    """Signed: positive means hand follows face; no circular wrap or signal inversion."""
    n = len(face)
    lags = np.arange(-radius, radius + 1)
    scores = []
    for lag in lags:
        if lag < 0:
            a, b = face[-lag:], hand[:lag]
        elif lag > 0:
            a, b = face[:-lag], hand[lag:]
        else:
            a, b = face, hand
        scores.append(_correlation(a, b))
    values = np.asarray(scores)
    if not np.isfinite(values).all():
        return float("nan"), float("nan"), True, True
    best = int(np.argmax(values))
    boundary = best in (0, len(values)-1)
    polarity_problem = -float(np.min(values)) > float(values[best]) + 0.08
    frac = 0.0
    if not boundary:
        left, mid, right = values[best-1:best+2]
        denom = left - 2*mid + right
        if denom < -1e-10:
            frac = float(np.clip(.5*(left-right)/denom, -.5, .5))
    return float((lags[best]+frac)*dt_ms), float(values[best]), boundary, polarity_problem


def estimate_signed_lag(timestamps_s, face, hand, *, max_delay_ms: float = 180.,
                        minimum_duration_s: float = 6., min_peak_corr: float = .72,
                        max_half_disagreement_frames: float = 1.5) -> LagEstimate:
    """Conservative engineering estimate, not a physiological measurement.

    Requires one near-regular, fully finite timestamp grid. Any gaps, incompatible
    times, insufficient SNR proxy, search boundaries or half-instability abstain.
    Even successful subframe values keep >= one-source-sample uncertainty floor.
    """
    t, x, y = (np.asarray(v, dtype=float) for v in (timestamps_s, face, hand))
    if any(z.ndim != 1 for z in (t, x, y)) or not (len(t) == len(x) == len(y)) or len(t) < 12:
        return _refuse("shape_or_minimum_support")
    if not all(np.isfinite(v).all() for v in (t, x, y)):
        return _refuse("nonfinite_data_or_missing_frame")
    dt = np.diff(t)
    if not np.all(dt > 0):
        return _refuse("timestamp_nonmonotonic")
    cadence = float(np.median(dt))
    interval_ms = 1e3*cadence
    if (not np.isfinite(cadence) or cadence <= 0 or
            not np.all(np.abs(dt - cadence) <= max(1e-8, cadence*.02))):
        return _refuse("irregular_or_gapped_clock", interval_ms)
    if t[-1] - t[0] < minimum_duration_s:
        return _refuse("too_short", interval_ms)
    if not all(math.isfinite(v) and v > 0 for v in (max_delay_ms, minimum_duration_s)):
        raise ValueError("max_delay_ms and minimum_duration_s must be positive and finite")
    if not math.isfinite(min_peak_corr) or not -1 <= min_peak_corr <= 1:
        raise ValueError("min_peak_corr must be finite and within [-1, 1]")
    if not math.isfinite(max_half_disagreement_frames) or max_half_disagreement_frames <= 0:
        raise ValueError("max_half_disagreement_frames must be positive and finite")
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return _refuse("flat_signal", interval_ms)
    radius = int(math.ceil(max_delay_ms / interval_ms))
    if radius < 1 or 2*radius+4 >= len(t)//2:
        return _refuse("unsupported_search_span", interval_ms)
    lag_ms, peak, boundary, polarity = _best_grid_lag(x, y, radius, interval_ms)
    if polarity:
        return _refuse("polarity_ambiguous", interval_ms, peak)
    if boundary:
        return _refuse("search_boundary", interval_ms, peak)
    if peak < min_peak_corr:
        return _refuse("low_correlation", interval_ms, peak)
    middle = len(t)//2
    h1, q1, b1, p1 = _best_grid_lag(x[:middle], y[:middle], radius, interval_ms)
    h2, q2, b2, p2 = _best_grid_lag(x[middle:], y[middle:], radius, interval_ms)
    halves = (h1 if not b1 and not p1 else None, h2 if not b2 and not p2 else None)
    if any(z is None for z in halves):
        return _refuse("half_window_unstable", interval_ms, peak, halves)
    if min(q1, q2) < min_peak_corr:
        return _refuse("half_window_low_correlation", interval_ms, peak, halves)
    difference = abs(h1-h2)
    if difference > max_half_disagreement_frames*interval_ms:
        return _refuse("half_window_inconsistent_delay", interval_ms, peak, halves)
    return LagEstimate("diagnostic_only", "synthetic_timing_proxy_not_physiological",
                       lag_ms, peak, h1, h2, difference, interval_ms, interval_ms)


def _corr(a, b):
    if a is None or b is None:
        return None
    a,b=np.asarray(a,float),np.asarray(b,float)
    return _correlation(a,b) if a.ndim==b.ndim==1 and a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() else None


def audit_synthetic_timing_transfer(timestamps_s, *, source_face, source_hand,
                                    reconstructed_face, reconstructed_hand,
                                    known_source_delay_ms: float,
                                    max_inter_site_delay_ms: float = 180.,
                                    max_absolute_shift_ms: float = 350.,
                                    allowed_error_frames: float = .5) -> dict[str, Any]:
    """Measure differential-shift preservation AND camera-time anchoring.

    The known delay is an injected ground-truth property of SYNTHETIC signals.
    A passing differential delay is not enough: identical latency imposed on
    both reconstructed streams cancels in the difference but corrupts camera
    anchoring. No reference-derived correction is ever applied to outputs.
    """
    if not math.isfinite(known_source_delay_ms):
        raise ValueError("An independently injected finite synthetic delay is required")
    if not math.isfinite(allowed_error_frames) or allowed_error_frames <= 0:
        raise ValueError("allowed_error_frames must be finite and positive")
    delays = {
        "source_inter_site": estimate_signed_lag(timestamps_s, source_face, source_hand,
                                                  max_delay_ms=max_inter_site_delay_ms),
        "reconstructed_inter_site": estimate_signed_lag(timestamps_s, reconstructed_face, reconstructed_hand,
                                                         max_delay_ms=max_inter_site_delay_ms),
        "face_output_vs_source": estimate_signed_lag(timestamps_s, source_face, reconstructed_face,
                                                      max_delay_ms=max_absolute_shift_ms),
        "hand_output_vs_source": estimate_signed_lag(timestamps_s, source_hand, reconstructed_hand,
                                                      max_delay_ms=max_absolute_shift_ms),
    }
    good = all(v.status == "diagnostic_only" for v in delays.values())
    interval = next((v.source_sample_spacing_ms for v in delays.values() if v.source_sample_spacing_ms is not None), None)
    tolerance = (interval * allowed_error_frames) if interval is not None else None
    report: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION, "scope": "synthetic_only", "known_source_delay_ms": known_source_delay_ms,
        "allowed_error_frames": allowed_error_frames, "sample_spacing_ms": interval,
        "subframe_values_are_not_verified_sensor_precision": True,
        "estimators": {k: v.json() for k,v in delays.items()},
        "zero_lag_correlation_face": _corr(source_face, reconstructed_face),
        "zero_lag_correlation_hand": _corr(source_hand, reconstructed_hand),
        "differential_delay_error_ms": None, "source_delay_error_ms": None,
        "face_anchor_shift_ms": None, "hand_anchor_shift_ms": None,
        "outcome": "ABSTAIN_UNVERIFIED", "reason": "timing_estimator_abstained",
    }
    if not good or tolerance is None:
        return report
    s = delays["source_inter_site"].delay_ms
    p = delays["reconstructed_inter_site"].delay_ms
    face_shift = delays["face_output_vs_source"].delay_ms
    hand_shift = delays["hand_output_vs_source"].delay_ms
    report.update(source_delay_error_ms=s-known_source_delay_ms,
                  differential_delay_error_ms=p-known_source_delay_ms,
                  face_anchor_shift_ms=face_shift, hand_anchor_shift_ms=hand_shift)
    if abs(s-known_source_delay_ms) > tolerance:
        report.update(outcome="FAIL", reason="source_delay_not_recoverable")
    elif abs(p-known_source_delay_ms) > tolerance:
        report.update(outcome="FAIL", reason="differential_delay_distortion")
    elif max(abs(face_shift),abs(hand_shift)) > tolerance:
        report.update(outcome="FAIL", reason="camera_time_anchor_shift")
    else:
        report.update(outcome="PASS_SYNTHETIC_ONLY", reason="within_half_source_frame_synthetic_threshold")
    return report


def synthetic_pulse(timestamps_s, *, delay_ms=0., hr_bpm=73., morphology=0.0) -> np.ndarray:
    """Pure, deterministic harmonic waveform with moderate beat-rate variation."""
    t=np.asarray(timestamps_s,float)-delay_ms/1000.
    phase=2*np.pi*(hr_bpm/60.*t + .035*np.sin(2*np.pi*.13*t))
    return (np.sin(phase)+.31*np.sin(2*phase + .4 + morphology)
            +.09*np.sin(3*phase + .15 + .5*morphology))


def demonstration_cases(fs: float=60.) -> list[dict[str, Any]]:
    """Educational synthetic scenarios only; no video, models or external data."""
    if not math.isfinite(fs) or fs < 24:
        raise ValueError("Demo requires a finite synthetic source sampling rate >=24 Hz")
    t=np.arange(int(16*fs),dtype=float)/fs
    known=45.0
    f=synthetic_pulse(t)
    h=synthetic_pulse(t,delay_ms=known)
    scenarios={
        "identity":(f,h),
        "common_latency_100ms":(synthetic_pulse(t,delay_ms=100.),synthetic_pulse(t,delay_ms=known+100.)),
        "differential_hand_latency_85ms":(f,synthetic_pulse(t,delay_ms=known+85.)),
        "polarity_inversion_hand":(f,-h),
        "unrelated_hand_noise":(f,np.random.default_rng(20261103).normal(size=len(t))),
        "hand_morphology_change":(f,synthetic_pulse(t,delay_ms=known,morphology=1.7)),
    }
    output=[]
    for name,(fp,hp) in scenarios.items():
        r=audit_synthetic_timing_transfer(t,source_face=f,source_hand=h,
                    reconstructed_face=fp,reconstructed_hand=hp,known_source_delay_ms=known)
        r["case"] = name
        output.append(r)
    return output
