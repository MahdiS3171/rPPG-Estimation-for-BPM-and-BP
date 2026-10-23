"""External, offline timebase adapter; Frozen Waveform V1 remains unchanged.

Native irregular RGB -> bounded cubic regularization at 60 Hz -> centered
anti-alias FIR -> filtered values evaluated back on the *actual* camera PTS.
The unchanged frozen API then does its usual bounded interpolation to 30 Hz.
Keeping that last step preserves its native-source support rule and clock
validation. This is not claimed equivalent to historical 30 Hz acquisition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import firwin, freqz, kaiserord
from scipy.interpolate import CubicSpline

from .extraction_cache import FaceROIExtraction, _validate
from .processing import shared_grid, resample_trace, valid_runs
from .types import RGBTrace, SharedTimebase, validate_timestamps

ADAPTER_VERSION = "native60_cubic_fir_camera_pts_to30_v2"
LINEAR_VERSION = "native60_fir_camera_pts_to30_v1"


@dataclass(frozen=True)
class TimebaseConfig:
    version: str = ADAPTER_VERSION
    native_grid_hz: float = 60.0
    target_hz: float = 30.0
    passband_hz: float = 8.0
    stopband_hz: float = 15.0
    attenuation_db: float = 80.0
    max_interpolation_gap_sec: float = 0.050001
    native_rate_tolerance_hz: float = 1.0

    def __post_init__(self):
        if self.version not in (ADAPTER_VERSION, LINEAR_VERSION):
            raise ValueError("Unknown timebase adapter version")
        if not (self.native_grid_hz == 60 and self.target_hz == 30
                and 0 < self.passband_hz < self.stopband_hz <= self.target_hz / 2
                and self.attenuation_db >= 60
                and 0 < self.max_interpolation_gap_sec <= 0.15
                and 0 < self.native_rate_tolerance_hz <= 1):
            raise ValueError("Invalid 60-to-30 adapter specification")


def antialias_fir(config=TimebaseConfig()):
    """Odd, symmetric unity-DC FIR; valid convolution discards all edge padding."""
    width = (config.stopband_hz - config.passband_hz) / (config.native_grid_hz / 2)
    taps, beta = kaiserord(config.attenuation_db, width)
    taps += 1 - taps % 2
    return firwin(taps, (config.passband_hz + config.stopband_hz) / 2,
                  window=("kaiser", beta), fs=config.native_grid_hz, scale=True)


def filter_characteristics(config=TimebaseConfig()):
    h = antialias_fir(config)
    f, response = freqz(h, worN=32768, fs=config.native_grid_hz)
    amplitude = abs(response)
    return dict(taps=len(h), coefficients=h.tolist(),
        half_support_sec=(len(h) - 1) / (2 * config.native_grid_hz),
        causal_group_delay_samples=(len(h) - 1) / 2,
        retained_samples_time_shift_sec=0.0,
        passband_max_absolute_gain_error=float(abs(amplitude[f <= config.passband_hz] - 1).max()),
        stopband_max_gain_db=float(20 * np.log10(amplitude[f >= config.stopband_hz].max())),
        edge_policy="valid convolution only; no reflection, zero padding, or extrapolation")


def bounded_regularize(trace, grid, max_gap_sec, cubic):
    """Bounded interpolation; cubic never changes the existing support mask.

    Splines are fitted independently between gaps exceeding the explicit limit.
    Only finite, acquisition-valid observations enter a fit. Natural endpoint
    curvature is outside the retained full-FIR-support interval. No extrapolation.
    """
    values, imputed = resample_trace(trace, grid, max_gap_sec)
    if not cubic:
        return values, imputed
    valid = trace.validity_mask & np.isfinite(trace.values).all(axis=1)
    t, x = trace.timestamps[valid], trace.values[valid]
    splits = np.r_[0, np.flatnonzero(np.diff(t) > max_gap_sec + 1e-8) + 1, len(t)]
    for a, b in zip(splits[:-1], splits[1:]):
        if b - a < 3:
            continue
        inside = (grid >= t[a]) & (grid <= t[b-1]) & np.isfinite(values).all(axis=1)
        if inside.any():
            values[inside] = CubicSpline(t[a:b], x[a:b], axis=0, bc_type="natural", extrapolate=False)(grid[inside])
    return values, imputed


def adapt_native_face(extraction, frozen_config, config=TimebaseConfig()):
    """Return API-compatible filtered native-clock traces and explicit provenance.

    Raw traces are never changed. Effective validity is acquisition validity AND
    full FIR support. Frozen roi_source_valid_fraction therefore conservatively
    measures adapter-supported source frames; acquisition fractions are saved
    separately. Short holes up to max_interpolation_gap_sec are allowed and
    recorded; longer holes are never bridged or filtered across.
    """
    t = validate_timestamps(extraction.shared.original_timestamps)
    if len(t) < 2 or frozen_config.sample_rate != config.target_hz:
        raise ValueError("Require native timestamps and the frozen 30 Hz configuration")
    if tuple(extraction.traces) != frozen_config.requested_face_rois:
        raise ValueError("Native facial ROI ordering differs from the frozen contract")
    for trace in extraction.traces.values():
        if not np.array_equal(trace.timestamps, t):
            raise ValueError("Every native ROI must share the unmodified camera clock")
    rate = 1 / np.median(np.diff(t))
    grid = shared_grid(t, config.target_hz)
    # A true 30 Hz extraction uses the historical path without any numeric edit.
    if abs(rate - config.target_hz) < 1e-4 and np.allclose(np.diff(t), 1 / rate, atol=1e-7):
        result = _validate(FaceROIExtraction(SharedTimebase(grid, t.copy()), extraction.traces), frozen_config)
        return result, dict(version=config.version, operation="identity_30_hz", camera_pts_preserved=True)
    if abs(rate - config.native_grid_hz) > config.native_rate_tolerance_hz:
        raise ValueError(f"Unsupported median acquisition rate {rate:.6f} Hz; no implicit rate conversion")
    native_grid = shared_grid(t, config.native_grid_hz)
    h = antialias_fir(config)
    radius = (len(h) - 1) // 2
    traces, diagnostics = {}, {}
    for name, trace in extraction.traces.items():
        regular, interpolation = bounded_regularize(trace, native_grid, config.max_interpolation_gap_sec,
                                                   config.version == ADAPTER_VERSION)
        supported = np.isfinite(regular).all(axis=1)
        filtered = np.full_like(regular, np.nan)
        for a, b in valid_runs(supported):
            if b - a < len(h):
                continue
            for channel in range(3):
                filtered[a + radius:b - radius, channel] = np.convolve(regular[a:b, channel], h, mode="valid")
        reconstructed, at_camera_interpolated = bounded_regularize(
            RGBTrace(native_grid, filtered, name, np.isfinite(filtered).all(axis=1)),
            t, config.max_interpolation_gap_sec, config.version == ADAPTER_VERSION)
        valid = trace.validity_mask & np.isfinite(reconstructed).all(axis=1)
        reconstructed[~valid] = np.nan
        quality = {key: value.copy() for key, value in trace.quality.items()}
        quality.update(acquisition_valid=trace.validity_mask.copy(), adapter_supported=valid.copy(),
                       adapter_at_camera_interpolated=at_camera_interpolated)
        traces[name] = RGBTrace(t.copy(), reconstructed, trace.roi_name, valid, quality)
        target, target_interpolation = resample_trace(traces[name], grid, frozen_config.max_gap_sec)
        diagnostics[name] = dict(acquisition_valid_fraction=float(trace.validity_mask.mean()),
            effective_source_valid_fraction=float(valid.mean()),
            regularization_interpolated_fraction=float(interpolation.mean()),
            uniform_60_valid_fraction=float(supported.mean()),
            target_30_valid_fraction=float(np.isfinite(target).all(axis=1).mean()),
            target_30_interpolated_fraction=float(target_interpolation.mean()))
    result = _validate(FaceROIExtraction(SharedTimebase(grid, t.copy()), traces), frozen_config)
    dt = np.diff(t)
    report = dict(version=config.version, config=asdict(config), operation="irregular_60_to_30_external_adapter",
        median_native_hz=float(rate), camera_pts_preserved=np.array_equal(result.shared.original_timestamps, t),
        camera_first_pts_sec=float(t[0]), camera_last_pts_sec=float(t[-1]),
        native_grid_samples=len(native_grid), target_grid_samples=len(grid),
        interval_quantiles_sec=np.quantile(dt, [0, .01, .5, .99, 1]).tolist(),
        acquisition_intervals_over_1_5_native_step=int(np.sum(dt > 1.5 / config.native_grid_hz)),
        acquisition_intervals_exceeding_gap_limit=int(np.sum(dt > config.max_interpolation_gap_sec)),
        filter=filter_characteristics(config), rois=diagnostics,
        processing=["timestamp-aware bounded " + ("natural cubic" if config.version == ADAPTER_VERSION else "linear") + " interpolation onto a globally anchored 60 Hz grid",
            "centered lowpass FIR separately on contiguous valid runs, preserving DC intensity",
            "bounded interpolation of filtered values back to actual native PTS; raw RGB retained separately",
            "unchanged FrozenWaveformExtractor bounded interpolation onto globally anchored 30 Hz grid",
            "unchanged window-local RGB normalization, priors, support checks and signed inference"],
        uncertainty="Interpolation changes the transfer function at irregular intervals; FIR centering does not validate neural phase or pulse timing.")
    return result, report


def validate_on_camera_clock(timestamps, frozen_config, config=TimebaseConfig()):
    """Synthetic transfer diagnostics on the actual acquisition schedule.

    A regular FIR's frequency response alone cannot characterize interpolation
    on a jittered/dropped-frame clock. These are tests of the adapter, with no
    claim about the camera exposure response or the neural model's phase.
    """
    t = validate_timestamps(timestamps)
    results = []
    for frequency in (.7, 1.2, 3.5, 8., 20., 28.):
        wave = np.sin(2*np.pi*frequency*t)
        values = 100 + wave[:, None] * np.asarray([1., 2., .3])
        traces = {name:RGBTrace(t.copy(), values.copy(), name, np.ones(len(t), bool))
                  for name in frozen_config.requested_face_rois}
        raw = FaceROIExtraction(SharedTimebase(shared_grid(t, config.target_hz), t.copy()), traces)
        adapted, _ = adapt_native_face(raw, frozen_config, config)
        x = resample_trace(next(iter(adapted.traces.values())), adapted.shared.timestamps,
                           frozen_config.max_gap_sec)[0][:, 1]
        good = np.isfinite(x)
        alias = frequency if frequency < config.target_hz/2 else abs(frequency-config.target_hz)
        axis = adapted.shared.timestamps[good]
        matrix = np.column_stack([np.ones(len(axis)), np.sin(2*np.pi*alias*axis), np.cos(2*np.pi*alias*axis)])
        if good.sum() < 300:
            raise ValueError("Insufficient synthetic clock support to validate adapter")
        coef = np.linalg.lstsq(matrix, x[good], rcond=None)[0]
        gain = float(np.hypot(coef[1], coef[2])/2)
        phase = float(np.arctan2(coef[2], coef[1]))
        results.append(dict(input_hz=frequency, fitted_output_hz=alias, fitted_gain=gain,
            fitted_gain_db=float(20*np.log10(max(gain, 1e-15))),
            fitted_delay_ms=float(-1000*phase/(2*np.pi*frequency)) if frequency < 15 else None,
            output_ac_rms_relative_to_input=float(np.std(x[good])/np.sqrt(2)),
            finite_target_samples=int(good.sum())))
    # The unchanged API finally uses linear interpolation at the actual 60 Hz
    # PTS. Even an ideal 8 Hz sinusoid has a worst half-step gain cos(pi*8/60)
    # = .9135 there. V2 requires <=2% gain error through the 3.5 Hz HR band and
    # <=10% at 8 Hz, explicitly reporting that residual morphology limitation.
    passed = all((.90 if config.version == ADAPTER_VERSION and row["input_hz"] == 8 else .98)
                 <= row["fitted_gain"] <= 1.02 and abs(row["fitted_delay_ms"]) <= 5
                 if row["input_hz"] <= 8 else row["fitted_gain_db"] <= -40 for row in results)
    return dict(adapter_version=config.version, passed=passed,
        criteria=("0.7/1.2/3.5 Hz fitted gain within 2%; 8 Hz gain 0.90–1.02; delay within 5 ms; 20/28 Hz fitted alias ≤ -40 dB"
                  if config.version == ADAPTER_VERSION else
                  "0.7/1.2/3.5/8 Hz fitted gain within 2%, delay within 5 ms; 20/28 Hz fitted alias ≤ -40 dB"),
        tests=results, scope="Synthetic adapter transfer on actual PTS only; no neural/camera/physiological phase validation")
