"""Signal processing, HR estimation, PTT/phase, and feature extraction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, welch, find_peaks, correlate, correlation_lags

EPS = 1e-8


@dataclass
class HREstimate:
    hr_bpm: float
    hr_hz: float
    confidence: float
    snr_db: float


def standardize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x - np.nanmean(x)) / (np.nanstd(x) + EPS)).astype(np.float32)


def standardize_channels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x - np.nanmean(x, axis=0, keepdims=True)) / (np.nanstd(x, axis=0, keepdims=True) + EPS)).astype(np.float32)


def bandpass(sig: np.ndarray, fs: float, f_low: float = 0.7, f_high: float = 4.0, order: int = 3) -> np.ndarray:
    x = np.asarray(sig, dtype=np.float64)
    x = np.nan_to_num(x - np.nanmean(x), nan=0.0)
    fs = float(fs)
    if x.size < max(12, order * 6):
        return x.astype(np.float32)
    nyq = 0.5 * fs
    low = max(1e-6, f_low / nyq)
    high = min(0.999, f_high / nyq)
    if low >= high:
        return x.astype(np.float32)
    b, a = butter(order, [low, high], btype="band")
    try:
        return filtfilt(b, a, x).astype(np.float32)
    except ValueError:
        return x.astype(np.float32)


def resample_uniform(
    t: np.ndarray,
    x: np.ndarray,
    fs_target: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample 1-D or 2-D signal x(t) to a uniform grid."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x)
    ok = np.isfinite(t)
    if x.ndim == 1:
        ok &= np.isfinite(x)
    else:
        ok &= np.all(np.isfinite(x), axis=1)
    t = t[ok]
    x = x[ok]
    if t.size < 2:
        raise ValueError("Need at least two valid timestamps to resample")
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    # Drop duplicate timestamps.
    keep = np.r_[True, np.diff(t) > 1e-9]
    t = t[keep]
    x = x[keep]
    if t_start is None:
        t_start = float(t[0])
    if t_end is None:
        t_end = float(t[-1])
    if t_end <= t_start:
        raise ValueError("t_end must be greater than t_start")
    tu = np.arange(t_start, t_end, 1.0 / float(fs_target), dtype=np.float64)
    f = interp1d(t, x, axis=0, bounds_error=False, fill_value="extrapolate")
    return tu, f(tu)


def estimate_hr_welch(
    sig: np.ndarray,
    fs: float,
    fmin: float = 0.7,
    fmax: float = 3.5,
    window_sec: float = 30.0,
    zero_pad_factor: int = 4,
) -> HREstimate:
    x = np.asarray(sig, dtype=np.float64)
    x = np.nan_to_num(x - np.nanmean(x), nan=0.0)
    fs = float(fs)

    if x.size < int(fs * 3):
        return HREstimate(np.nan, np.nan, 0.0, -np.inf)

    nperseg = min(len(x), max(32, int(round(fs * window_sec))))

    # Zero-padding gives a smoother PSD grid. It does not create new
    # information, but it helps avoid coarse 7.5 bpm jumps in peak picking.
    nfft = int(2 ** np.ceil(np.log2(max(nperseg, zero_pad_factor * nperseg))))

    f, pxx = welch(x, fs=fs, nperseg=nperseg, nfft=nfft)
    band = (f >= fmin) & (f <= fmax)

    if not np.any(band):
        return HREstimate(np.nan, np.nan, 0.0, -np.inf)

    fb = f[band]
    pb = pxx[band]

    idx = int(np.argmax(pb))

    # Parabolic interpolation around PSD peak for sub-bin refinement.
    if 0 < idx < len(pb) - 1:
        y0 = np.log(pb[idx - 1] + EPS)
        y1 = np.log(pb[idx] + EPS)
        y2 = np.log(pb[idx + 1] + EPS)
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > EPS:
            delta = 0.5 * (y0 - y2) / denom
            delta = float(np.clip(delta, -1.0, 1.0))
            df = fb[1] - fb[0]
            f0 = float(fb[idx] + delta * df)
        else:
            f0 = float(fb[idx])
    else:
        f0 = float(fb[idx])

    mean_band = float(np.mean(pb) + EPS)
    confidence = float(np.clip(pb[idx] / (3.0 * mean_band), 0.0, 1.0))
    snr = snr_db_from_psd(f, pxx, f0, fmin=fmin, fmax=fmax)

    return HREstimate(
        hr_bpm=60.0 * f0,
        hr_hz=f0,
        confidence=confidence,
        snr_db=snr,
    )


def snr_db_from_psd(
    f: np.ndarray,
    pxx: np.ndarray,
    f0: float,
    main_bw_hz: float = 0.12,
    fmin: float = 0.7,
    fmax: float = 4.0,
    harmonics: int = 2,
) -> float:
    f = np.asarray(f, dtype=np.float64)
    pxx = np.asarray(pxx, dtype=np.float64)
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return -np.inf
    fb, pb = f[band], pxx[band]
    sig_power = 0.0
    for k in range(1, harmonics + 1):
        fk = k * f0
        if fk < fmin or fk > fmax:
            continue
        mask = (fb >= fk - main_bw_hz) & (fb <= fk + main_bw_hz)
        sig_power += float(np.sum(pb[mask]))
    total = float(np.sum(pb) + EPS)
    noise = max(EPS, total - sig_power)
    return float(10.0 * np.log10(max(EPS, sig_power) / noise))


def align_by_xcorr(x: np.ndarray, y: np.ndarray, fs: float, max_lag_sec: float = 0.5) -> Tuple[float, float]:
    """Estimate lag y-vs-x via cross-correlation.

    Returns
    -------
    lag_sec : positive means y lags x
    score   : normalized correlation score at the selected lag
    """
    x = standardize_1d(x)
    y = standardize_1d(y)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    c = correlate(y, x, mode="full")
    lags = correlation_lags(y.size, x.size, mode="full")
    max_lag = int(round(max_lag_sec * fs))
    keep = np.abs(lags) <= max_lag
    c, lags = c[keep], lags[keep]
    idx = int(np.argmax(c))
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y)) + EPS
    return float(lags[idx] / fs), float(c[idx] / denom)


def phase_delay_at_hr(x: np.ndarray, y: np.ndarray, fs: float, hr_hz: Optional[float] = None) -> Tuple[float, float]:
    """Estimate y-vs-x delay from the phase difference at HR frequency.

    Positive delay means y lags x. For noisy rPPG, this should be used together
    with xcorr delay and quality metrics, not as a standalone BP estimator.
    """
    x = standardize_1d(x)
    y = standardize_1d(y)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    if hr_hz is None or not np.isfinite(hr_hz):
        hr_hz = estimate_hr_welch(x, fs).hr_hz
    if not np.isfinite(hr_hz) or hr_hz <= 0:
        return np.nan, np.nan
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    k = int(np.argmin(np.abs(freqs - hr_hz)))
    if k <= 0:
        return np.nan, np.nan
    phase = np.angle(Y[k] * np.conj(X[k]))
    delay = phase / (2.0 * np.pi * freqs[k])
    # Wrap to half a cycle.
    period = 1.0 / freqs[k]
    if delay > 0.5 * period:
        delay -= period
    if delay < -0.5 * period:
        delay += period
    coherence_proxy = float(np.abs(Y[k] * np.conj(X[k])) / (np.abs(Y[k]) * np.abs(X[k]) + EPS))
    return float(delay), coherence_proxy


def pulse_wave_features(sig: np.ndarray, fs: float) -> Dict[str, float]:
    """Extract simple PWA-style features from one rPPG/PPG segment.

    These features are intentionally conservative; they are useful for pilot BP
    modeling but should not be treated as clinical-grade morphology estimates.
    """
    x = bandpass(sig, fs)
    x = standardize_1d(x)
    hr = estimate_hr_welch(x, fs)
    min_dist = max(1, int(0.30 * fs))
    peaks, _ = find_peaks(x, distance=min_dist, prominence=max(0.05, 0.2 * np.std(x)))
    troughs, _ = find_peaks(-x, distance=min_dist)
    out: Dict[str, float] = {
        "hr_bpm": hr.hr_bpm,
        "hr_confidence": hr.confidence,
        "snr_db": hr.snr_db,
        "amp_mean": np.nan,
        "amp_std": np.nan,
        "width50_mean_sec": np.nan,
        "upstroke_mean_sec": np.nan,
        "ibi_mean_sec": np.nan,
        "ibi_std_sec": np.nan,
    }
    if peaks.size >= 3:
        ibi = np.diff(peaks) / fs
        out["ibi_mean_sec"] = float(np.mean(ibi))
        out["ibi_std_sec"] = float(np.std(ibi))
    amps = []
    widths = []
    upstrokes = []
    for p in peaks:
        prev_troughs = troughs[troughs < p]
        next_troughs = troughs[troughs > p]
        if prev_troughs.size == 0 or next_troughs.size == 0:
            continue
        l = int(prev_troughs[-1])
        r = int(next_troughs[0])
        amp = float(x[p] - min(x[l], x[r]))
        if amp <= 0:
            continue
        amps.append(amp)
        half = min(x[l], x[r]) + 0.5 * amp
        segment = x[l:r + 1]
        above = np.where(segment >= half)[0]
        if above.size > 1:
            widths.append(float((above[-1] - above[0]) / fs))
        upstrokes.append(float((p - l) / fs))
    if amps:
        out["amp_mean"] = float(np.mean(amps))
        out["amp_std"] = float(np.std(amps))
    if widths:
        out["width50_mean_sec"] = float(np.mean(widths))
    if upstrokes:
        out["upstroke_mean_sec"] = float(np.mean(upstrokes))
    return out


def two_site_features(proximal: np.ndarray, distal: np.ndarray, fs: float, distance_m: Optional[float] = None) -> Dict[str, float]:
    """Features for future face-hand BP experiments."""
    prox = bandpass(proximal, fs)
    dist = bandpass(distal, fs)
    hr = estimate_hr_welch(prox, fs)
    xlag, xscore = align_by_xcorr(prox, dist, fs, max_lag_sec=0.6)
    plag, pscore = phase_delay_at_hr(prox, dist, fs, hr_hz=hr.hr_hz)
    out = {
        "ptt_xcorr_sec": xlag,
        "ptt_xcorr_score": xscore,
        "ptt_phase_sec": plag,
        "ptt_phase_score": pscore,
    }
    if distance_m is not None and np.isfinite(xlag) and xlag > 0:
        out["pwv_xcorr_m_s"] = float(distance_m / xlag)
    else:
        out["pwv_xcorr_m_s"] = np.nan
    if distance_m is not None and np.isfinite(plag) and plag > 0:
        out["pwv_phase_m_s"] = float(distance_m / plag)
    else:
        out["pwv_phase_m_s"] = np.nan
    return out
