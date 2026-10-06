"""Label-free exploratory optical delay and pulse morphology features."""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate, correlation_lags, coherence, find_peaks, peak_widths
from scipy.stats import skew, kurtosis
from .types import PhysiologicalSignal
from .quality import signal_quality


MORPHOLOGY_KEYS = ("amplitude_native", "amplitude_normalized", "rise_time_sec", "fall_time_sec",
                   "width50_sec", "area_normalized_sec", "ibi_mean_sec", "ibi_std_sec",
                   "derivative_max_per_sec", "second_derivative_std_per_sec2", "skewness", "kurtosis", "beat_count")
DELAY_KEYS = ("delay_xcorr_sec", "delay_xcorr_score", "delay_phase_sec", "delay_phase_coherence",
              "delay_beat_median_sec", "delay_beat_std_sec", "delay_valid_beat_fraction")
QUALITY_KEYS = ("hr_bpm", "snr_db", "spectral_peak_to_mean", "periodicity", "amplitude_std")
FEATURE_KEYS = tuple(f"{site}_{key}" for site in ("face", "hand") for key in MORPHOLOGY_KEYS + QUALITY_KEYS) + DELAY_KEYS + (
    "face_hand_hr_difference_bpm", "face_hand_zero_lag_correlation", "face_valid_frame_fraction", "hand_valid_frame_fraction",
    "face_interpolated_fraction", "hand_interpolated_fraction")


def estimate_delay(face: np.ndarray, hand: np.ndarray, fs: float, max_delay_sec: float,
                   hr_hz: float) -> dict[str, float]:
    """Positive delay = hand lags face. No shifting/alignment of stored signals.

    Optical polarity is assumed consistent. Delay is ambiguous modulo a pulse
    period; bound it by an a priori configured search range. Fractional estimates
    interpolate the correlation peak, not acquisition resolution.
    """
    x, y = np.asarray(face, dtype=float), np.asarray(hand, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Delay inputs must be equal, finite, one-dimensional arrays")
    if fs <= 0 or max_delay_sec <= 0:
        raise ValueError("Invalid delay settings")
    out = dict.fromkeys(DELAY_KEYS, float("nan"))
    if len(x) < 3 * fs or min(x.std(), y.std()) < 1e-10 or not np.isfinite(hr_hz) or hr_hz <= 0:
        return out
    # Restrict to half a cycle to reduce periodic ambiguity.
    max_lag = min(int(np.floor(max_delay_sec * fs)), int(np.floor(0.49 * fs / hr_hz)), len(x) - 3)
    if max_lag < 1:
        return out
    x, y = x - x.mean(), y - y.mean()
    corr = correlate(y, x, mode="full", method="fft")
    lags = correlation_lags(len(y), len(x))
    keep = np.abs(lags) <= max_lag
    corr, lags = corr[keep], lags[keep]
    corr /= np.sqrt(np.sum(x*x) * np.sum(y*y))
    k = int(np.argmax(corr))
    offset = 0.0
    if 0 < k < len(corr) - 1:
        denom = corr[k-1] - 2*corr[k] + corr[k+1]
        if abs(denom) > 1e-12:
            offset = float(np.clip(0.5 * (corr[k-1] - corr[k+1]) / denom, -0.5, 0.5))
        out["delay_xcorr_sec"] = float((lags[k] + offset) / fs)
    # A maximum at the search boundary is unresolved, not a valid lag.
    out["delay_xcorr_score"] = float(corr[k])
    t = np.arange(len(x)) / fs
    basis = np.exp(-2j * np.pi * hr_hz * t)
    X, Y = np.dot(x, basis), np.dot(y, basis)
    out["delay_phase_sec"] = float(-np.angle(Y * np.conj(X)) / (2 * np.pi * hr_hz))
    nperseg = min(int(4 * fs), len(x) // 2)
    if nperseg >= int(2 * fs):
        f, c = coherence(x, y, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        out["delay_phase_coherence"] = float(c[np.argmin(abs(f - hr_hz))])
    distance = max(1, int(0.6 * fs / hr_hz))
    xp, _ = find_peaks(x, distance=distance, prominence=0.3*x.std())
    yp, _ = find_peaks(y, distance=distance, prominence=0.3*y.std())
    used, delays = set(), []
    for p in xp:
        eligible = [q for q in yp if q not in used and abs(q-p) <= max_lag]
        if eligible:
            q = min(eligible, key=lambda q: abs(q-p))
            used.add(q)
            delays.append((q-p)/fs)
    if delays:
        out["delay_beat_median_sec"] = float(np.median(delays))
        out["delay_beat_std_sec"] = float(np.std(delays))
    out["delay_valid_beat_fraction"] = len(delays) / max(1, len(xp))
    return out


def morphology(values: np.ndarray, fs: float, hr_hz: float) -> dict[str, float]:
    """Features of a filtered algorithm waveform, not calibrated blood volume.

    Native amplitudes depend on algorithm/illumination. Normalized quantities use
    segment SD. Trough/peak morphology is exploratory; no dicrotic claims.
    """
    x = np.asarray(values, dtype=float)
    out = dict.fromkeys(MORPHOLOGY_KEYS, float("nan"))
    if len(x) < 3*fs or not np.isfinite(x).all() or x.std() < 1e-10 or not np.isfinite(hr_hz):
        return out
    z = (x-x.mean())/x.std()
    distance = max(1, int(0.6 * fs / hr_hz))
    peaks, _ = find_peaks(z, distance=distance, prominence=0.3)
    troughs, _ = find_peaks(-z, distance=distance)
    beats = []
    for p in peaks:
        left, right = troughs[troughs < p], troughs[troughs > p]
        if len(left) and len(right):
            l, r = left[-1], right[0]
            beats.append((p, l, r))
    if beats:
        out.update(amplitude_native=float(np.mean([x[p]-x[l] for p,l,r in beats])),
                   amplitude_normalized=float(np.mean([z[p]-z[l] for p,l,r in beats])),
                   rise_time_sec=float(np.mean([(p-l)/fs for p,l,r in beats])),
                   fall_time_sec=float(np.mean([(r-p)/fs for p,l,r in beats])),
                   width50_sec=float(np.mean(peak_widths(z, [p for p,l,r in beats])[0])/fs),
                   area_normalized_sec=float(np.mean([np.sum(np.maximum(z[l:r+1]-z[l], 0))/fs for p,l,r in beats])))
    if len(peaks) > 1:
        out["ibi_mean_sec"] = float(np.mean(np.diff(peaks))/fs)
        out["ibi_std_sec"] = float(np.std(np.diff(peaks))/fs)
    d = np.gradient(z, 1/fs)
    out.update(derivative_max_per_sec=float(d.max()), second_derivative_std_per_sec2=float(np.gradient(d, 1/fs).std()),
               skewness=float(skew(z)), kurtosis=float(kurtosis(z)), beat_count=float(len(beats)))
    return out


def extract_features(face: PhysiologicalSignal, hand: PhysiologicalSignal, max_delay_sec: float = 0.3,
                     fmin: float = 0.7, fmax: float = 3.5) -> dict[str, float]:
    """Consumes only signals. Reference labels and demographics are not inputs."""
    if face.sample_rate != hand.sample_rate or not np.array_equal(face.timestamps, hand.timestamps):
        raise ValueError("Face and hand must use the exact common time base")
    if not np.isfinite(face.values).all() or not np.isfinite(hand.values).all():
        raise ValueError("Select a contiguous finite paired interval; do not compress gaps")
    out = dict.fromkeys(FEATURE_KEYS, float("nan"))
    qualities = {}
    for name, signal in (("face", face), ("hand", hand)):
        q = signal_quality(signal.values, signal.sample_rate, fmin, fmax)
        qualities[name] = q
        morph = morphology(signal.values, signal.sample_rate, q["hr_bpm"]/60)
        out.update({f"{name}_{k}": v for k,v in {**q, **morph}.items()})
    out.update(estimate_delay(face.values, hand.values, face.sample_rate, max_delay_sec, qualities["face"]["hr_bpm"]/60))
    out["face_hand_hr_difference_bpm"] = abs(qualities["face"]["hr_bpm"]-qualities["hand"]["hr_bpm"])
    if min(face.values.std(), hand.values.std()) > 1e-10:
        out["face_hand_zero_lag_correlation"] = float(np.corrcoef(face.values, hand.values)[0,1])
    return out
