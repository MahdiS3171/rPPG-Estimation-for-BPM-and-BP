"""Traditional rPPG construction methods.

All functions accept an RGB trace with shape (T, 3) in RGB channel order and return
one 1-D rPPG waveform of length T. These methods are intentionally kept simple,
transparent, and suitable as baselines/priors for deep models.
"""
from __future__ import annotations

from typing import Callable, Dict
import numpy as np
from scipy.signal import butter, filtfilt, detrend as scipy_detrend, welch
from sklearn.decomposition import PCA, FastICA

EPS = 1e-8


def check_rgb(rgb_ts: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_ts, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb_ts must have shape (T, 3), got {rgb.shape}")
    if rgb.shape[0] < 4:
        raise ValueError("rgb_ts is too short")
    return rgb


def standardize_channels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - np.nanmean(x, axis=0, keepdims=True)) / (np.nanstd(x, axis=0, keepdims=True) + EPS)


def mean_normalize_channels(rgb: np.ndarray) -> np.ndarray:
    return rgb / (np.nanmean(rgb, axis=0, keepdims=True) + EPS) - 1.0


def bandpass_filter(sig: np.ndarray, fs: float, f_low: float = 0.7, f_high: float = 4.0, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth bandpass in the physiological HR band."""
    x = np.asarray(sig, dtype=np.float64)
    x = np.nan_to_num(x - np.nanmean(x), nan=0.0)
    fs = float(fs)
    if fs <= 0 or x.size < max(12, order * 6):
        return x
    nyq = 0.5 * fs
    low = max(1e-6, f_low / nyq)
    high = min(0.999, f_high / nyq)
    if not (0 < low < high < 1):
        return x
    b, a = butter(order, [low, high], btype="band")
    try:
        return filtfilt(b, a, x)
    except ValueError:
        return x


def _finish(sig: np.ndarray, fs: float, do_filter: bool = True) -> np.ndarray:
    y = np.asarray(sig, dtype=np.float64)
    y = np.nan_to_num(y - np.nanmean(y), nan=0.0)
    if do_filter:
        y = bandpass_filter(y, fs)
    return y.astype(np.float32)


def green(rgb_ts: np.ndarray, fs: float, do_filter: bool = True) -> np.ndarray:
    rgb = check_rgb(rgb_ts)
    return _finish(rgb[:, 1], fs, do_filter)


def chrom(rgb_ts: np.ndarray, fs: float, do_filter: bool = True) -> np.ndarray:
    """CHROM-style projection (de Haan & Jeanne)."""
    rgb = check_rgb(rgb_ts)
    r, g, b = mean_normalize_channels(rgb).T
    x1 = 3.0 * r - 2.0 * g
    x2 = 1.5 * r + g - 1.5 * b
    alpha = np.nanstd(x1) / (np.nanstd(x2) + EPS)
    return _finish(x1 - alpha * x2, fs, do_filter)


def pos(rgb_ts: np.ndarray, fs: float, do_filter: bool = True) -> np.ndarray:
    """Single-window POS projection.

    For publication-grade classical baselines, prefer pos_windowed because the
    original POS formulation uses overlap-add windowing.
    """
    rgb = check_rgb(rgb_ts)
    r, g, b = mean_normalize_channels(rgb).T
    x1 = r - g
    x2 = r + g - 2.0 * b
    alpha = np.nanstd(x1) / (np.nanstd(x2) + EPS)
    return _finish(x1 + alpha * x2, fs, do_filter)


def _overlap_add_windows(rgb: np.ndarray, fs: float, method: str, win_sec: float = 1.6) -> np.ndarray:
    rgb = check_rgb(rgb)
    n = rgb.shape[0]
    win = max(8, int(round(win_sec * fs)))
    if win >= n:
        return chrom(rgb, fs, do_filter=False) if method == "chrom" else pos(rgb, fs, do_filter=False)

    out = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    for start in range(0, n - win + 1):
        seg = rgb[start:start + win]
        if method == "pos":
            y = pos(seg, fs, do_filter=False)
        elif method == "chrom":
            y = chrom(seg, fs, do_filter=False)
        else:
            raise ValueError(method)
        y = y - np.nanmean(y)
        out[start:start + win] += y
        counts[start:start + win] += 1.0
    counts[counts == 0] = 1.0
    return out / counts


def pos_windowed(rgb_ts: np.ndarray, fs: float, win_sec: float = 1.6, do_filter: bool = True) -> np.ndarray:
    return _finish(_overlap_add_windows(check_rgb(rgb_ts), fs, "pos", win_sec), fs, do_filter)


def chrom_windowed(rgb_ts: np.ndarray, fs: float, win_sec: float = 1.6, do_filter: bool = True) -> np.ndarray:
    return _finish(_overlap_add_windows(check_rgb(rgb_ts), fs, "chrom", win_sec), fs, do_filter)


def pca_method(rgb_ts: np.ndarray, fs: float, do_filter: bool = True) -> np.ndarray:
    rgb = check_rgb(rgb_ts)
    x = standardize_channels(rgb)
    comps = PCA(n_components=3).fit_transform(x)
    # Pick component with maximum HR-band power rather than blindly choosing PC1.
    best = _select_hr_band_component(comps, fs)
    return _finish(best, fs, do_filter)


def ica_method(rgb_ts: np.ndarray, fs: float, do_filter: bool = True) -> np.ndarray:
    rgb = check_rgb(rgb_ts)
    x = standardize_channels(rgb)
    try:
        comps = FastICA(n_components=3, random_state=0, max_iter=1000, whiten="unit-variance").fit_transform(x)
    except Exception:
        comps = PCA(n_components=3).fit_transform(x)
    best = _select_hr_band_component(comps, fs)
    return _finish(best, fs, do_filter)


def _select_hr_band_component(comps: np.ndarray, fs: float, fmin: float = 0.7, fmax: float = 4.0) -> np.ndarray:
    best_idx, best_power = 0, -np.inf
    for i in range(comps.shape[1]):
        f, pxx = welch(comps[:, i], fs=float(fs), nperseg=min(len(comps), max(16, int(fs * 8))))
        band = (f >= fmin) & (f <= fmax)
        power = float(np.sum(pxx[band])) if np.any(band) else -np.inf
        if power > best_power:
            best_idx, best_power = i, power
    return comps[:, best_idx]


def pbv(rgb_ts: np.ndarray, fs: float, do_filter: bool = True) -> np.ndarray:
    """Simplified PBV-like projection.

    This is a baseline approximation, not a claim of exact reproduction of every
    PBV variant in the literature.
    """
    rgb = check_rgb(rgb_ts)
    x = standardize_channels(rgb)
    sigma = np.cov(x, rowvar=False)
    u = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    try:
        w = np.linalg.solve(sigma + 1e-6 * np.eye(3), u)
    except np.linalg.LinAlgError:
        w = u
    return _finish(x @ w, fs, do_filter)


def lgi(rgb_ts: np.ndarray, fs: float, do_filter: bool = True) -> np.ndarray:
    """Log-chrominance LGI-inspired baseline."""
    rgb = np.clip(check_rgb(rgb_ts), 1e-6, None)
    log_rgb = np.log(rgb)
    x1 = log_rgb[:, 1] - log_rgb[:, 2]
    x2 = log_rgb[:, 1] - log_rgb[:, 0]
    alpha = np.nanstd(x1) / (np.nanstd(x2) + EPS)
    return _finish(x1 - alpha * x2, fs, do_filter)


def omit(rgb_ts: np.ndarray, fs: float, do_filter: bool = True) -> np.ndarray:
    """OMIT-inspired baseline: remove intensity direction and project in chromatic plane."""
    rgb = check_rgb(rgb_ts)
    x = rgb - np.nanmean(rgb, axis=0, keepdims=True)
    n = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    n /= np.linalg.norm(n)
    xp = x - (x @ n)[:, None] * n[None, :]
    if np.allclose(xp, 0):
        y = x[:, 1]
    else:
        y = PCA(n_components=1).fit_transform(xp).ravel()
    return _finish(y, fs, do_filter)


def detrended(method: Callable[[np.ndarray, float], np.ndarray]) -> Callable[[np.ndarray, float], np.ndarray]:
    def wrapped(rgb_ts: np.ndarray, fs: float) -> np.ndarray:
        y = method(rgb_ts, fs)
        return bandpass_filter(scipy_detrend(y), fs).astype(np.float32)
    return wrapped


METHOD_FUNCS: Dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "GREEN": green,
    "CHROM": chrom,
    "CHROM_WIN": chrom_windowed,
    "POS": pos,
    "POS_WIN": pos_windowed,
    "PCA": pca_method,
    "ICA": ica_method,
    "PBV": pbv,
    "LGI": lgi,
    "OMIT": omit,
}

DEFAULT_PRIORS = ["GREEN", "CHROM", "CHROM_WIN", "PBV", "POS_WIN", "OMIT"]
