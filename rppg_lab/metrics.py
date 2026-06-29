"""Evaluation metrics for HR/rPPG/BP experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    me: float
    sd: float
    pearson: float
    n: int


def regression_metrics(y_true, y_pred) -> RegressionMetrics:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if y.size == 0:
        return RegressionMetrics(np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    e = p - y
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e * e)))
    me = float(np.mean(e))
    sd = float(np.std(e, ddof=1)) if e.size > 1 else 0.0
    r = float(np.corrcoef(y, p)[0, 1]) if e.size > 1 and np.std(y) > 0 and np.std(p) > 0 else np.nan
    return RegressionMetrics(mae, rmse, me, sd, r, int(y.size))


def bland_altman(y_true, y_pred) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    m = np.isfinite(y) & np.isfinite(p)
    d = p[m] - y[m]
    if d.size == 0:
        return {"bias": np.nan, "sd": np.nan, "loa_low": np.nan, "loa_high": np.nan}
    bias = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    return {"bias": bias, "sd": sd, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}


def waveform_corr(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    n = min(y.size, p.size)
    y, p = y[:n], p[:n]
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if y.size < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])

def waveform_corr_aligned(y_true, y_pred, max_lag: int = 15) -> float:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    n = min(y.size, p.size)
    y, p = y[:n], p[:n]

    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]

    if y.size < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan

    best = -np.inf

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            yy = y[-lag:]
            pp = p[:lag]
        elif lag > 0:
            yy = y[:-lag]
            pp = p[lag:]
        else:
            yy = y
            pp = p

        if yy.size < 3 or np.std(yy) == 0 or np.std(pp) == 0:
            continue

        c = np.corrcoef(yy, pp)[0, 1]
        if np.isfinite(c):
            best = max(best, float(c))

    return best if np.isfinite(best) else np.nan


def aami_summary(y_true, y_pred) -> Dict[str, float | bool]:
    """Convenience BP agreement summary.

    This does not replace full ISO/AAMI validation. It reports the usual mean
    error and SD thresholds often used as a first sanity check.
    """
    m = regression_metrics(y_true, y_pred)
    return {
        "mean_error_mmHg": m.me,
        "error_sd_mmHg": m.sd,
        "mae_mmHg": m.mae,
        "rmse_mmHg": m.rmse,
        "passes_5_8_sanity": bool(abs(m.me) <= 5.0 and m.sd <= 8.0),
        "n": m.n,
    }
