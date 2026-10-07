"""Window-local training priors; distinct from recording-wide Phase 1 priors."""
from __future__ import annotations

from typing import Sequence
import numpy as np

from .classical import DEFAULT_PRIORS, METHOD_FUNCS
from .signals import standardize_1d


def build_window_priors(
    rgb_window: np.ndarray,
    fs: float,
    methods: Sequence[str] = tuple(DEFAULT_PRIORS),
) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized [K,L] priors and bool [K] success mask.

    Only the supplied [L,3] raw-intensity window is visible to each unchanged
    METHOD_FUNCS algorithm, including its internal filtering. Each method gets
    its own copy. Exceptions, wrong shape and nonfinite output produce an
    invalid, all-zero channel; zero/constant finite output is still a successful
    computation. No state or recording-level statistics are retained.
    """
    rgb = np.asarray(rgb_window, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or len(rgb) < 4 or not np.isfinite(rgb).all():
        raise ValueError("rgb_window must be finite [L,3] with L >= 4")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    if isinstance(methods, str) or not methods or len(set(methods)) != len(methods) or not set(methods) <= METHOD_FUNCS.keys():
        raise ValueError("methods must be nonempty, unique classical method names")
    priors = np.zeros((len(methods), len(rgb)), dtype=np.float32)
    valid = np.zeros(len(methods), dtype=bool)
    for k, name in enumerate(methods):
        try:
            waveform = np.asarray(METHOD_FUNCS[name](rgb.copy(), fs), dtype=np.float64)
            if waveform.shape != (len(rgb),) or not np.isfinite(waveform).all():
                continue
            normalized = standardize_1d(waveform)
            if not np.isfinite(normalized).all():
                continue
        except Exception:
            # A failed algorithm is represented by a mask, never a valid zero prior.
            continue
        priors[k] = normalized
        valid[k] = True
    return priors, valid
