"""Shared discrete overlap convention for waveform training and evaluation."""
from __future__ import annotations


def lag_candidates(length: int, max_lag: int) -> list[int]:
    """Keep at least three samples when possible; short windows use zero lag.

    Ties visit zero first, then smaller absolute lags, then negative before
    positive. A positive lag means prediction lags reference (trim prediction
    at its beginning and reference at its end).
    """
    if length < 1 or type(max_lag) is not int or max_lag < 0:
        raise ValueError("Require nonempty time dimension and nonnegative integer max_lag")
    limit = min(max_lag, max(0, length - 3))
    return sorted(range(-limit, limit + 1), key=lambda lag: (abs(lag), lag))


def aligned_overlap(pred, target, lag: int):
    """Slice tensor/array overlaps without wrapping, interpolation or warping."""
    if lag < 0:
        return pred[..., :lag], target[..., -lag:]
    if lag > 0:
        return pred[..., lag:], target[..., :-lag]
    return pred, target
