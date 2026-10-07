"""Classical comparisons on the exact Part 2 evaluation tensors, without extraction."""
from __future__ import annotations

import hashlib
import numpy as np


def target_sha256(target, reference_hr: float) -> str:
    """Fingerprint the shared target and HR label, including their sample order."""
    return hashlib.sha256(np.asarray(target, dtype="<f4").tobytes()
                          + np.asarray([reference_hr], dtype="<f4").tobytes()).hexdigest()


def classical_window_predictions(x, roi_valid, prior_valid, roi_names, prior_names):
    """Yield every ROI/prior and validity-aware equal-ROI prior average.

    Unavailable baselines retain the window with NaN metrics. They never gain
    an easier subset silently, and neither polarity nor lag is optimized here.
    """
    x = np.asarray(x)
    valid = np.asarray(prior_valid, bool) & np.asarray(roi_valid, bool)[:, None]
    for r, roi in enumerate(roi_names):
        for k, prior in enumerate(prior_names):
            yield f"{roi}/{prior}", (x[r, 3 + k] if valid[r, k]
                                      else np.full(x.shape[-1], np.nan))
    for k, prior in enumerate(prior_names):
        keep = valid[:, k]
        yield f"{prior} multi-ROI average", (x[keep, 3 + k].mean(axis=0) if keep.any()
                                             else np.full(x.shape[-1], np.nan))
