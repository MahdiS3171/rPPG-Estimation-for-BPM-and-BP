"""Exploratory frame/mask geometry evidence only; never reads d_ppg or chooses a shift."""
from __future__ import annotations

import cv2
import h5py
import numpy as np


def continuous_edge_agreement(rgb: np.ndarray, stored_mask: np.ndarray) -> float | None:
    """Cosine of spatial gradient magnitudes, WITHOUT thresholding raw masks.

    This is an unvalidated geometric descriptor, not segmentation ground truth
    or a timing acceptance test. Similar/static frames need not identify an offset.
    """
    if (rgb.ndim != 3 or rgb.shape[-1] != 3 or stored_mask.shape != rgb.shape[:2]
            or not np.isfinite(rgb).all() or not np.isfinite(stored_mask).all()):
        raise ValueError("Require finite spatially corresponding RGB/mask arrays")
    gray = np.mean(rgb, axis=-1).astype(np.float64)
    def magnitude(values):
        x = cv2.Sobel(values.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
        y = cv2.Sobel(values.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
        return np.hypot(x, y)
    a, b = magnitude(gray).ravel(), magnitude(stored_mask).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else None


def audit_mask_candidates(recording, indices=(1, 149, 299, 449, 597)) -> dict:
    """Inspect three RGB neighbours of each mask index; no label dataset is opened.

    Original arrays/indices remain unchanged. Offsets refer only to candidate
    comparisons in this report and are NEVER applied to an extraction pathway.
    """
    rows = []
    count = recording.layout.frames
    if not indices or any(type(i) is not int or not 1 <= i < count - 1 for i in indices):
        raise ValueError("Interior explicit mask indices required for preceding/same/following comparisons")
    with h5py.File(recording.path, "r", rdcc_nbytes=1024 * 1024) as f:
        for index in indices:
            if any(index >= f[name].shape[-1] for name in ("skin_mask", "combined_mask")):
                raise ValueError("Requested mask index outside its original stream")
            frames = f["Xsub"][:, :, :, index-1:index+2].transpose(3, 2, 1, 0)
            row = dict(mask_index=index, candidate_frame_indices=[index-1, index, index+1],
                       adjacent_rgb_mean_absolute_difference=[float(np.abs(frames[k+1] - frames[k]).mean()) for k in (0, 1)])
            row["edge_agreement"] = {}
            for name in ("skin_mask", "combined_mask"):
                mask = f[name][:, :, index].T
                row["edge_agreement"][name] = {str(offset): continuous_edge_agreement(frames[k], mask)
                                                for k, offset in enumerate((-1, 0, 1))}
            rows.append(row)
    return dict(filename=recording.path.name, source_sha256=recording.info["source_sha256"],
                evidence="unthresholded_spatial_edge_agreement_and_adjacent_RGB_differences",
                ppg_read_or_scored=False, candidate_offsets_applied=False, mask_values_modified=False,
                decision="UNVERIFIED", rows=rows,
                limitation="No independently validated geometric alignment criterion; similar/static poses may not distinguish adjacent frames")
