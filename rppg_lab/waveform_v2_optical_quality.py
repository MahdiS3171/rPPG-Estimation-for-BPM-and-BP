"""Label-free optical descriptors; none are confirmed skin/pulse confidence.

Existing V1 YCrCb thresholds are reused as an experimental color-compatibility
proxy. Skin-colored hair and illumination/pigmentation changes can defeat it.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .region_masks import extract_frame_rgb, skin_mask

METRICS = ("chroma_compatible_fraction", "clipped_fraction", "brightness",
           "spatial_chroma_mad", "finite_fraction", "pixel_count", "centroid_x", "centroid_y")


@dataclass(frozen=True)
class OpticalQualityConfig:
    patch_min_pixels: int = 16
    support_fraction: float = .95
    max_clipped_fraction: float = .05
    min_chroma_fraction: float = .5
    min_eligible_patches: int = 2
    score_floor: float = .5
    block_samples: int = 300

    def __post_init__(self):
        if (type(self.patch_min_pixels) is not int or self.patch_min_pixels < 1
                or type(self.min_eligible_patches) is not int or not 2 <= self.min_eligible_patches <= 6
                or type(self.block_samples) is not int or self.block_samples < 1
                or not np.isfinite([self.support_fraction, self.max_clipped_fraction,
                                    self.min_chroma_fraction, self.score_floor]).all()
                or not 0 < self.support_fraction <= 1 or not 0 <= self.max_clipped_fraction <= 1
                or not 0 <= self.min_chroma_fraction <= 1 or not .5 <= self.score_floor <= 1):
            raise ValueError("Invalid optical engineering configuration")


def optical_measurement(frame_rgb_255: np.ndarray, mask: np.ndarray, min_pixels: int):
    """Measure a generated polygon/patch on ORIGINAL RGB intensities.

    Only the selected finite in-range pixels reach the chroma heuristic. Raw
    frames are never modified. Missing geometry returns unavailable descriptors.
    """
    frame = np.asarray(frame_rgb_255)
    if (frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype.kind != "f"
            or mask.shape != frame.shape[:2] or mask.dtype != bool
            or type(min_pixels) is not int or min_pixels < 1):
        raise ValueError("Expected floating RGB [H,W,3], bool mask and positive pixel count")
    selected = frame[mask]
    finite = np.isfinite(selected).all(-1) & ((selected >= 0) & (selected <= 255)).all(-1)
    pixels = selected[finite]
    metrics = {k: float("nan") for k in METRICS}
    metrics.update(pixel_count=float(len(pixels)), finite_fraction=float(finite.mean()) if len(finite) else 0.)
    yy, xx = np.nonzero(mask)
    if len(xx):
        metrics.update(centroid_x=float(xx.mean()/frame.shape[1]), centroid_y=float(yy.mean()/frame.shape[0]))
    if len(pixels) < min_pixels:
        return np.full(3, np.nan), False, metrics
    compatible = skin_mask(np.rint(pixels).astype(np.uint8).reshape(-1, 1, 3)).reshape(-1)
    chroma = pixels / np.maximum(pixels.sum(-1, keepdims=True), 1e-8)
    metrics.update(chroma_compatible_fraction=float(compatible.mean()),
                   clipped_fraction=float(np.any((pixels <= 1) | (pixels >= 254), axis=-1).mean()),
                   brightness=float(pixels.mean()/255.),
                   spatial_chroma_mad=float(np.median(np.abs(chroma - np.median(chroma, axis=0)).mean(-1))))
    # Reuse the same measurement helper as V1, with explicit finite support.
    usable_mask = mask.copy()
    usable_mask[mask] = finite
    rgb, _ = extract_frame_rgb(frame, usable_mask, min_pixels)
    return rgb, True, metrics


def finite_median(values, default=float("nan")):
    values = np.asarray(values, dtype=float)
    return float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else default


def temporal_descriptors(rgb: np.ndarray, centroids: np.ndarray | None = None) -> dict:
    """Descriptive RGB/motion differences only, never pulse-based rejection."""
    if rgb.ndim != 2 or rgb.shape[-1] != 3:
        raise ValueError("Expected [T,3] RGB")
    pairs = np.isfinite(rgb[1:]).all(-1) & np.isfinite(rgb[:-1]).all(-1)
    first = np.abs(np.diff(rgb, axis=0)).mean(-1)/255.
    result = dict(valid_adjacent_pairs=int(pairs.sum()),
                  median_absolute_rgb_step=finite_median(first[pairs]),
                  maximum_absolute_rgb_step=float(np.max(first[pairs])) if pairs.any() else None,
                  interpretation="descriptive_change_not_physiological_quality")
    triples = np.isfinite(rgb[2:]).all(-1) & np.isfinite(rgb[1:-1]).all(-1) & np.isfinite(rgb[:-2]).all(-1)
    second = np.abs(rgb[2:] - 2*rgb[1:-1] + rgb[:-2]).mean(-1)/255.
    result["median_second_difference_flicker_proxy"] = finite_median(second[triples])
    if centroids is not None:
        if centroids.shape != (len(rgb), 2):
            raise ValueError("Centroids must be [T,2]")
        steps = np.linalg.norm(np.diff(centroids, axis=0), axis=-1)
        result["median_normalized_centroid_step"] = finite_median(steps)
    return result
