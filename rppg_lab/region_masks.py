"""Configurable geometric masks; skin chroma gating remains experimental."""
from __future__ import annotations

from typing import Iterable, Mapping
import numpy as np
from .roi import FACE_ROIS
from .types import RegionObservation

# Face oval with eye and mouth holes. Vertex masks are approximations, not
# semantic segmentation; inspect debug overlays for every acquisition setup.
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
FACE_HOLES = [[33,160,158,133,153,144], [263,387,385,362,380,373], [61,40,37,0,267,270,291,321,314,17,84,91]]
CANONICAL_FACE_ROIS = tuple(FACE_ROIS)
DEFAULT_FACE_ROIS = (*CANONICAL_FACE_ROIS, "combined")
FACE_MASK_STRATEGIES = frozenset((*DEFAULT_FACE_ROIS, "full_skin"))


def _polygon(mask: np.ndarray, points: np.ndarray, fill: int = 1) -> None:
    import cv2
    # Convex hull is deliberate: legacy FACE_ROIS vertices are not perimeter
    # ordered. Do not use a potentially self-intersecting polygon.
    h, w = mask.shape
    pts = np.rint(points).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    if len(np.unique(pts, axis=0)) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), fill)


def skin_mask(frame_rgb: np.ndarray, cr_range=(133,173), cb_range=(77,127)) -> np.ndarray:
    """Experimental YCrCb gate. Thresholds can bias coverage by skin/lighting."""
    import cv2
    ycc = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    return ((ycc[:, :, 1] >= cr_range[0]) & (ycc[:, :, 1] <= cr_range[1]) &
            (ycc[:, :, 2] >= cb_range[0]) & (ycc[:, :, 2] <= cb_range[1]))


def get_face_masks(frame_rgb: np.ndarray, observation: RegionObservation,
                   strategies: Iterable[str] = DEFAULT_FACE_ROIS,
                   use_skin_mask: bool = False, cr_range=(133,173), cb_range=(77,127)) -> dict[str, np.ndarray]:
    """Construct distinct masks from one observation; no detector calls here.

    Combined is the pixel union, including overlaps only once. Geometry uses
    the authoritative FACE_ROIS vertex groups and the existing rounded hulls.
    The optional chroma gate is evaluated once and applied to each mask.
    """
    names = tuple(dict.fromkeys(strategies))
    if not set(names) <= FACE_MASK_STRATEGIES:
        raise ValueError("Unknown face mask strategy")
    masks = {name: np.zeros(frame_rgb.shape[:2], dtype=bool) for name in names}
    if not observation.valid or observation.landmarks is None:
        return masks
    points = observation.landmarks
    components = {}
    for name in CANONICAL_FACE_ROIS:
        if name in names or "combined" in names:
            mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
            _polygon(mask, points[list(FACE_ROIS[name])])
            components[name] = mask.astype(bool)
    for name in names:
        if name == "combined":
            masks[name] = np.logical_or.reduce(list(components.values()))
        elif name == "full_skin":
            mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
            _polygon(mask, points[FACE_OVAL])
            for hole in FACE_HOLES:
                _polygon(mask, points[hole], 0)
            masks[name] = mask.astype(bool)
        else:
            masks[name] = components[name]
    if use_skin_mask or "full_skin" in names:
        gate = skin_mask(frame_rgb, cr_range, cb_range)
        for name in names:
            if use_skin_mask or name == "full_skin":
                masks[name] &= gate
    return masks


def get_region_mask(frame_rgb: np.ndarray, observation: RegionObservation, region: str, strategy: str,
                    use_skin_mask: bool = False, cr_range=(133,173), cb_range=(77,127)) -> np.ndarray:
    if region == "face":
        return get_face_masks(frame_rgb, observation, (strategy,), use_skin_mask, cr_range, cb_range)[strategy]
    mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
    if not observation.valid or observation.landmarks is None:
        return mask.astype(bool)
    points = observation.landmarks
    if region == "hand":
        if strategy in {"palm", "back_of_hand"}:
            # Same geometric central-hand polygon; orientation cannot be inferred
            # reliably here. back_of_hand is a protocol designation, experimental.
            _polygon(mask, points[[0,1,2,5,9,13,17]])
        elif strategy == "landmark_polygon":
            _polygon(mask, points)
        elif strategy in {"bounding_box", "skin_masked_bbox"}:
            if observation.bbox is not None:
                x1,y1,x2,y2 = observation.bbox
                mask[y1:y2, x1:x2] = 1
        else:
            raise ValueError("Unknown hand mask strategy")
    else:
        raise ValueError("region must be face or hand")
    if use_skin_mask or strategy in {"full_skin", "skin_masked_bbox"}:
        mask &= skin_mask(frame_rgb, cr_range, cb_range).astype(np.uint8)
    return mask.astype(bool)


def extract_frame_rgb(frame_rgb: np.ndarray, mask: np.ndarray, min_pixels: int) -> tuple[np.ndarray, dict]:
    """Mean of finite RGB pixels; insufficient support stays NaN, not zero."""
    mask = np.asarray(mask, dtype=bool)
    pixels = frame_rgb[mask]
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    metrics = {"skin_pixel_count": int(len(pixels)), "mask_pixel_count": int(mask.sum()),
               "mask_coverage": float(mask.mean()), "extraction_valid": len(pixels) >= min_pixels,
               "clipping_fraction": float("nan"), "brightness": float("nan")}
    if len(pixels) < min_pixels:
        return np.full(3, np.nan), metrics
    metrics["clipping_fraction"] = float(np.mean(np.any((pixels <= 1) | (pixels >= 254), axis=1)))
    metrics["brightness"] = float(np.mean(pixels) / 255)
    return pixels.mean(axis=0), metrics


def extract_frame_rgb_by_roi(frame_rgb: np.ndarray, masks: Mapping[str, np.ndarray],
                             min_pixels: int) -> dict[str, tuple[np.ndarray, dict]]:
    """Measure all masks against the same decoded frame, independently."""
    return {name: extract_frame_rgb(frame_rgb, mask, min_pixels) for name, mask in masks.items()}
