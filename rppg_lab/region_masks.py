"""Configurable geometric masks; skin chroma gating remains experimental."""
from __future__ import annotations

import numpy as np
from .roi import FACE_ROIS
from .types import RegionObservation

# Face oval with eye and mouth holes. Vertex masks are approximations, not
# semantic segmentation; inspect debug overlays for every acquisition setup.
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
FACE_HOLES = [[33,160,158,133,153,144], [263,387,385,362,380,373], [61,40,37,0,267,270,291,321,314,17,84,91]]


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


def get_region_mask(frame_rgb: np.ndarray, observation: RegionObservation, region: str, strategy: str,
                    use_skin_mask: bool = False, cr_range=(133,173), cb_range=(77,127)) -> np.ndarray:
    mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
    if not observation.valid or observation.landmarks is None:
        return mask.astype(bool)
    points = observation.landmarks
    if region == "face":
        if strategy == "full_skin":
            _polygon(mask, points[FACE_OVAL])
            for hole in FACE_HOLES:
                _polygon(mask, points[hole], 0)
        elif strategy == "combined":
            for indices in FACE_ROIS.values():
                _polygon(mask, points[list(indices)])
        elif strategy in FACE_ROIS:
            _polygon(mask, points[list(FACE_ROIS[strategy])])
        else:
            raise ValueError("Unknown face mask strategy")
    elif region == "hand":
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
    pixels = frame_rgb[mask]
    metrics = {"skin_pixel_count": int(len(pixels)), "clipping_fraction": float("nan"), "brightness": float("nan")}
    if len(pixels) < min_pixels:
        return np.full(3, np.nan), metrics
    metrics["clipping_fraction"] = float(np.mean(np.any((pixels <= 1) | (pixels >= 254), axis=1)))
    metrics["brightness"] = float(np.mean(pixels) / 255)
    return pixels.mean(axis=0), metrics
