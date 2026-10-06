"""Video ROI extraction for rPPG experiments.

The extractor uses MediaPipe FaceMesh to produce per-frame mean RGB traces for
multiple ROIs. It records real frame timestamps where possible; this matters for
future high-FPS BP/PTT experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import hashlib
import json

import numpy as np


@dataclass
class ROIExtractionResult:
    timestamps: np.ndarray
    fs_reported: float
    roi_rgb: Dict[str, np.ndarray]
    quality: np.ndarray
    frame_shape: Tuple[int, int]

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamps": self.timestamps,
            "fs_reported": np.array([self.fs_reported], dtype=np.float32),
            "quality": self.quality,
            "frame_shape": np.array(self.frame_shape, dtype=np.int32),
            "roi_names": np.array(list(self.roi_rgb.keys())),
        }
        for k, v in self.roi_rgb.items():
            payload[f"roi__{k}"] = v
        np.savez_compressed(path, **payload)

    @staticmethod
    def load_npz(path: str | Path) -> "ROIExtractionResult":
        with np.load(path, allow_pickle=False) as data:
            roi_names = [str(x) for x in data["roi_names"].tolist()]
            roi_rgb = {name: data[f"roi__{name}"] for name in roi_names}
            return ROIExtractionResult(
                timestamps=data["timestamps"],
                fs_reported=float(data["fs_reported"][0]),
                roi_rgb=roi_rgb,
                quality=data["quality"],
                frame_shape=tuple(int(x) for x in data["frame_shape"]),
            )


# MediaPipe FaceMesh landmark groups. These are practical polygon ROIs, not
# anatomical claims. We keep them explicit for reproducibility.
FACE_ROIS: Dict[str, Iterable[int]] = {
    "forehead": [107, 66, 69, 109, 10, 338, 299, 296, 336, 9],
    "left_cheek": [118, 119, 100, 126, 209, 49, 129, 203, 205, 50],
    "right_cheek": [347, 348, 329, 355, 429, 279, 358, 423, 425, 280],
}


def _cache_key(video_path: str | Path, config: dict) -> str:
    p = Path(video_path)
    stat = p.stat()
    payload = json.dumps(
        {"path": str(p.resolve()), "mtime": stat.st_mtime, "size": stat.st_size, **config},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _polygon_mean_rgb(frame_rgb: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, int]:
    import cv2

    h, w = frame_rgb.shape[:2]
    pts = np.asarray(pts, dtype=np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    pixels = frame_rgb[mask > 0]
    if pixels.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32), 0
    return pixels.mean(axis=0).astype(np.float32), int(pixels.shape[0])


def _quality_from_frame(frame_rgb: np.ndarray, face_pts: np.ndarray, roi_pixels: int, prev_center: Optional[np.ndarray]) -> Tuple[float, np.ndarray]:
    import cv2

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray) / 255.0)
    # Penalize under/over exposure.
    brightness_q = float(np.clip(1.0 - abs(brightness - 0.50) / 0.50, 0.0, 1.0))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp_q = float(np.clip(sharpness / 80.0, 0.0, 1.0))
    h, w = gray.shape
    area_q = float(np.clip(roi_pixels / max(1.0, 0.015 * h * w), 0.0, 1.0))
    center = np.mean(face_pts, axis=0)
    if prev_center is None:
        motion_q = 1.0
    else:
        motion_px = float(np.linalg.norm(center - prev_center))
        motion_q = float(np.clip(1.0 - motion_px / 20.0, 0.0, 1.0))
    q = 0.35 * brightness_q + 0.25 * sharp_q + 0.25 * area_q + 0.15 * motion_q
    return float(np.clip(q, 0.0, 1.0)), center


class FaceROIExtractor:
    """Extract multi-ROI RGB traces from a face video.

    Parameters
    ----------
    min_detection_confidence, min_tracking_confidence:
        MediaPipe FaceMesh settings.
    include_full_face:
        Adds a simple full-face ROI as the average of forehead and cheeks. It is
        not a segmentation mask; it is a stable aggregated trace for baselines.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        include_full_face: bool = True,
        detector_backend: str = "auto",
        model_path: str | Path = "models/face_landmarker.task",
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.include_full_face = include_full_face
        self.detector_backend = detector_backend
        self.model_path = Path(model_path)

    def extract(
        self,
        video_path: str | Path,
        max_frames: Optional[int] = None,
        cache_dir: Optional[str | Path] = None,
        force: bool = False,
    ) -> ROIExtractionResult:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        config = {
            "max_frames": max_frames,
            "min_det": self.min_detection_confidence,
            "min_track": self.min_tracking_confidence,
            "full": self.include_full_face,
            "detector_backend": self.detector_backend,
            "extraction_version": 2,
            "model_path": str(self.model_path.resolve()),
            "model_mtime_ns": self.model_path.stat().st_mtime_ns if self.model_path.exists() else None,
        }
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"roi_{_cache_key(video_path, config)}.npz"
            if cache_path.exists() and not force:
                return ROIExtractionResult.load_npz(cache_path)
        else:
            cache_path = None

        from .video import VideoReader
        from .detection import FaceRegionDetector
        reader = VideoReader(video_path)
        fs_reported = reader.metadata.fps_nominal
        try:
            face_mesh = FaceRegionDetector(
                backend=self.detector_backend,
                model_path=self.model_path,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        except Exception:
            reader.close()
            raise

        timestamps = []
        roi_values = {name: [] for name in FACE_ROIS}
        qualities = []
        prev_center = None
        frame_shape = (0, 0)
        try:
            for frame_idx, t, frame_rgb in reader.frames(max_frames):
                h, w = frame_rgb.shape[:2]
                frame_shape = (h, w)
                res = face_mesh.detect(frame_rgb, frame_idx, t)
                if not res.valid:
                    continue
                landmarks = res.landmarks
                roi_pixels_total = 0
                this_roi = {}
                all_pts = []
                for name, idxs in FACE_ROIS.items():
                    pts = np.asarray(landmarks[list(idxs)], dtype=np.float32)
                    all_pts.append(pts)
                    mean_rgb, n_pix = _polygon_mean_rgb(frame_rgb, pts)
                    this_roi[name] = mean_rgb
                    roi_pixels_total += n_pix
                face_pts = np.vstack(all_pts)
                q, prev_center = _quality_from_frame(frame_rgb, face_pts, roi_pixels_total, prev_center)
                timestamps.append(t)
                qualities.append(q)
                for name in FACE_ROIS:
                    roi_values[name].append(this_roi[name])
        finally:
            face_mesh.close()
            reader.close()

        if not timestamps:
            raise RuntimeError(f"No usable face frames found in {video_path}")
        roi_rgb = {name: np.asarray(vals, dtype=np.float32) for name, vals in roi_values.items()}
        if self.include_full_face:
            roi_rgb["face"] = np.nanmean(np.stack([roi_rgb[k] for k in FACE_ROIS], axis=0), axis=0).astype(np.float32)
        result = ROIExtractionResult(
            timestamps=np.asarray(timestamps, dtype=np.float64),
            fs_reported=fs_reported,
            roi_rgb=roi_rgb,
            quality=np.asarray(qualities, dtype=np.float32),
            frame_shape=frame_shape,
        )
        if cache_path is not None:
            result.save_npz(cache_path)
        return result


def extract_rgb_trace(
    video_path: str | Path,
    roi: str = "face",
    cache_dir: Optional[str | Path] = None,
    fs_target: Optional[float] = 30.0,
    min_quality: float = 0.0,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Convenience function returning one uniformly sampled ROI RGB trace.

    Returns rgb, fs, timestamps, quality. If fs_target is None, the original
    usable-frame timestamps are returned and fs is the reported video FPS.
    """
    from .signals import resample_uniform

    result = FaceROIExtractor().extract(video_path, max_frames=max_frames, cache_dir=cache_dir)
    if roi not in result.roi_rgb:
        raise ValueError(f"Unknown ROI {roi!r}. Available: {sorted(result.roi_rgb)}")
    t = result.timestamps
    rgb = result.roi_rgb[roi]
    q = result.quality
    keep = q >= float(min_quality)
    if np.sum(keep) < 4:
        keep = np.ones_like(q, dtype=bool)
    t, rgb, q = t[keep], rgb[keep], q[keep]
    if fs_target is None:
        fs = result.fs_reported if result.fs_reported > 0 else 1.0 / np.median(np.diff(t))
        return rgb, float(fs), t, q
    tu, rgb_u = resample_uniform(t, rgb, fs_target)
    _, q_u = resample_uniform(t, q, fs_target, t_start=float(tu[0]), t_end=float(tu[-1]) + 0.5 / fs_target)
    return rgb_u.astype(np.float32), float(fs_target), tu, q_u.astype(np.float32)
