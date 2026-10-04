"""Video-clip datasets for spatio-temporal HR models.

This module intentionally starts with UBFC video clips.  The previous
HRQualityNet path used 1D traces and classical priors; this path uses raw face
video clips so that the model can learn spatial/color/motion cues that are lost
when we average an ROI into a single RGB trace.

Design choices for the first video model:
- subject-wise split is still handled outside with ``find_ubfc_subjects``;
- clips are sampled in seconds and then uniformly resampled to a fixed number of
  frames, so OpenCV FPS differences do not change tensor length;
- a cached crop box is used per subject to avoid running face detection on every
  sample;
- if MediaPipe is unavailable or detection fails, a conservative center crop is
  used, so training can still run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import UBFCSubject, load_ubfc_ground_truth
from .signals import estimate_hr_welch, resample_uniform


@dataclass(frozen=True)
class VideoWindow:
    rec_idx: int
    start_sec: float
    end_sec: float


def _sliding_time_windows(duration_sec: float, win_sec: float, stride_sec: float) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    s = 0.0
    eps = 1e-6
    while s + win_sec <= duration_sec + eps:
        out.append((float(s), float(s + win_sec)))
        s += stride_sec
    return out


def _safe_video_info(video_path: str | Path) -> Tuple[float, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 1e-6:
        fps = 30.0
    duration = n_frames / fps if n_frames > 0 else 0.0
    cap.release()
    if n_frames <= 0 or duration <= 0:
        raise RuntimeError(f"Empty or invalid video: {video_path}")
    return fps, n_frames, duration


def _center_square_box(width: int, height: int, scale: float = 0.82) -> Tuple[int, int, int, int]:
    side = int(round(min(width, height) * scale))
    x1 = max(0, (width - side) // 2)
    y1 = max(0, (height - side) // 2)
    return x1, y1, min(width, x1 + side), min(height, y1 + side)


def _expand_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int, margin: float = 0.22) -> Tuple[int, int, int, int]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(bw, bh) * (1.0 + margin)
    nx1 = int(round(cx - side / 2))
    ny1 = int(round(cy - side / 2))
    nx2 = int(round(cx + side / 2))
    ny2 = int(round(cy + side / 2))
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(width, nx2)
    ny2 = min(height, ny2)
    if nx2 <= nx1 or ny2 <= ny1:
        return _center_square_box(width, height)
    return nx1, ny1, nx2, ny2


def detect_face_crop_box(video_path: str | Path, cache_path: Optional[str | Path] = None, sample_frames: int = 8) -> Tuple[int, int, int, int]:
    """Detect one stable face crop box for a video.

    Uses MediaPipe face detection if available.  If it is unavailable or no face
    is found, falls back to a center square crop.  The result is cached as JSON.
    """
    video_path = Path(video_path)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return int(data["x1"]), int(data["y1"]), int(data["x2"]), int(data["y2"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions: {video_path}")

    fallback = _center_square_box(width, height)
    boxes: List[Tuple[int, int, int, int]] = []

    try:
        import mediapipe as mp  # type: ignore
        detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.45)
        frame_ids = np.linspace(0, max(0, n_frames - 1), num=max(1, sample_frames), dtype=int)
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = detector.process(rgb)
            if not res.detections:
                continue
            det = max(res.detections, key=lambda d: float(d.score[0]) if d.score else 0.0)
            bb = det.location_data.relative_bounding_box
            x1 = int(round(bb.xmin * width))
            y1 = int(round(bb.ymin * height))
            x2 = int(round((bb.xmin + bb.width) * width))
            y2 = int(round((bb.ymin + bb.height) * height))
            boxes.append(_expand_box(x1, y1, x2, y2, width, height))
        detector.close()
    except Exception:
        boxes = []

    cap.release()
    if boxes:
        arr = np.asarray(boxes, dtype=np.float32)
        med = np.median(arr, axis=0).round().astype(int).tolist()
        box = tuple(med)  # type: ignore[assignment]
    else:
        box = fallback

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        x1, y1, x2, y2 = box
        cache_path.write_text(json.dumps({"x1": x1, "y1": y1, "x2": x2, "y2": y2}, indent=2), encoding="utf-8")
    return box


def read_video_clip_uniform(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    num_frames: int,
    crop_box: Tuple[int, int, int, int],
    image_size: int = 64,
) -> np.ndarray:
    """Read a fixed-length RGB video clip.

    Returns float32 array with shape (T, 3, H, W), values normalized per clip.
    """
    fps, n_frames, _ = _safe_video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    x1, y1, x2, y2 = crop_box
    times = np.linspace(float(start_sec), float(end_sec), num=int(num_frames), endpoint=False)
    frame_ids = np.clip(np.round(times * fps).astype(int), 0, max(0, n_frames - 1))
    frames: List[np.ndarray] = []
    last: Optional[np.ndarray] = None
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, frame = cap.read()
        if not ok or frame is None:
            if last is None:
                frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                rgb = frame
            else:
                rgb = last
        else:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = frame
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
            last = rgb
        frames.append(rgb)
    cap.release()
    clip = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T,H,W,3)
    # Per-clip, per-channel standardization.  This removes static skin tone and
    # illumination level, forcing the model to use temporal color variation.
    mean = clip.mean(axis=(0, 1, 2), keepdims=True)
    std = clip.std(axis=(0, 1, 2), keepdims=True) + 1e-6
    clip = (clip - mean) / std
    clip = np.transpose(clip, (0, 3, 1, 2))  # (T,3,H,W)
    return clip.astype(np.float32)


class UBFCVideoHRDataset(Dataset):
    """UBFC subject-wise video-clip dataset for direct HR training."""

    def __init__(
        self,
        subjects: Sequence[UBFCSubject],
        clip_sec: float = 10.0,
        stride_sec: float = 2.0,
        clip_frames: int = 160,
        image_size: int = 64,
        fs_target: float = 30.0,
        cache_dir: str | Path = "cache_video",
        min_hr: float = 40.0,
        max_hr: float = 180.0,
    ) -> None:
        self.subjects = list(subjects)
        self.clip_sec = float(clip_sec)
        self.stride_sec = float(stride_sec)
        self.clip_frames = int(clip_frames)
        self.image_size = int(image_size)
        self.fs_target = float(fs_target)
        self.cache_dir = Path(cache_dir)
        self.min_hr = float(min_hr)
        self.max_hr = float(max_hr)

        self.records: List[Dict] = []
        self.windows: List[VideoWindow] = []
        self._load_metadata()

    def _load_metadata(self) -> None:
        for rec_idx, subj in enumerate(self.subjects):
            fps, n_frames, duration_vid = _safe_video_info(subj.video_path)
            t_gt, hr_gt, ppg_gt = load_ubfc_ground_truth(subj.gt_path)
            duration = min(duration_vid, float(t_gt[-1] - t_gt[0]))
            if duration < self.clip_sec:
                continue
            box_path = self.cache_dir / "boxes" / f"{subj.subject_id}_face_box.json"
            box = detect_face_crop_box(subj.video_path, cache_path=box_path)
            rec = {
                "subject_id": subj.subject_id,
                "video_path": subj.video_path,
                "fps": fps,
                "n_frames": n_frames,
                "duration": duration,
                "crop_box": box,
                "t_gt": t_gt,
                "hr_gt": hr_gt,
                "ppg_gt": ppg_gt,
            }
            self.records.append(rec)
            for s, e in _sliding_time_windows(duration, self.clip_sec, self.stride_sec):
                self.windows.append(VideoWindow(len(self.records) - 1, s, e))

    def __len__(self) -> int:
        return len(self.windows)

    def _window_hr(self, rec: Dict, start_sec: float, end_sec: float) -> Tuple[float, float, float]:
        t_gt = rec["t_gt"]
        ppg = rec["ppg_gt"]
        hr = rec["hr_gt"]
        t_start = float(t_gt[0]) + start_sec
        t_end = float(t_gt[0]) + end_sec
        _, ppg_u = resample_uniform(t_gt, ppg, self.fs_target, t_start=t_start, t_end=t_end)
        est = estimate_hr_welch(ppg_u, self.fs_target, fmin=0.67, fmax=3.2)
        hr_bpm = float(est.hr_bpm)
        if not np.isfinite(hr_bpm) or hr_bpm < self.min_hr or hr_bpm > self.max_hr:
            m = (t_gt >= t_start) & (t_gt <= t_end)
            hr_bpm = float(np.nanmean(hr[m])) if np.any(m) else float("nan")
        snr = float(est.snr_db) if np.isfinite(est.snr_db) else -20.0
        # A simple supervised quality target for now.  This is intentionally
        # conservative; for video-only training we mainly use it as a ranking aid.
        quality = 1.0 if (np.isfinite(hr_bpm) and snr >= 0.0) else 0.0
        return hr_bpm, quality, snr

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        win = self.windows[idx]
        rec = self.records[win.rec_idx]
        clip = read_video_clip_uniform(
            rec["video_path"],
            start_sec=win.start_sec,
            end_sec=win.end_sec,
            num_frames=self.clip_frames,
            crop_box=rec["crop_box"],
            image_size=self.image_size,
        )
        hr_bpm, quality, snr = self._window_hr(rec, win.start_sec, win.end_sec)
        return {
            "video": torch.from_numpy(clip),  # (T,3,H,W)
            "y_hr": torch.tensor(hr_bpm, dtype=torch.float32),
            "quality": torch.tensor(float(quality), dtype=torch.float32),
            "gt_snr_db": torch.tensor(float(snr), dtype=torch.float32),
            "subject_id": rec["subject_id"],
            "start_sec": torch.tensor(float(win.start_sec), dtype=torch.float32),
        }
