"""Dataset loaders and PyTorch datasets.

Currently supports UBFC-rPPG style folders:
  subjectXX/
    vid.avi or video.avi
    ground_truth.txt
where ground_truth.txt has rows: PPG, HR, time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset

from .classical import METHOD_FUNCS, DEFAULT_PRIORS
from .roi import extract_rgb_trace
from .signals import resample_uniform, standardize_channels, standardize_1d, estimate_hr_welch


@dataclass(frozen=True)
class UBFCSubject:
    subject_id: str
    video_path: Path
    gt_path: Path


def find_ubfc_subjects(root_dir: str | Path) -> List[UBFCSubject]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    subjects: List[UBFCSubject] = []
    for folder in sorted([p for p in root.iterdir() if p.is_dir()]):
        video = None
        for name in ["vid.avi", "video.avi", "video.mp4", "vid.mp4"]:
            if (folder / name).exists():
                video = folder / name
                break
        gt = folder / "ground_truth.txt"
        if video is not None and gt.exists():
            subjects.append(UBFCSubject(folder.name, video, gt))
    if not subjects:
        raise RuntimeError(f"No UBFC subjects found under {root}")
    return subjects


def load_ubfc_ground_truth(gt_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(gt_path)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        if data.size % 3 != 0:
            raise ValueError(f"Unexpected ground_truth shape {data.shape}")
        data = data.reshape(3, -1)
    if data.shape[0] != 3 and data.shape[1] == 3:
        data = data.T
    if data.shape[0] != 3:
        raise ValueError(f"Expected 3 rows in {gt_path}, got {data.shape}")
    ppg, hr, t = data[0], data[1], data[2]
    return t.astype(np.float64), hr.astype(np.float32), ppg.astype(np.float32)


def sliding_windows(num_samples: int, fs: float, win_sec: float, stride_sec: float) -> List[Tuple[int, int]]:
    win = int(round(win_sec * fs))
    stride = max(1, int(round(stride_sec * fs)))
    out = []
    s = 0
    while s + win <= num_samples:
        out.append((s, s + win))
        s += stride
    return out


def subject_split(subjects: Sequence[UBFCSubject], val_fraction: float = 0.2, seed: int = 42) -> Tuple[List[UBFCSubject], List[UBFCSubject]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(subjects))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(subjects) * val_fraction)))
    val_idx = set(idx[:n_val].tolist())
    train, val = [], []
    for i, s in enumerate(subjects):
        (val if i in val_idx else train).append(s)
    return train, val


class UBFCRPPGDataset(Dataset):
    """UBFC waveform dataset with optional classical-prior channels.

    Each item returns a dict:
      x: (C, L), y_ppg: (L,), y_hr: scalar, subject_id: str
    """

    def __init__(
        self,
        subjects: Sequence[UBFCSubject],
        fs_target: float = 30.0,
        win_sec: float = 8.0,
        stride_sec: float = 1.0,
        roi: str = "face",
        cache_dir: str | Path = "cache_roi",
        prior_methods: Optional[Sequence[str]] = DEFAULT_PRIORS,
        min_quality: float = 0.0,
        max_frames: Optional[int] = None,
    ) -> None:
        self.subjects = list(subjects)
        self.fs_target = float(fs_target)
        self.win_sec = float(win_sec)
        self.stride_sec = float(stride_sec)
        self.roi = roi
        self.cache_dir = Path(cache_dir)
        self.prior_methods = [] if prior_methods is None else list(prior_methods)
        self.min_quality = float(min_quality)
        self.max_frames = max_frames
        self.records = []
        self.samples: List[Tuple[int, int, int]] = []
        self._load_all()

    def _load_all(self) -> None:
        for rec_idx, subj in enumerate(self.subjects):
            rgb, fs, t_vid, q = extract_rgb_trace(
                subj.video_path,
                roi=self.roi,
                cache_dir=self.cache_dir,
                fs_target=self.fs_target,
                min_quality=self.min_quality,
                max_frames=self.max_frames,
            )
            t_gt, hr_gt, ppg_gt = load_ubfc_ground_truth(subj.gt_path)
            # Align GT to video time grid. Use the common support.
            t0 = max(float(t_vid[0]), float(t_gt[0]))
            t1 = min(float(t_vid[-1]), float(t_gt[-1]))
            keep = (t_vid >= t0) & (t_vid <= t1)
            t_vid2 = t_vid[keep]
            rgb = rgb[keep]
            q = q[keep]
            _, ppg_u = resample_uniform(t_gt, ppg_gt, self.fs_target, t_start=t0, t_end=t1)
            _, hr_u = resample_uniform(t_gt, hr_gt, self.fs_target, t_start=t0, t_end=t1)
            n = min(len(rgb), len(ppg_u), len(hr_u))
            rgb, q, ppg_u, hr_u, t_vid2 = rgb[:n], q[:n], ppg_u[:n], hr_u[:n], t_vid2[:n]

            priors = []
            for name in self.prior_methods:
                if name not in METHOD_FUNCS:
                    raise ValueError(f"Unknown prior method {name}. Available: {sorted(METHOD_FUNCS)}")
                try:
                    priors.append(METHOD_FUNCS[name](rgb, self.fs_target).astype(np.float32))
                except Exception as exc:
                    warnings.warn(
                        f"Prior {name} failed for {subj.subject_id}; substituting zeros: {exc}",
                        RuntimeWarning,
                    )
                    priors.append(np.zeros(n, dtype=np.float32))
            self.records.append({
                "subject_id": subj.subject_id,
                "rgb": rgb.astype(np.float32),
                "quality": q.astype(np.float32),
                "ppg": np.asarray(ppg_u, dtype=np.float32),
                "hr": np.asarray(hr_u, dtype=np.float32),
                "priors": priors,
                "timestamps": t_vid2.astype(np.float64),
            })
            for s, e in sliding_windows(n, self.fs_target, self.win_sec, self.stride_sec):
                self.samples.append((rec_idx, s, e))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        rec_idx, s, e = self.samples[idx]
        rec = self.records[rec_idx]
        rgb = standardize_channels(rec["rgb"][s:e])  # (L, 3)
        channels = [rgb.T]
        for p in rec["priors"]:
            channels.append(standardize_1d(p[s:e])[None, :])
        x = np.concatenate(channels, axis=0).astype(np.float32)
        ppg_win_raw = np.asarray(rec["ppg"][s:e], dtype=np.float32)
        y = standardize_1d(ppg_win_raw)

        # For window-level training/evaluation, define HR consistently from
        # the same reference PPG window used as waveform target.
        hr_est = estimate_hr_welch(ppg_win_raw, self.fs_target)
        hr = float(hr_est.hr_bpm)

        # Fallback only if Welch fails.
        if not np.isfinite(hr):
            hr = float(np.nanmean(rec["hr"][s:e]))

        q = float(np.nanmean(rec["quality"][s:e]))
        return {
            "x": torch.from_numpy(x),
            "y_ppg": torch.from_numpy(y.astype(np.float32)),
            "y_hr": torch.tensor(hr, dtype=torch.float32),
            "quality": torch.tensor(q, dtype=torch.float32),
            "subject_id": rec["subject_id"],
        }


class SessionBPDataset(Dataset):
    """Feature-level BP dataset for the group's future recordings.

    Expected folder layout::

      data_sessions/S001/session_name/
        features.json
        labels.json

    Each item includes ``subject_id`` so train/validation/test partitions can be
    formed at the participant level. Missing numeric features are represented as
    NaN in ``x_raw`` and are imputed only after the split using training-set
    statistics.
    """

    def __init__(self, data_dir: str | Path, feature_keys: Optional[Sequence[str]] = None, target: str = "both"):
        self.data_dir = Path(data_dir)
        self.target = target
        self.rows: List[Dict] = []
        for feat_path in sorted(self.data_dir.glob("*/*/features.json")):
            label_path = feat_path.with_name("labels.json")
            if not label_path.exists():
                continue
            feat = json.loads(feat_path.read_text(encoding="utf-8"))
            lab = json.loads(label_path.read_text(encoding="utf-8"))
            cuff = lab.get("cuff", {})
            if "sbp" not in cuff or "dbp" not in cuff:
                continue
            subject_id = str(lab.get("subject_id") or feat_path.parent.parent.name)
            session_id = str(lab.get("session_id") or feat_path.parent.name)
            self.rows.append({
                "features": feat,
                "labels": lab,
                "sbp": float(cuff["sbp"]),
                "dbp": float(cuff["dbp"]),
                "subject_id": subject_id,
                "session_id": session_id,
            })
        if not self.rows:
            raise RuntimeError(f"No feature/label rows found in {self.data_dir}")
        if feature_keys is None:
            keys = sorted({
                k
                for row in self.rows
                for k, v in row["features"].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            })
            self.feature_keys = keys
        else:
            self.feature_keys = list(feature_keys)
        if not self.feature_keys:
            raise RuntimeError("No numeric BP feature keys were found.")

    @property
    def subject_ids(self) -> List[str]:
        return [str(row["subject_id"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        feat = row["features"]
        x = np.array([float(feat.get(k, np.nan)) for k in self.feature_keys], dtype=np.float32)
        if self.target == "sbp":
            y = np.array([row["sbp"]], dtype=np.float32)
        elif self.target == "dbp":
            y = np.array([row["dbp"]], dtype=np.float32)
        else:
            y = np.array([row["sbp"], row["dbp"]], dtype=np.float32)
        return {
            "x_raw": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "subject_id": str(row["subject_id"]),
            "session_id": str(row["session_id"]),
        }
