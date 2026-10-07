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
from .config import PipelineConfig
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
    if fs <= 0 or win_sec <= 0 or stride_sec <= 0:
        raise ValueError("Sampling rate and window/stride duration must be positive")
    win = int(round(win_sec * fs))
    if win < 1:
        raise ValueError("Window must contain at least one sample")
    stride = max(1, int(round(stride_sec * fs)))
    out = []
    s = 0
    while s + win <= num_samples:
        out.append((s, s + win))
        s += stride
    return out


def subject_split(subjects: Sequence[UBFCSubject], val_fraction: float = 0.2, seed: int = 42) -> Tuple[List[UBFCSubject], List[UBFCSubject]]:
    ids = [s.subject_id for s in subjects]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate subject IDs; group recordings before partitioning")
    if len(subjects) < 2 or not 0 < val_fraction < 1:
        raise ValueError("Need at least two subjects and validation fraction in (0,1)")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(subjects))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(subjects) * val_fraction)))
    n_val = min(n_val, len(subjects)-1)
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
            from .types import validate_timestamps
            validate_timestamps(t_gt)
            # The target must be evaluated at the actual retained video grid,
            # not a new grid starting at an arbitrary common-support boundary.
            ppg_u = np.interp(t_vid2, t_gt, ppg_gt)
            hr_u = np.interp(t_vid2, t_gt, hr_gt)
            n = len(t_vid2)
            if n < 4:
                raise ValueError(f"Insufficient overlapping video/reference support for {subj.subject_id}")

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


def _prepare_multi_roi_reference(t: np.ndarray, hr: np.ndarray, ppg: np.ndarray
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Keep contact time order; average observations at identical timestamps.

    Some local UBFC reference files contain repeated rounded timestamps. This
    new-dataset-only rule makes interpolation unambiguous without changing the
    video clock. Backward/nonfinite timestamps raise; no sorting or dropping of
    nonfinite signal values occurs. A nonfinite duplicate poisons its group so
    affected target windows still get excluded.
    """
    t = np.asarray(t, dtype=np.float64)
    hr, ppg = np.asarray(hr), np.asarray(ppg)
    if t.ndim != 1 or t.shape != hr.shape or t.shape != ppg.shape or len(t) < 2:
        raise ValueError("Contact reference requires matching [T] arrays with T >= 2")
    if not np.isfinite(t).all() or np.any(np.diff(t) < 0):
        raise ValueError("Contact reference timestamps must be finite and nondecreasing; will not reorder them")
    starts = np.r_[0, np.flatnonzero(np.diff(t) > 0) + 1]
    removed = len(t) - len(starts)
    if removed:
        counts = np.diff(np.r_[starts, len(t)])
        hr = np.add.reduceat(hr.astype(np.float64), starts) / counts
        ppg = np.add.reduceat(ppg.astype(np.float64), starts) / counts
        t = t[starts]
    if len(t) < 2:
        raise ValueError("Contact reference needs at least two distinct timestamps")
    return t, hr, ppg, removed


class UBFCMultiROIRPPGDataset(Dataset):
    """Face-only Phase 1 traces with strictly window-local classical priors.

    Ordering is stored in roi_names, prior_names and channel_names. Each item
    contains x [R,3+K,L], y_ppg [L], scalar y_hr, bool roi_valid [R],
    roi_quality [R], bool prior_valid [R,K], subject_id and start_sec.
    Quality is exactly the source-frame valid fraction in [start,end), without
    reference information. A window survives when at least one ROI has enough
    source support, finite bounded-interpolated RGB and a successful prior.
    """

    def __init__(
        self,
        subjects: Sequence[UBFCSubject],
        fs_target: float = 30.0,
        win_sec: float = 10.0,
        stride_sec: float = 2.0,
        roi_names: Sequence[str] = ("forehead", "left_cheek", "right_cheek"),
        prior_methods: Sequence[str] = tuple(DEFAULT_PRIORS),
        cache_dir: str | Path | None = "cache_roi_multi_phase1",
        extraction_config: PipelineConfig | None = None,
        min_valid_fraction: float | None = None,
        max_frames: int | None = None,
    ) -> None:
        from dataclasses import replace
        from .region_masks import FACE_MASK_STRATEGIES
        if isinstance(roi_names, str) or not roi_names or len(set(roi_names)) != len(roi_names) or not set(roi_names) <= FACE_MASK_STRATEGIES:
            raise ValueError("roi_names must be nonempty, unique facial ROI names")
        if isinstance(prior_methods, str) or not prior_methods or len(set(prior_methods)) != len(prior_methods) or not set(prior_methods) <= METHOD_FUNCS.keys():
            raise ValueError("prior_methods must be nonempty, unique classical method names")
        if not np.isfinite([fs_target, win_sec, stride_sec]).all() or min(fs_target, win_sec, stride_sec) <= 0:
            raise ValueError("fs_target, win_sec and stride_sec must be positive and finite")
        if round(win_sec * fs_target) < 4:
            raise ValueError("A training window must contain at least four samples")
        if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
            raise ValueError("max_frames must be a positive integer or None")
        self.subjects = list(subjects)
        if len({s.subject_id for s in self.subjects}) != len(self.subjects):
            raise ValueError("Duplicate subject IDs")
        self.fs_target = float(fs_target)
        self.win_sec = float(win_sec)
        self.stride_sec = float(stride_sec)
        self.roi_names = tuple(roi_names)
        self.prior_names = tuple(prior_methods)
        self.channel_names = ("RGB_R", "RGB_G", "RGB_B", *self.prior_names)
        config = extraction_config or PipelineConfig(sample_rate=self.fs_target)
        if config.sample_rate != self.fs_target:
            raise ValueError("extraction_config.sample_rate must equal fs_target")
        self.extraction_config = replace(config, regions=("face",), face_rois=self.roi_names,
                                         face_roi=self.roi_names[0])
        self.min_valid_fraction = float(config.min_valid_fraction if min_valid_fraction is None else min_valid_fraction)
        if not np.isfinite(self.min_valid_fraction) or not 0 <= self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must lie in [0,1]")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.max_frames = max_frames
        self.records: list[dict] = []
        self.samples: list[tuple[int, int, int]] = []
        self.exclusions: list[dict] = []
        self.reference_diagnostics: list[dict] = []
        self._load_all()

    def _load_all(self) -> None:
        from .extraction_cache import load_face_roi_extraction
        from .processing import resample_trace
        from .window_priors import build_window_priors
        for subject in self.subjects:
            extraction = load_face_roi_extraction(subject.video_path, self.extraction_config,
                                                   self.cache_dir, self.max_frames)
            grid = extraction.shared.timestamps
            t_gt, hr_gt, ppg_gt = load_ubfc_ground_truth(subject.gt_path)
            t_gt, hr_gt, ppg_gt, duplicates = _prepare_multi_roi_reference(t_gt, hr_gt, ppg_gt)
            self.reference_diagnostics.append(dict(subject_id=subject.subject_id,
                duplicate_timestamp_samples_averaged=duplicates,
                duplicate_policy="mean_at_identical_contact_times; video_clock_unchanged"))
            if duplicates:
                warnings.warn(f"{subject.subject_id}: averaging {duplicates} duplicate contact-reference "
                              "timestamp samples; video timestamps are unchanged", RuntimeWarning)
            keep = (grid >= t_gt[0]) & (grid <= t_gt[-1])
            retained = grid[keep]  # preserve the VIDEO grid's exact origin/spacing
            uniform, imputed = [], []
            for name in self.roi_names:
                rgb, flags = resample_trace(extraction.traces[name], grid, self.extraction_config.max_gap_sec)
                uniform.append(rgb[keep])
                imputed.append(flags[keep])
            rec = dict(subject_id=subject.subject_id, timestamps=retained,
                       rgb=np.stack(uniform), interpolated=np.stack(imputed),
                       traces=extraction.traces, ppg=np.interp(retained, t_gt, ppg_gt),
                       hr=np.interp(retained, t_gt, hr_gt))
            rec_idx = len(self.records)
            self.records.append(rec)
            if len(retained) < round(self.win_sec * self.fs_target):
                self.exclusions.append(dict(subject_id=subject.subject_id, reason="insufficient_common_support"))
            for start, end in sliding_windows(len(retained), self.fs_target, self.win_sec, self.stride_sec):
                target = rec["ppg"][start:end]
                if not np.isfinite(target).all() or np.std(target) <= 1e-8:
                    self.exclusions.append(dict(subject_id=subject.subject_id,
                        start_sec=float(retained[start]), reason="invalid_reference_window"))
                    continue
                valid, _ = self._window_support(rec, start, end)
                # Screen for at least one successful prior, never recording-wide.
                # Usually GREEN succeeds immediately; full priors are computed
                # only for __getitem__, with no global or on-disk prior cache.
                for r in np.flatnonzero(valid):
                    rgb_window = rec["rgb"][r, start:end]
                    valid[r] = any(build_window_priors(rgb_window, self.fs_target, (name,))[1][0]
                                   for name in self.prior_names)
                if valid.any():
                    self.samples.append((rec_idx, start, end))
                else:
                    self.exclusions.append(dict(subject_id=subject.subject_id,
                        start_sec=float(retained[start]), reason="no_usable_roi"))

    def _window_support(self, rec: dict, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        quality = np.zeros(len(self.roi_names), dtype=np.float32)
        valid = np.zeros(len(self.roi_names), dtype=bool)
        a = rec["timestamps"][start]
        b = rec["timestamps"][end - 1] + 1 / self.fs_target
        for r, name in enumerate(self.roi_names):
            trace = rec["traces"][name]
            source = (trace.timestamps >= a) & (trace.timestamps < b)
            fraction = float(trace.validity_mask[source].mean()) if source.any() else 0.0
            quality[r] = fraction
            valid[r] = bool(source.any() and fraction >= self.min_valid_fraction
                            and np.isfinite(rec["rgb"][r, start:end]).all())
        return valid, quality

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        from .window_priors import build_window_priors
        rec_idx, start, end = self.samples[idx]
        rec = self.records[rec_idx]
        valid, quality = self._window_support(rec, start, end)
        prior_valid = np.zeros((len(self.roi_names), len(self.prior_names)), dtype=bool)
        x = np.zeros((len(self.roi_names), len(self.channel_names), end - start), dtype=np.float32)
        for r in np.flatnonzero(valid):
            rgb_window = rec["rgb"][r, start:end]
            priors, success = build_window_priors(rgb_window, self.fs_target, self.prior_names)
            valid[r] = bool(success.any())
            if valid[r]:
                x[r, :3] = standardize_channels(rgb_window).T
                x[r, 3:] = priors
                prior_valid[r] = success
        if not valid.any():
            raise RuntimeError("Previously retained window now has no usable ROI; classical methods/data changed")
        raw_target = rec["ppg"][start:end]
        hr = float(estimate_hr_welch(raw_target, self.fs_target).hr_bpm)
        if not np.isfinite(hr):
            fallback = rec["hr"][start:end]
            hr = float(fallback[np.isfinite(fallback)].mean()) if np.isfinite(fallback).any() else float("nan")
        if not np.isfinite(hr):
            raise ValueError(f"No valid window-level HR target for {rec['subject_id']} at {rec['timestamps'][start]}")
        return dict(x=torch.from_numpy(x), y_ppg=torch.from_numpy(standardize_1d(raw_target)),
                    y_hr=torch.tensor(hr, dtype=torch.float32), roi_valid=torch.from_numpy(valid),
                    roi_quality=torch.from_numpy(quality), prior_valid=torch.from_numpy(prior_valid),
                    subject_id=rec["subject_id"], start_sec=torch.tensor(rec["timestamps"][start], dtype=torch.float64))


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
        if target not in {"both", "sbp", "dbp"}:
            raise ValueError("BP target must be both, sbp or dbp")
        self.data_dir = Path(data_dir)
        self.target = target
        self.rows: List[Dict] = []
        self.exclusions: List[Dict] = []
        for feat_path in sorted(self.data_dir.glob("*/*/features.json")):
            label_path = feat_path.with_name("labels.json")
            if not label_path.exists():
                self.exclusions.append({"path": str(feat_path), "reason": "missing_bp_label"})
                continue
            feat = json.loads(feat_path.read_text(encoding="utf-8"))
            lab = json.loads(label_path.read_text(encoding="utf-8"))
            cuff = lab.get("cuff", {})
            if "sbp" not in cuff or "dbp" not in cuff:
                self.exclusions.append({"path": str(feat_path), "reason": "invalid_bp_reference"})
                continue
            try:
                sbp, dbp = float(cuff["sbp"]), float(cuff["dbp"])
            except (TypeError, ValueError):
                self.exclusions.append({"path": str(feat_path), "reason": "invalid_bp_reference"})
                continue
            if not np.isfinite([sbp, dbp]).all() or not 0 < dbp < sbp:
                self.exclusions.append({"path": str(feat_path), "reason": "invalid_bp_reference"})
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
        reserved = {"sbp", "dbp", "sbp_mmhg", "dbp_mmhg", "bp", "target", "label", "reference_sbp", "reference_dbp"}
        if any(k.lower() in reserved for k in self.feature_keys):
            raise ValueError("BP target/reference labels cannot be feature columns")

    @property
    def subject_ids(self) -> List[str]:
        return [str(row["subject_id"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        feat = row["features"]
        x = np.array([float(feat[k]) if feat.get(k) is not None else np.nan for k in self.feature_keys], dtype=np.float32)
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
