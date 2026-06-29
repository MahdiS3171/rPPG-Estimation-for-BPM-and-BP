from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from scipy.signal import butter, filtfilt, find_peaks
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.classical import METHOD_FUNCS
from rppg_lab.metrics import regression_metrics, bland_altman
from rppg_lab.signals import estimate_hr_welch, resample_uniform


ROI_FILE_SUFFIX = {
    "forehead": "Forehead",
    "cheek1": "Cheek1",
    "cheek2": "Cheek2",
}


def natural_subject_key(path: Path) -> Tuple[int, str]:
    m = re.search(r"Subject_(\d+)", path.name)
    if m:
        return int(m.group(1)), path.name
    return 10**9, path.name


def read_cropped_rgb_trace(video_path: Path, fs_target: float = 30.0) -> Tuple[np.ndarray, float, np.ndarray, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps_reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    means = []
    times = []
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # OpenCV reads BGR; convert average to RGB.
        mean_bgr = frame_bgr.reshape(-1, 3).mean(axis=0)
        mean_rgb = mean_bgr[::-1]
        means.append(mean_rgb)

        pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if pos_msec > 0:
            t = pos_msec / 1000.0
        elif fps_reported > 0:
            t = frame_idx / fps_reported
        else:
            t = float(frame_idx)
        times.append(t)

        frame_idx += 1

    cap.release()

    if len(means) < 2:
        raise RuntimeError(f"Too few frames in video: {video_path}")

    rgb = np.asarray(means, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)

    # Normalize timeline to start at zero.
    t = t - t[0]

    # If CAP_PROP_POS_MSEC is weird, fall back to reported FPS.
    if not np.all(np.isfinite(t)) or np.sum(np.diff(t) > 0) < len(t) * 0.9:
        if fps_reported <= 0:
            raise RuntimeError(f"Invalid timestamps and FPS for video: {video_path}")
        t = np.arange(len(rgb), dtype=np.float64) / fps_reported

    duration = float(t[-1] - t[0])
    if duration <= 0 and fps_reported > 0:
        duration = len(rgb) / fps_reported

    t_u, rgb_u = resample_uniform(t, rgb, fs_target, t_start=0.0, t_end=duration)
    return rgb_u.astype(np.float32), float(fs_target), t_u, float(fps_reported), len(rgb)


def load_ecg_1d(ecg_path: Path) -> np.ndarray:
    arr = np.load(ecg_path, allow_pickle=False)
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        ecg = arr
    elif arr.ndim == 2:
        # If one dimension looks like channels, use the first channel.
        if arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0]:
            ecg = arr[0, :]
        elif arr.shape[1] <= 4 and arr.shape[0] > arr.shape[1]:
            ecg = arr[:, 0]
        else:
            # Conservative fallback: use first row flattened.
            ecg = arr.reshape(arr.shape[0], -1)[0]
    else:
        ecg = arr.reshape(-1)

    ecg = np.asarray(ecg, dtype=np.float64)
    ecg = np.nan_to_num(ecg, nan=np.nanmedian(ecg))
    return ecg

def bandpass_ecg_for_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(ecg, dtype=np.float64)
    x = np.nan_to_num(x - np.nanmedian(x), nan=0.0)

    # ECG R-peaks are usually clear in roughly 5-20 Hz.
    # This is not for medical ECG analysis; it is only for HR reference extraction.
    low = 5.0 / (0.5 * fs)
    high = 20.0 / (0.5 * fs)

    if high >= 1.0:
        high = 0.99
    if low <= 0.0:
        low = 0.001

    b, a = butter(3, [low, high], btype="bandpass")
    return filtfilt(b, a, x)


def estimate_hr_from_ecg_rpeaks(
    ecg: np.ndarray,
    fs_ecg: float,
    start_sec: float,
    end_sec: float,
) -> float:
    start_idx = max(0, int(round(start_sec * fs_ecg)))
    end_idx = min(len(ecg), int(round(end_sec * fs_ecg)))

    if end_idx - start_idx < int(5 * fs_ecg):
        return float("nan")

    segment = ecg[start_idx:end_idx]
    y = bandpass_ecg_for_rpeaks(segment, fs_ecg)

    # Robust threshold based on signal distribution.
    abs_y = np.abs(y)
    height = np.percentile(abs_y, 90)

    # Minimum distance between R-peaks.
    # 0.30s allows up to 200 bpm.
    min_distance = int(round(0.30 * fs_ecg))

    peaks_pos, _ = find_peaks(y, height=height, distance=min_distance)
    peaks_neg, _ = find_peaks(-y, height=height, distance=min_distance)

    # ECG polarity may vary; choose the polarity with more plausible peaks.
    duration = end_sec - start_sec

    candidates = []
    for peaks in [peaks_pos, peaks_neg]:
        n_beats = len(peaks)
        if duration > 0:
            hr = 60.0 * n_beats / duration
        else:
            hr = float("nan")

        if np.isfinite(hr) and 35.0 <= hr <= 220.0:
            candidates.append((n_beats, hr))

    if not candidates:
        return float("nan")

    # Prefer the polarity with more detected peaks.
    candidates.sort(key=lambda x: x[0], reverse=True)
    return float(candidates[0][1])


def load_rppg10_subject(root: Path, subject_dir: Path, roi: str, fs_target: float) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    subject_name = subject_dir.name
    subject_num = subject_name.split("_")[-1]

    ecg_path = subject_dir / f"Subject_{subject_num}_ECG.npy"
    if not ecg_path.exists():
        raise FileNotFoundError(ecg_path)

    if roi == "face":
        rgb_list = []
        t_list = []
        fps_list = []
        raw_frame_counts = []

        for roi_name in ["forehead", "cheek1", "cheek2"]:
            suffix = ROI_FILE_SUFFIX[roi_name]
            video_path = subject_dir / f"Subject_{subject_num}_{suffix}_.avi"
            if not video_path.exists():
                raise FileNotFoundError(video_path)
            rgb, fs, t, fps_reported, n_raw = read_cropped_rgb_trace(video_path, fs_target=fs_target)
            rgb_list.append(rgb)
            t_list.append(t)
            fps_list.append(fps_reported)
            raw_frame_counts.append(n_raw)

        n = min(len(x) for x in rgb_list)
        rgb = np.mean(np.stack([x[:n] for x in rgb_list], axis=0), axis=0)
        t_rgb = t_list[0][:n]
        fps_reported = float(np.nanmean(fps_list))
        n_raw_frames = int(min(raw_frame_counts))

    else:
        suffix = ROI_FILE_SUFFIX[roi]
        video_path = subject_dir / f"Subject_{subject_num}_{suffix}_.avi"
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        rgb, fs, t_rgb, fps_reported, n_raw_frames = read_cropped_rgb_trace(video_path, fs_target=fs_target)

    ecg = load_ecg_1d(ecg_path)

    duration_rgb = float(t_rgb[-1] - t_rgb[0] + 1.0 / fs_target)

    # rPPG-10 ECG files appear to be 10 minutes at ~1000 Hz
    # e.g. Subject_1_ECG.npy has length 600000.
    fs_ecg_inferred = float(len(ecg) / duration_rgb)

    t_end = duration_rgb
    t_common, rgb_u = resample_uniform(t_rgb, rgb, fs_target, t_start=0.0, t_end=t_end)

    return (
        rgb_u.astype(np.float32),
        ecg.astype(np.float64),
        float(fs_target),
        float(fs_ecg_inferred),
        fps_reported,
        n_raw_frames,
    )


def sliding_windows(n: int, fs: float, win_sec: float, stride_sec: float) -> List[Tuple[int, int]]:
    win = int(round(win_sec * fs))
    stride = max(1, int(round(stride_sec * fs)))
    out = []
    s = 0
    while s + win <= n:
        out.append((s, s + win))
        s += stride
    return out


def find_subject_dirs(root: Path) -> List[Path]:
    dirs = [p for p in root.iterdir() if p.is_dir() and re.match(r"Subject_\d+$", p.name)]
    return sorted(dirs, key=natural_subject_key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rppg10-root", required=True)
    ap.add_argument("--method", default="CHROM", choices=sorted(METHOD_FUNCS))
    ap.add_argument("--roi", default="face", choices=["face", "forehead", "cheek1", "cheek2"])
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--win-sec", type=float, default=30.0)
    ap.add_argument("--stride-sec", type=float, default=10.0)
    ap.add_argument("--out", default="outputs/rppg10_classical.csv")
    ap.add_argument("--max-subjects", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.rppg10_root)
    subject_dirs = find_subject_dirs(root)
    if args.max_subjects and args.max_subjects > 0:
        subject_dirs = subject_dirs[: args.max_subjects]

    rows = []
    y_true = []
    y_pred = []

    for subject_dir in subject_dirs:
        try:
            rgb, ecg, fs, fs_ecg, fps_reported, n_raw = load_rppg10_subject(root, subject_dir, args.roi, args.fs)
            rppg = METHOD_FUNCS[args.method](rgb, fs)

            windows = sliding_windows(len(rppg), fs, args.win_sec, args.stride_sec)
            if not windows:
                print(f"{subject_dir.name}: skipped, too short after loading")
                continue

            subject_errs = []

            for w_idx, (s, e) in enumerate(windows):
                pred = estimate_hr_welch(rppg[s:e], fs)

                start_sec = s / fs
                end_sec = e / fs
                gt_hr = estimate_hr_from_ecg_rpeaks(ecg, fs_ecg, start_sec, end_sec)

                if not np.isfinite(pred.hr_bpm) or not np.isfinite(gt_hr):
                    continue

                err = float(pred.hr_bpm - gt_hr)
                abs_err = abs(err)

                rows.append({
                    "subject_id": subject_dir.name,
                    "window_idx": w_idx,
                    "start_sec": s / fs,
                    "end_sec": e / fs,
                    "gt_hr": gt_hr,
                    "pred_hr": pred.hr_bpm,
                    "err": err,
                    "abs_err": abs_err,
                    "confidence": pred.confidence,
                    "snr_db": pred.snr_db,
                    "method": args.method,
                    "roi": args.roi,
                    "fps_reported": fps_reported,
                    "fs_ecg": fs_ecg,
                    "n_raw_frames": n_raw,
                    "n_samples": len(rppg),
                })

                y_true.append(gt_hr)
                y_pred.append(pred.hr_bpm)
                subject_errs.append(abs_err)

            if subject_errs:
                print(
                    f"{subject_dir.name}: windows={len(subject_errs)} "
                    f"subject_MAE={np.mean(subject_errs):.3f} "
                    f"fps={fps_reported:.3f} raw_frames={n_raw}"
                )

        except Exception as e:
            print(f"{subject_dir.name}: ERROR {e}")

    if not rows:
        raise RuntimeError("No benchmark rows produced.")

    m = regression_metrics(y_true, y_pred)
    ba = bland_altman(y_true, y_pred)

    print("\nWindow-level Summary")
    print(f"N={m.n} MAE={m.mae:.3f} RMSE={m.rmse:.3f} ME={m.me:.3f} SD={m.sd:.3f} r={m.pearson:.3f}")
    print(f"Bland-Altman: bias={ba['bias']:.3f}, LoA=[{ba['loa_low']:.3f}, {ba['loa_high']:.3f}]")

    # Subject-level average of per-window MAE.
    import pandas as pd
    df = pd.DataFrame(rows)
    subj_mae = df.groupby("subject_id")["abs_err"].mean()
    print("\nSubject-level MAE summary")
    print(f"N_subjects={len(subj_mae)} mean_subject_MAE={subj_mae.mean():.3f} median_subject_MAE={subj_mae.median():.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", out)


if __name__ == "__main__":
    main()