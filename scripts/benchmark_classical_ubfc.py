#!/usr/bin/env python
"""Classical rPPG benchmark on UBFC with subject-wise reporting.

This script is deliberately simple: it evaluates full-video HR per subject first.
Window-level evaluation can be added later, but subject-level is a safer first
sanity check and avoids accidental window leakage.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import csv
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.datasets import find_ubfc_subjects, load_ubfc_ground_truth
from rppg_lab.roi import extract_rgb_trace
from rppg_lab.classical import METHOD_FUNCS
from rppg_lab.signals import estimate_hr_welch
from rppg_lab.metrics import regression_metrics, bland_altman


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ubfc-root", required=True)
    ap.add_argument("--roi", default="face")
    ap.add_argument("--method", default="POS_WIN", choices=sorted(METHOD_FUNCS))
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--cache-dir", default="cache_roi")
    ap.add_argument("--out", default="outputs/classical_ubfc.csv")
    args = ap.parse_args()

    subjects = find_ubfc_subjects(args.ubfc_root)
    rows = []
    y_true, y_pred = [], []
    for subj in subjects:
        rgb, fs, t, q = extract_rgb_trace(subj.video_path, roi=args.roi, cache_dir=args.cache_dir, fs_target=args.fs)
        rppg = METHOD_FUNCS[args.method](rgb, fs)
        pred = estimate_hr_welch(rppg, fs)
        gt_t, gt_hr, _ = load_ubfc_ground_truth(subj.gt_path)
        gt = float(np.nanmean(gt_hr))
        rows.append({
            "subject_id": subj.subject_id,
            "gt_hr": gt,
            "pred_hr": pred.hr_bpm,
            "confidence": pred.confidence,
            "snr_db": pred.snr_db,
            "n_frames": len(rgb),
        })
        y_true.append(gt)
        y_pred.append(pred.hr_bpm)
        print(f"{subj.subject_id}: gt={gt:.2f} pred={pred.hr_bpm:.2f} snr={pred.snr_db:.2f}")

    m = regression_metrics(y_true, y_pred)
    ba = bland_altman(y_true, y_pred)
    print("\nSummary")
    print(f"N={m.n} MAE={m.mae:.3f} RMSE={m.rmse:.3f} ME={m.me:.3f} SD={m.sd:.3f} r={m.pearson:.3f}")
    print(f"Bland-Altman: bias={ba['bias']:.3f}, LoA=[{ba['loa_low']:.3f}, {ba['loa_high']:.3f}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", out)


if __name__ == "__main__":
    main()
