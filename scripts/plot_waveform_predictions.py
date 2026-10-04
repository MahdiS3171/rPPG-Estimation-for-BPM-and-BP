#!/usr/bin/env python
"""Plot predicted waveform vs. reference PPG on UBFC validation windows.

This script is intentionally visual: HR numbers alone can hide failure modes.
It plots target PPG, model output, and optionally the prior channels for selected
validation windows.  Use it after training to check whether the model is really
recovering a pulse-like waveform or only producing the right dominant frequency.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.datasets import find_ubfc_subjects, subject_split, UBFCRPPGDataset
from rppg_lab.classical import DEFAULT_PRIORS
from rppg_lab.models import WaveformNet, PriorResidualWaveformNet
from rppg_lab.metrics import waveform_corr, waveform_corr_aligned
from rppg_lab.signals import estimate_hr_welch, standardize_1d


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "waveform")
    in_ch = int(ckpt.get("in_channels", 3))
    prior_names = list(ckpt.get("prior_names", []))
    args = ckpt.get("args", {})

    if model_type == "residual" or ckpt.get("model_class") == "PriorResidualWaveformNet":
        model = PriorResidualWaveformNet(
            in_channels=in_ch,
            prior_names=prior_names,
            base_channels=int(args.get("base_channels", 32)),
            num_blocks=int(args.get("num_blocks", 3)),
            dropout=float(args.get("dropout", 0.20)),
            residual_scale=float(args.get("residual_scale", 0.10)),
            init_prior=str(args.get("init_prior", "auto")),
        )
    else:
        model = WaveformNet(
            in_channels=in_ch,
            base_channels=int(args.get("base_channels", 32)),
            num_blocks=int(args.get("num_blocks", 3)),
            dropout=float(args.get("dropout", 0.15)),
        )

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ubfc-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default="cache_roi")
    ap.add_argument("--out-dir", default="outputs/waveform_plots")
    ap.add_argument("--fs", type=float, default=None)
    ap.add_argument("--win-sec", type=float, default=None)
    ap.add_argument("--stride-sec", type=float, default=2.0)
    ap.add_argument("--roi", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-plots", type=int, default=12)
    ap.add_argument("--mode", choices=["first", "best", "worst", "random"], default="worst")
    ap.add_argument("--show-priors", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(Path(args.checkpoint), device)

    ckpt_args = ckpt.get("args", {})
    fs = float(args.fs if args.fs is not None else ckpt.get("fs", ckpt_args.get("fs", 30.0)))
    win_sec = float(args.win_sec if args.win_sec is not None else ckpt.get("win_sec", ckpt_args.get("win_sec", 10.0)))
    roi = args.roi if args.roi is not None else ckpt.get("roi", ckpt_args.get("roi", "face"))
    prior_names = list(ckpt.get("prior_names", []))

    subjects = find_ubfc_subjects(args.ubfc_root)
    _, val_subj = subject_split(subjects, val_fraction=0.2, seed=args.seed)
    ds = UBFCRPPGDataset(
        val_subj,
        fs_target=fs,
        win_sec=win_sec,
        stride_sec=args.stride_sec,
        roi=roi,
        cache_dir=args.cache_dir,
        prior_methods=prior_names,
    )

    rows = []
    with torch.no_grad():
        for idx in range(len(ds)):
            item = ds[idx]
            x = item["x"].unsqueeze(0).to(device)
            out = model(x, return_dict=True)
            pred = out["ppg"].squeeze(0).cpu().numpy()
            y = item["y_ppg"].numpy()
            gt_hr = float(item["y_hr"])
            pred_hr = estimate_hr_welch(pred, fs).hr_bpm
            rows.append({
                "idx": idx,
                "subject_id": item["subject_id"],
                "abs_err": abs(pred_hr - gt_hr),
                "gt_hr": gt_hr,
                "pred_hr": pred_hr,
                "corr": waveform_corr(y, pred),
                "corr_aligned": waveform_corr_aligned(y, pred, max_lag=int(0.5 * fs)),
            })

    if args.mode == "first":
        selected = rows[: args.num_plots]
    elif args.mode == "best":
        selected = sorted(rows, key=lambda r: r["abs_err"])[: args.num_plots]
    elif args.mode == "random":
        rng = np.random.default_rng(args.seed)
        selected = [rows[i] for i in rng.choice(len(rows), size=min(args.num_plots, len(rows)), replace=False)]
    else:
        selected = sorted(rows, key=lambda r: r["abs_err"], reverse=True)[: args.num_plots]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(int(round(win_sec * fs))) / fs
    for j, row in enumerate(selected):
        item = ds[row["idx"]]
        x = item["x"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x, return_dict=True)
        pred = out["ppg"].squeeze(0).cpu().numpy()
        target = item["y_ppg"].numpy()
        pred_z = standardize_1d(pred)
        target_z = standardize_1d(target)

        plt.figure(figsize=(11, 5))
        plt.plot(t[:len(target_z)], target_z, label="Target PPG (z)", linewidth=1.6)
        plt.plot(t[:len(pred_z)], pred_z, label="Model output (z)", linewidth=1.4)

        if args.show_priors and prior_names:
            arr = item["x"].numpy()
            for k, name in enumerate(prior_names):
                ch = 3 + k
                if ch < arr.shape[0]:
                    plt.plot(t[:arr.shape[1]], standardize_1d(arr[ch]), alpha=0.45, linewidth=0.8, label=f"prior:{name}")

        title = (
            f"{row['subject_id']} idx={row['idx']} | "
            f"GT={row['gt_hr']:.1f} Pred={row['pred_hr']:.1f} "
            f"Err={row['abs_err']:.1f} | corr={row['corr']:.2f} aligned={row['corr_aligned']:.2f}"
        )
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("z-score")
        plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        out_path = out_dir / f"{j:02d}_{row['subject_id']}_idx{row['idx']}_err{row['abs_err']:.1f}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print("saved", out_path)


if __name__ == "__main__":
    main()
