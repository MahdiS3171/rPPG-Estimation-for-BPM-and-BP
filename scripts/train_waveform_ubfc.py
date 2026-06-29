#!/usr/bin/env python
"""Train waveform-first rPPG models on UBFC with subject-wise split.

Default behavior
----------------
If prior channels are enabled, the script uses PriorResidualWaveformNet.  This
model starts from a conservative classical-prior fusion and learns a small
residual.  This is much safer than rebuilding the waveform from scratch when a
strong prior such as CHROM_WIN is already available.

Use --model waveform for the older generic WaveformNet ablation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.datasets import find_ubfc_subjects, subject_split, UBFCRPPGDataset
from rppg_lab.classical import DEFAULT_PRIORS
from rppg_lab.models import WaveformNet, PriorResidualWaveformNet
from rppg_lab.losses import rppg_training_loss, waveform_combo_loss
from rppg_lab.metrics import waveform_corr, waveform_corr_aligned, regression_metrics
from rppg_lab.signals import estimate_hr_welch


def _safe_mean(xs):
    return float(np.nanmean(xs)) if len(xs) else float("nan")


def evaluate_model(model, loader, device, fs: float):
    model.eval()
    cors, cors_aligned, gt_hr, pr_hr = [], [], [], []
    prior_weights = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y_ppg"].numpy()
            out = model(x, return_dict=True) if hasattr(model, "forward") else model(x)
            if isinstance(out, dict):
                pred_t = out["ppg"]
                if "prior_weights" in out:
                    prior_weights.append(out["prior_weights"].detach().cpu().numpy())
            else:
                pred_t = out
            pred = pred_t.detach().cpu().numpy()

            for i in range(pred.shape[0]):
                cors.append(waveform_corr(y[i], pred[i]))
                cors_aligned.append(waveform_corr_aligned(y[i], pred[i], max_lag=int(0.5 * fs)))
                pr_hr.append(estimate_hr_welch(pred[i], fs).hr_bpm)
                gt_hr.append(float(batch["y_hr"][i]))

    hr_m = regression_metrics(gt_hr, pr_hr)
    out = {
        "wave_corr": _safe_mean(cors),
        "wave_corr_aligned": _safe_mean(cors_aligned),
        "hr_metrics": hr_m,
    }
    if prior_weights:
        out["prior_weights_mean"] = np.concatenate(prior_weights, axis=0).mean(axis=0)
    return out


def evaluate_input_channel_baselines(ds, fs: float, names: list[str]):
    gt = []
    preds = {name: [] for name in names}
    preds["TARGET_PPG_WELCH"] = []
    for i in range(len(ds)):
        item = ds[i]
        x = item["x"].numpy()
        y = item["y_ppg"].numpy()
        y_hr = float(item["y_hr"])
        gt.append(y_hr)
        for ch, name in enumerate(names):
            preds[name].append(estimate_hr_welch(x[ch], fs).hr_bpm)
        preds["TARGET_PPG_WELCH"].append(estimate_hr_welch(y, fs).hr_bpm)

    rows = []
    for name, pr in preds.items():
        m = regression_metrics(gt, pr)
        rows.append((name, m.mae, m.rmse, m.me, m.sd, m.pearson))
    rows.sort(key=lambda r: r[1])
    return rows


def build_model(args, in_ch: int, prior_names: list[str], device: torch.device):
    model_name = args.model
    if model_name == "auto":
        model_name = "residual" if prior_names else "waveform"

    if model_name == "residual":
        if not prior_names:
            raise ValueError("--model residual requires prior channels. Remove --no-priors.")
        model = PriorResidualWaveformNet(
            in_channels=in_ch,
            prior_names=prior_names,
            base_channels=args.base_channels,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            residual_scale=args.residual_scale,
            init_prior=args.init_prior,
        )
    elif model_name == "waveform":
        model = WaveformNet(
            in_channels=in_ch,
            base_channels=args.base_channels,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model.to(device), model_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ubfc-root", required=True)
    ap.add_argument("--cache-dir", default="cache_roi")
    ap.add_argument("--out", default="checkpoints/waveformnet_ubfc.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--win-sec", type=float, default=10.0)
    ap.add_argument("--stride-sec", type=float, default=2.0)
    ap.add_argument("--roi", default="face")
    ap.add_argument("--no-priors", action="store_true")
    ap.add_argument("--priors", default=",".join(DEFAULT_PRIORS), help="Comma-separated prior method names")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--model", choices=["auto", "residual", "waveform"], default="auto")
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--num-blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--residual-scale", type=float, default=0.10)
    ap.add_argument("--init-prior", default="auto", help="auto or a prior name such as CHROM_WIN/GREEN")

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["rppg", "legacy"], default="rppg")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subjects = find_ubfc_subjects(args.ubfc_root)
    train_subj, val_subj = subject_split(subjects, val_fraction=0.2, seed=args.seed)

    prior_names = [] if args.no_priors else [p.strip() for p in args.priors.split(",") if p.strip()]
    print(f"Using priors: {prior_names}")
    print("Train subjects:", [s.subject_id for s in train_subj])
    print("Val subjects:", [s.subject_id for s in val_subj])

    train_ds = UBFCRPPGDataset(
        train_subj,
        fs_target=args.fs,
        win_sec=args.win_sec,
        stride_sec=args.stride_sec,
        roi=args.roi,
        cache_dir=args.cache_dir,
        prior_methods=prior_names,
    )
    val_ds = UBFCRPPGDataset(
        val_subj,
        fs_target=args.fs,
        win_sec=args.win_sec,
        stride_sec=args.stride_sec,
        roi=args.roi,
        cache_dir=args.cache_dir,
        prior_methods=prior_names,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    sample = train_ds[0]
    in_ch = int(sample["x"].shape[0])
    channel_names = ["RGB_R", "RGB_G", "RGB_B"] + prior_names

    print(f"Input channels: {in_ch}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print("\nValidation input-channel baselines:")
    for name, mae, rmse, me, sd, r in evaluate_input_channel_baselines(val_ds, args.fs, channel_names):
        print(f"  {name:16s} MAE={mae:7.3f} RMSE={rmse:7.3f} ME={me:7.3f} SD={sd:7.3f} r={r:7.3f}")

    model, model_name = build_model(args, in_ch=in_ch, prior_names=prior_names, device=device)
    print(f"\nModel: {model_name} ({model.__class__.__name__})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best = np.inf
    bad_epochs = 0

    # Evaluate and save the untrained prior-preserving initialization if it is best.
    init_eval = evaluate_model(model, val_loader, device, args.fs)
    init_hr = init_eval["hr_metrics"]
    print(
        f"epoch=0 train_loss=nan "
        f"val_wave_corr={init_eval['wave_corr']:.3f} "
        f"val_wave_corr_aligned={init_eval['wave_corr_aligned']:.3f} "
        f"val_HR_MAE={init_hr.mae:.3f}"
    )
    if "prior_weights_mean" in init_eval:
        weights = init_eval["prior_weights_mean"]
        print("initial prior weights:", ", ".join(f"{n}={w:.3f}" for n, w in zip(prior_names, weights)))

    # Save initialization if it is already a strong baseline.
    best = init_hr.mae
    torch.save({
        "model_state": model.state_dict(),
        "model_type": model_name,
        "model_class": model.__class__.__name__,
        "in_channels": in_ch,
        "prior_names": prior_names,
        "fs": args.fs,
        "win_sec": args.win_sec,
        "roi": args.roi,
        "args": vars(args),
        "train_subjects": [s.subject_id for s in train_subj],
        "val_subjects": [s.subject_id for s in val_subj],
        "best_val_hr_mae": best,
        "epoch": 0,
    }, out_path)
    print("saved initial/best:", out_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = batch["x"].to(device)
            y = batch["y_ppg"].to(device)
            y_hr = batch["y_hr"].to(device)

            out = model(x, return_dict=True)
            pred = out["ppg"] if isinstance(out, dict) else out

            if args.loss == "legacy":
                loss = waveform_combo_loss(pred, y)
            else:
                residual = out.get("residual") if isinstance(out, dict) else None
                loss = rppg_training_loss(
                    pred=pred,
                    target=y,
                    hr_bpm=y_hr,
                    fs=args.fs,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    residual=residual,
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tr.append(float(loss.item()))

        ev = evaluate_model(model, val_loader, device, args.fs)
        hr_m = ev["hr_metrics"]
        val_score = hr_m.mae
        print(
            f"epoch={epoch} "
            f"train_loss={np.mean(tr):.4f} "
            f"val_wave_corr={ev['wave_corr']:.3f} "
            f"val_wave_corr_aligned={ev['wave_corr_aligned']:.3f} "
            f"val_HR_MAE={hr_m.mae:.3f} "
            f"val_HR_RMSE={hr_m.rmse:.3f} "
            f"val_HR_r={hr_m.pearson:.3f}"
        )
        if "prior_weights_mean" in ev:
            weights = ev["prior_weights_mean"]
            print("prior weights:", ", ".join(f"{n}={w:.3f}" for n, w in zip(prior_names, weights)))

        if val_score < best - args.min_delta:
            best = val_score
            bad_epochs = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_type": model_name,
                "model_class": model.__class__.__name__,
                "in_channels": in_ch,
                "prior_names": prior_names,
                "fs": args.fs,
                "win_sec": args.win_sec,
                "roi": args.roi,
                "args": vars(args),
                "train_subjects": [s.subject_id for s in train_subj],
                "val_subjects": [s.subject_id for s in val_subj],
                "best_val_hr_mae": best,
                "epoch": epoch,
            }, out_path)
            print("saved best:", out_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs. Best val_HR_MAE={best:.3f}")
                break


if __name__ == "__main__":
    main()
