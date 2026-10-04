from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rppg_lab.datasets import find_ubfc_subjects, subject_split, UBFCRPPGDataset
from rppg_lab.classical import DEFAULT_PRIORS
from rppg_lab.metrics import regression_metrics
from rppg_lab.models import VideoHRNet
from rppg_lab.signals import estimate_hr_welch
from rppg_lab.video_datasets import UBFCVideoHRDataset


def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_hr_targets(hr: torch.Tensor, bins: torch.Tensor, sigma_bpm: float = 3.0) -> torch.Tensor:
    hr = hr.float().view(-1, 1)
    target = torch.exp(-0.5 * ((bins.view(1, -1) - hr) / float(sigma_bpm)) ** 2)
    target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)
    return target


def hr_distribution_ce(logits: torch.Tensor, hr: torch.Tensor, bins: torch.Tensor, sigma_bpm: float = 3.0) -> torch.Tensor:
    target = soft_hr_targets(hr, bins.to(logits.device), sigma_bpm=sigma_bpm)
    logp = torch.log_softmax(logits, dim=-1)
    return -(target * logp).sum(dim=-1).mean()


def quality_metrics(gt: np.ndarray, pred: np.ndarray, q: np.ndarray, coverages=(1.0, 0.8, 0.6, 0.4, 0.2)) -> List[Tuple[float, int, float, float]]:
    out = []
    order = np.argsort(-q)
    n = len(gt)
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        idx = order[:k]
        err = np.abs(pred[idx] - gt[idx])
        out.append((100.0 * cov, k, float(err.mean()), float(np.median(err))))
    return out


def evaluate(model: VideoHRNet, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    gt, pred, q, subjects = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            y = batch["y_hr"].cpu().numpy().astype(float)
            out = model(video, return_dict=True)
            p = out["hr_bpm"].detach().cpu().numpy().astype(float)
            qq = out["quality"].detach().cpu().numpy().astype(float)
            gt.extend(y.tolist())
            pred.extend(p.tolist())
            q.extend(qq.tolist())
            subjects.extend(list(batch["subject_id"]))
    gt_a = np.asarray(gt, dtype=float)
    pred_a = np.asarray(pred, dtype=float)
    q_a = np.asarray(q, dtype=float)
    m = regression_metrics(gt_a, pred_a)
    err = np.abs(pred_a - gt_a)
    corr_q = np.corrcoef(q_a, -err)[0, 1] if len(q_a) > 2 and np.std(q_a) > 0 and np.std(err) > 0 else np.nan
    return {
        "gt": gt_a,
        "pred": pred_a,
        "quality": q_a,
        "subjects": subjects,
        "metrics": m,
        "quality_error_corr": float(corr_q) if np.isfinite(corr_q) else float("nan"),
        "coverage": quality_metrics(gt_a, pred_a, q_a),
    }


def print_eval(ev: Dict, prefix: str = "val") -> None:
    m = ev["metrics"]
    print(f"{prefix}_HR_MAE={m.mae:.3f} {prefix}_HR_RMSE={m.rmse:.3f} {prefix}_HR_r={m.pearson:.3f}")
    print(f"{prefix}_quality_error_corr={ev['quality_error_corr']:.3f} quality_mean={np.mean(ev['quality']):.3f}")
    for cov, k, mae, med in ev["coverage"]:
        print(f"  top_quality_coverage={cov:5.1f}% N={k:4d} MAE={mae:7.3f} median={med:7.3f}")


def print_trace_baselines(val_subjects, fs: float, clip_sec: float, stride_sec: float, roi: str, cache_dir: str) -> None:
    """Print 1D baseline performance on the same validation subjects/windows.

    This is not exactly the same tensor path as video training, but it anchors
    the video model against the strong HRQualityNet/prior baselines.
    """
    try:
        ds = UBFCRPPGDataset(
            val_subjects,
            fs_target=fs,
            win_sec=clip_sec,
            stride_sec=stride_sec,
            roi=roi,
            cache_dir=cache_dir,
            prior_methods=list(DEFAULT_PRIORS),
        )
        names = ["RGB_R", "RGB_G", "RGB_B"] + list(DEFAULT_PRIORS)
        gt = []
        preds = {name: [] for name in names}
        for i in range(len(ds)):
            item = ds[i]
            x = item["x"].numpy()
            gt.append(float(item["y_hr"]))
            for ch, name in enumerate(names):
                preds[name].append(estimate_hr_welch(x[ch], fs).hr_bpm)
        rows = []
        gt_a = np.asarray(gt, dtype=float)
        for name, p in preds.items():
            m = regression_metrics(gt_a, np.asarray(p, dtype=float))
            rows.append({"source": name, "MAE": m.mae, "RMSE": m.rmse, "ME": m.me, "r": m.pearson})
        df = pd.DataFrame(rows).sort_values("MAE")
        print("\nValidation 1D trace/prior baselines on same subjects/windows:")
        print(df.to_string(index=False, formatters={"MAE": "{:.3f}".format, "RMSE": "{:.3f}".format, "ME": "{:.3f}".format, "r": "{:.3f}".format}))
    except Exception as e:
        print(f"Could not compute 1D baselines: {e}")


def save_predictions_csv(ev: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "subject_id": ev["subjects"],
        "gt_hr": ev["gt"],
        "pred_hr": ev["pred"],
        "quality": ev["quality"],
    })
    df["abs_err"] = (df["pred_hr"] - df["gt_hr"]).abs()
    df.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train first video-based HR + quality model on UBFC clips.")
    ap.add_argument("--ubfc-root", required=True)
    ap.add_argument("--cache-dir", default="cache_video")
    ap.add_argument("--roi-cache-dir", default="cache_roi_oldpoints", help="Cache for 1D baseline extraction only.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--clip-sec", type=float, default=10.0)
    ap.add_argument("--stride-sec", type=float, default=2.0)
    ap.add_argument("--clip-frames", type=int, default=160)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows first; increase later if stable.")
    ap.add_argument("--hr-min", type=float, default=40.0)
    ap.add_argument("--hr-max", type=float, default=180.0)
    ap.add_argument("--hr-step", type=float, default=1.0)
    ap.add_argument("--sigma-bpm", type=float, default=3.0)
    ap.add_argument("--hr-reg-weight", type=float, default=0.05)
    ap.add_argument("--quality-weight", type=float, default=0.10)
    ap.add_argument("--quality-save-coverage", type=float, default=0.60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--out", default="checkpoints/video_hrnet_ubfc.pt")
    ap.add_argument("--pred-out", default="outputs/video_hrnet_ubfc_val_predictions.csv")
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    subjects = find_ubfc_subjects(args.ubfc_root)
    train_subj, val_subj = subject_split(subjects, val_fraction=args.val_fraction, seed=args.seed)
    print("Train subjects:", [s.subject_id for s in train_subj])
    print("Val subjects:", [s.subject_id for s in val_subj])

    print_trace_baselines(val_subj, fs=args.fs, clip_sec=args.clip_sec, stride_sec=args.stride_sec, roi="face", cache_dir=args.roi_cache_dir)

    train_ds = UBFCVideoHRDataset(
        train_subj,
        clip_sec=args.clip_sec,
        stride_sec=args.stride_sec,
        clip_frames=args.clip_frames,
        image_size=args.image_size,
        fs_target=args.fs,
        cache_dir=args.cache_dir,
        min_hr=args.hr_min,
        max_hr=args.hr_max,
    )
    val_ds = UBFCVideoHRDataset(
        val_subj,
        clip_sec=args.clip_sec,
        stride_sec=args.stride_sec,
        clip_frames=args.clip_frames,
        image_size=args.image_size,
        fs_target=args.fs,
        cache_dir=args.cache_dir,
        min_hr=args.hr_min,
        max_hr=args.hr_max,
    )
    print(f"Train video samples: {len(train_ds)} | Val video samples: {len(val_ds)}")
    print(f"Video tensor: T={args.clip_frames}, C=3, H=W={args.image_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = VideoHRNet(
        hr_min=args.hr_min,
        hr_max=args.hr_max,
        hr_step=args.hr_step,
        frame_feature_dim=96,
        temporal_channels=96,
        num_temporal_blocks=4,
        dropout=args.dropout,
    ).to(device)
    print(f"Model: VideoHRNet | params={sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    print("\nInitial evaluation:")
    init_ev = evaluate(model, val_loader, device)
    print_eval(init_ev, prefix="val")

    best_mae = math.inf
    best_quality_mae = math.inf
    best_epoch = -1
    no_improve = 0
    out_path = Path(args.out)
    quality_path = out_path.with_name(out_path.stem + f"_quality{int(args.quality_save_coverage*100)}" + out_path.suffix)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            video = batch["video"].to(device)
            y_hr = batch["y_hr"].to(device)
            q_target = batch["quality"].to(device)
            out = model(video, return_dict=True)
            ce = hr_distribution_ce(out["hr_logits"], y_hr, model.hr_bins, sigma_bpm=args.sigma_bpm)
            l1 = F.smooth_l1_loss(out["hr_bpm"], y_hr)
            q_loss = F.binary_cross_entropy_with_logits(out["quality_logit"], q_target)
            loss = ce + args.hr_reg_weight * l1 + args.quality_weight * q_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        ev = evaluate(model, val_loader, device)
        m = ev["metrics"]
        print(f"epoch={epoch} train_loss={np.mean(losses):.4f} lr={scheduler.get_last_lr()[0]:.2e} val_HR_MAE={m.mae:.3f} val_HR_RMSE={m.rmse:.3f} val_HR_r={m.pearson:.3f}")
        print_eval(ev, prefix="val")

        if m.mae < best_mae - 1e-6:
            best_mae = m.mae
            best_epoch = epoch
            no_improve = 0
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_mae": best_mae,
            }, out_path)
            save_predictions_csv(ev, args.pred_out)
            print(f"saved best overall: {out_path}")
        else:
            no_improve += 1

        # Save quality-aware checkpoint.
        qcov = float(args.quality_save_coverage)
        q_row = min(ev["coverage"], key=lambda x: abs(x[0] / 100.0 - qcov))
        q_mae = q_row[2]
        if q_mae < best_quality_mae - 1e-6:
            best_quality_mae = q_mae
            quality_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "quality_coverage": qcov,
                "quality_mae": best_quality_mae,
            }, quality_path)
            print(f"saved best quality@{int(qcov*100)}%: {quality_path} MAE={best_quality_mae:.3f}")

        if no_improve >= args.patience:
            print(f"Early stopping: no overall improvement for {args.patience} epochs. Best epoch={best_epoch}, best val_HR_MAE={best_mae:.3f}; best quality@{int(args.quality_save_coverage*100)}% MAE={best_quality_mae:.3f}")
            break


if __name__ == "__main__":
    main()
