#!/usr/bin/env python
"""Train a direct HR-distribution + quality model.

This is the next-stage model after waveform experiments.  It does not try to
reproduce finger PPG waveform morphology.  Instead, it predicts:

  1) a probability distribution over HR bins;
  2) a confidence/quality score for the current window.

This matches the practical objective better: accurate HR + calibrated reject
option when the rPPG evidence is weak.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.classical import DEFAULT_PRIORS, METHOD_FUNCS
from rppg_lab.datasets import find_ubfc_subjects, subject_split, UBFCRPPGDataset
from rppg_lab.losses import hr_quality_training_loss, hr_quality_selection_training_loss
from rppg_lab.metrics import regression_metrics
from rppg_lab.models import HRQualityNet, HRQualityNetV2
from rppg_lab.signals import estimate_hr_welch, standardize_channels, standardize_1d

# Reuse the rPPG-10 loader that was already validated with ECG R-peak targets.
from scripts.benchmark_classical_rppg10 import (
    find_subject_dirs as find_rppg10_subject_dirs,
    load_rppg10_subject,
    sliding_windows as rppg10_sliding_windows,
    estimate_hr_from_ecg_rpeaks,
)


class RPPG10WindowDataset(Dataset):
    """rPPG-10 cropped-ROI window dataset with the same x/y interface as UBFC."""

    def __init__(
        self,
        subject_dirs: Sequence[Path],
        root: str | Path,
        fs_target: float = 30.0,
        win_sec: float = 30.0,
        stride_sec: float = 10.0,
        roi: str = "face",
        prior_methods: Sequence[str] = DEFAULT_PRIORS,
    ) -> None:
        self.subject_dirs = list(subject_dirs)
        self.root = Path(root)
        self.fs_target = float(fs_target)
        self.win_sec = float(win_sec)
        self.stride_sec = float(stride_sec)
        self.roi = roi
        self.prior_methods = list(prior_methods)
        self.records: List[Dict] = []
        self.samples: List[Tuple[int, int, int]] = []
        self._load_all()

    def _load_all(self) -> None:
        for subject_dir in self.subject_dirs:
            try:
                rgb, ecg, fs, fs_ecg, fps_reported, n_raw = load_rppg10_subject(
                    self.root, subject_dir, self.roi, self.fs_target
                )
            except Exception as e:
                print(f"Skipping {subject_dir.name}: {e}")
                continue

            n = len(rgb)
            priors = []
            for name in self.prior_methods:
                if name not in METHOD_FUNCS:
                    raise ValueError(f"Unknown prior method {name}. Available: {sorted(METHOD_FUNCS)}")
                try:
                    priors.append(METHOD_FUNCS[name](rgb, self.fs_target).astype(np.float32))
                except Exception:
                    priors.append(np.zeros(n, dtype=np.float32))

            rec_idx = len(self.records)
            self.records.append({
                "subject_id": subject_dir.name,
                "rgb": rgb.astype(np.float32),
                "ecg": ecg.astype(np.float64),
                "fs_ecg": float(fs_ecg),
                "priors": priors,
                "fps_reported": float(fps_reported),
                "n_raw": int(n_raw),
            })

            for s, e in rppg10_sliding_windows(n, self.fs_target, self.win_sec, self.stride_sec):
                self.samples.append((rec_idx, s, e))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        rec_idx, s, e = self.samples[idx]
        rec = self.records[rec_idx]

        rgb = standardize_channels(rec["rgb"][s:e])
        channels = [rgb.T]
        for p in rec["priors"]:
            channels.append(standardize_1d(p[s:e])[None, :])
        x = np.concatenate(channels, axis=0).astype(np.float32)

        start_sec = s / self.fs_target
        end_sec = e / self.fs_target
        hr = estimate_hr_from_ecg_rpeaks(rec["ecg"], rec["fs_ecg"], start_sec, end_sec)

        if not np.isfinite(hr):
            # Rare fallback.  The training script will skip NaN rows at loss time
            # if this happens, but keeping a sane value avoids collate issues.
            hr = np.nan

        return {
            "x": torch.from_numpy(x),
            "y_hr": torch.tensor(float(hr), dtype=torch.float32),
            "subject_id": rec["subject_id"],
            "dataset": "rPPG10",
        }


class HRQualityWrapper(Dataset):
    """Adds model-independent quality target and input-baseline diagnostics.

    The quality target is supervised from the best available input channel in the
    current window.  It answers: does this window contain a usable pulse cue in
    at least one RGB/prior channel?  This is not available at inference time, but
    it is a useful supervised target for learning confidence.
    """

    def __init__(self, base: Dataset, fs: float, channel_names: Sequence[str], quality_tau_bpm: float = 5.0):
        self.base = base
        self.fs = float(fs)
        self.channel_names = list(channel_names)
        self.quality_tau_bpm = float(quality_tau_bpm)
        # Candidate channels for explicit selection: RGB_G + all classical priors.
        # R/B are kept as temporal inputs but not as trusted HR candidates.
        self.candidate_idx = ([1] if len(self.channel_names) > 1 else []) + list(range(3, len(self.channel_names)))
        self.candidate_names = [self.channel_names[i] for i in self.candidate_idx]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        item = self.base[idx]
        x = item["x"]
        y_hr = float(item["y_hr"])

        cand_hrs = []
        for ch in self.candidate_idx:
            try:
                cand_hrs.append(estimate_hr_welch(x[ch].numpy(), self.fs).hr_bpm)
            except Exception:
                cand_hrs.append(np.nan)
        cand_hrs = np.asarray(cand_hrs, dtype=float)
        cand_valid = np.isfinite(cand_hrs)

        if np.isfinite(y_hr) and np.any(cand_valid):
            errs = np.abs(cand_hrs - y_hr)
            errs[~cand_valid] = np.inf
            best_idx = int(np.argmin(errs))
            best_err = float(errs[best_idx])
        else:
            best_idx = -1
            best_err = float("inf")

        # Smooth target: 1 when best candidate is very close, near 0 when no
        # candidate is close.  tau=5 bpm means quality ~=0.37 at 5 bpm error.
        q = float(np.exp(-((best_err / self.quality_tau_bpm) ** 2))) if np.isfinite(best_err) else 0.0

        cand_hrs_filled = np.nan_to_num(cand_hrs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        cand_valid = cand_valid.astype(np.bool_)

        # Return only the fields needed by HRQualityNet/HRQualityNetV2.
        return {
            "x": x,
            "y_hr": item["y_hr"],
            "quality_target": torch.tensor(q, dtype=torch.float32),
            "best_input_err": torch.tensor(best_err if np.isfinite(best_err) else 999.0, dtype=torch.float32),
            "candidate_hrs": torch.from_numpy(cand_hrs_filled),
            "candidate_valid": torch.from_numpy(cand_valid),
            "best_candidate_index": torch.tensor(best_idx, dtype=torch.long),
            "subject_id": item.get("subject_id", ""),
            "dataset": item.get("dataset", "UBFC"),
        }


def split_paths(paths: Sequence[Path], val_fraction: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(paths) * val_fraction)))
    val_idx = set(idx[:n_val].tolist())
    train, val = [], []
    for i, p in enumerate(paths):
        (val if i in val_idx else train).append(p)
    return train, val


def build_datasets(args, prior_names: Sequence[str], channel_names: Sequence[str]):
    train_sets = []
    val_sets = []

    if args.ubfc_root:
        subjects = find_ubfc_subjects(args.ubfc_root)
        tr, va = subject_split(subjects, val_fraction=args.val_fraction, seed=args.seed)
        print("UBFC train subjects:", [s.subject_id for s in tr])
        print("UBFC val subjects:", [s.subject_id for s in va])

        train_sets.append(HRQualityWrapper(
            UBFCRPPGDataset(tr, fs_target=args.fs, win_sec=args.win_sec, stride_sec=args.stride_sec,
                            roi=args.roi, cache_dir=args.cache_dir, prior_methods=prior_names),
            fs=args.fs,
            channel_names=channel_names,
            quality_tau_bpm=args.quality_tau_bpm,
        ))
        val_sets.append(HRQualityWrapper(
            UBFCRPPGDataset(va, fs_target=args.fs, win_sec=args.win_sec, stride_sec=args.stride_sec,
                            roi=args.roi, cache_dir=args.cache_dir, prior_methods=prior_names),
            fs=args.fs,
            channel_names=channel_names,
            quality_tau_bpm=args.quality_tau_bpm,
        ))

    if args.rppg10_root:
        root = Path(args.rppg10_root)
        dirs = find_rppg10_subject_dirs(root)
        tr, va = split_paths(dirs, val_fraction=args.val_fraction, seed=args.seed)
        print("rPPG-10 train subjects:", [p.name for p in tr])
        print("rPPG-10 val subjects:", [p.name for p in va])

        # rPPG-10 videos are 10 minutes; keep its own window defaults unless
        # user explicitly overrides with --rppg10-win-sec/--rppg10-stride-sec.
        train_sets.append(HRQualityWrapper(
            RPPG10WindowDataset(tr, root=root, fs_target=args.fs, win_sec=args.rppg10_win_sec,
                                stride_sec=args.rppg10_stride_sec, roi=args.rppg10_roi, prior_methods=prior_names),
            fs=args.fs,
            channel_names=channel_names,
            quality_tau_bpm=args.quality_tau_bpm,
        ))
        val_sets.append(HRQualityWrapper(
            RPPG10WindowDataset(va, root=root, fs_target=args.fs, win_sec=args.rppg10_win_sec,
                                stride_sec=args.rppg10_stride_sec, roi=args.rppg10_roi, prior_methods=prior_names),
            fs=args.fs,
            channel_names=channel_names,
            quality_tau_bpm=args.quality_tau_bpm,
        ))

    if not train_sets or not val_sets:
        raise RuntimeError("Provide at least --ubfc-root or --rppg10-root.")

    train_ds = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val_ds = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
    return train_ds, val_ds


def safe_metrics(gt: Sequence[float], pred: Sequence[float]):
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    keep = np.isfinite(gt) & np.isfinite(pred)
    return regression_metrics(gt[keep], pred[keep])


def evaluate_channel_baselines(ds: Dataset, channel_names: Sequence[str], fs: float):
    gt = []
    preds = {name: [] for name in channel_names}
    best_input = []
    qtarget = []

    for i in range(len(ds)):
        item = ds[i]
        x = item["x"].numpy()
        y = float(item["y_hr"])
        if not np.isfinite(y):
            continue
        gt.append(y)
        for ch, name in enumerate(channel_names):
            preds[name].append(estimate_hr_welch(x[ch], fs).hr_bpm)
        best_input.append(float(item["best_input_err"]))
        qtarget.append(float(item["quality_target"]))

    rows = []
    for name, pr in preds.items():
        m = safe_metrics(gt, pr)
        rows.append((name, m.mae, m.rmse, m.me, m.sd, m.pearson))
    rows.sort(key=lambda r: r[1])

    best_input = np.asarray(best_input, dtype=float)
    print(f"Best input-channel oracle MAE={np.nanmean(best_input):.3f} median={np.nanmedian(best_input):.3f}")
    print(f"Quality target mean={np.nanmean(qtarget):.3f}")
    return rows


def evaluate_model(model, loader: DataLoader, device: torch.device):
    model.eval()
    gt, pred, qual, qtar, datasets = [], [], [], [], []
    sel_correct, sel_total = 0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y_hr"].cpu().numpy().astype(float)
            if hasattr(model, "num_candidates") and "candidate_hrs" in batch:
                cand = batch["candidate_hrs"].to(device)
                cvalid = batch["candidate_valid"].to(device)
                out = model(x, candidate_hrs=cand, candidate_valid=cvalid, return_dict=True)
                if "selection_probs" in out and "best_candidate_index" in batch:
                    best_idx = batch["best_candidate_index"].to(device)
                    valid_sel = best_idx >= 0
                    if valid_sel.any():
                        pred_idx = out["selection_probs"].argmax(dim=1)
                        sel_correct += int((pred_idx[valid_sel] == best_idx[valid_sel]).sum().item())
                        sel_total += int(valid_sel.sum().item())
            else:
                out = model(x, return_dict=True)
            p = out["hr_bpm"].cpu().numpy().astype(float)
            q = out["quality"].cpu().numpy().astype(float)
            qt = batch["quality_target"].cpu().numpy().astype(float)
            ds_names = batch.get("dataset", ["UBFC"] * len(y))
            if isinstance(ds_names, str):
                ds_names = [ds_names] * len(y)
            for yi, pi, qi, qti, dsi in zip(y, p, q, qt, ds_names):
                if np.isfinite(yi) and np.isfinite(pi):
                    gt.append(yi); pred.append(pi); qual.append(qi); qtar.append(qti); datasets.append(str(dsi))

    m = safe_metrics(gt, pred)
    abs_err = np.abs(np.asarray(pred) - np.asarray(gt))
    qual = np.asarray(qual, dtype=float)
    qtar = np.asarray(qtar, dtype=float)
    datasets = np.asarray(datasets)

    out = {
        "metrics": m,
        "abs_err": abs_err,
        "quality": qual,
        "quality_target": qtar,
        "datasets": datasets,
        "gt": np.asarray(gt),
        "pred": np.asarray(pred),
        "selection_acc": (sel_correct / sel_total) if sel_total > 0 else float("nan"),
    }
    return out


def print_eval_summary(ev: Dict, prefix: str = "val"):
    m = ev["metrics"]
    abs_err = ev["abs_err"]
    qual = ev["quality"]
    print(f"{prefix}_HR_MAE={m.mae:.3f} {prefix}_HR_RMSE={m.rmse:.3f} {prefix}_HR_r={m.pearson:.3f}")
    if np.isfinite(ev.get("selection_acc", float("nan"))):
        print(f"{prefix}_selection_acc={ev['selection_acc']:.3f}")
    if len(abs_err) and np.std(qual) > 1e-8:
        corr = np.corrcoef(qual, abs_err)[0, 1]
        print(f"{prefix}_quality_error_corr={corr:.3f}  quality_mean={np.mean(qual):.3f}")
    for cov in [1.0, 0.8, 0.6, 0.4, 0.2]:
        if len(abs_err) == 0:
            continue
        n = max(1, int(round(cov * len(abs_err))))
        idx = np.argsort(-qual)[:n]
        print(f"  top_quality_coverage={100*cov:5.1f}% N={n:4d} MAE={abs_err[idx].mean():7.3f} median={np.median(abs_err[idx]):7.3f}")

    for ds in sorted(set(ev["datasets"].tolist())):
        mask = ev["datasets"] == ds
        if mask.sum() < 2:
            continue
        mm = safe_metrics(ev["gt"][mask], ev["pred"][mask])
        print(f"  {ds:8s}: N={mask.sum():4d} MAE={mm.mae:7.3f} RMSE={mm.rmse:7.3f} r={mm.pearson:7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ubfc-root", default=None)
    ap.add_argument("--rppg10-root", default=None)
    ap.add_argument("--cache-dir", default="cache_roi")
    ap.add_argument("--out", default="checkpoints/hr_quality_net.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--win-sec", type=float, default=10.0)
    ap.add_argument("--stride-sec", type=float, default=2.0)
    ap.add_argument("--roi", default="face")
    ap.add_argument("--rppg10-win-sec", type=float, default=30.0)
    ap.add_argument("--rppg10-stride-sec", type=float, default=10.0)
    ap.add_argument("--rppg10-roi", default="face", choices=["face", "forehead", "cheek1", "cheek2"])
    ap.add_argument("--priors", default=",".join(DEFAULT_PRIORS))
    ap.add_argument("--no-priors", action="store_true")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--quality-tau-bpm", type=float, default=5.0)
    ap.add_argument("--sigma-bpm", type=float, default=3.0)
    ap.add_argument("--quality-weight", type=float, default=0.20)
    ap.add_argument("--model-version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--selection-weight", type=float, default=0.50)
    ap.add_argument("--hr-reg-weight", type=float, default=0.10)
    ap.add_argument("--dist-weight", type=float, default=1.00)
    ap.add_argument("--patience", type=int, default=12)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prior_names = [] if args.no_priors else [p.strip() for p in args.priors.split(",") if p.strip()]
    channel_names = ["RGB_R", "RGB_G", "RGB_B"] + prior_names
    print("Using priors:", prior_names)

    train_ds, val_ds = build_datasets(args, prior_names, channel_names)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print("Input channels:", len(channel_names))

    print("\nValidation input-channel baselines:")
    for name, mae, rmse, me, sd, r in evaluate_channel_baselines(val_ds, channel_names, args.fs):
        print(f"  {name:16s} MAE={mae:7.3f} RMSE={rmse:7.3f} ME={me:7.3f} SD={sd:7.3f} r={r:7.3f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Candidate HRs are RGB_G + all prior channels.
    num_candidates = (1 if len(channel_names) > 1 else 0) + max(0, len(channel_names) - 3)
    if args.model_version == "v2":
        model = HRQualityNetV2(in_channels=len(channel_names), num_candidates=num_candidates).to(device)
    else:
        model = HRQualityNet(in_channels=len(channel_names)).to(device)
    print(f"Model version: {args.model_version} | num_candidates={num_candidates}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = np.inf
    bad_epochs = 0

    print("\nInitial evaluation:")
    init_ev = evaluate_model(model, val_loader, device)
    print_eval_summary(init_ev, prefix="val")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = batch["x"].to(device)
            y_hr = batch["y_hr"].to(device)
            qtar = batch["quality_target"].to(device)

            finite = torch.isfinite(y_hr)
            if finite.sum() == 0:
                continue
            x = x[finite]
            y_hr = y_hr[finite]
            qtar = qtar[finite]

            if args.model_version == "v2":
                cand = batch["candidate_hrs"].to(device)[finite]
                cvalid = batch["candidate_valid"].to(device)[finite]
                best_idx = batch["best_candidate_index"].to(device)[finite]
                out = model(x, candidate_hrs=cand, candidate_valid=cvalid, return_dict=True)
                loss = hr_quality_selection_training_loss(
                    hr_logits=out["hr_logits"],
                    hr_bpm=y_hr,
                    hr_bins=model.hr_bins,
                    final_hr_bpm=out["hr_bpm"],
                    selection_logits=out["selection_logits"],
                    best_candidate_index=best_idx,
                    quality_logit=out["quality_logit"],
                    quality_target=qtar,
                    sigma_bpm=args.sigma_bpm,
                    dist_weight=args.dist_weight,
                    hr_reg_weight=args.hr_reg_weight,
                    selection_weight=args.selection_weight,
                    quality_weight=args.quality_weight,
                )
            else:
                out = model(x, return_dict=True)
                loss = hr_quality_training_loss(
                    hr_logits=out["hr_logits"],
                    hr_bpm=y_hr,
                    hr_bins=model.hr_bins,
                    quality_logit=out["quality_logit"],
                    quality_target=qtar,
                    sigma_bpm=args.sigma_bpm,
                    quality_weight=args.quality_weight,
                )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))

        ev = evaluate_model(model, val_loader, device)
        m = ev["metrics"]
        print(f"epoch={epoch} train_loss={np.mean(losses):.4f} val_HR_MAE={m.mae:.3f} val_HR_RMSE={m.rmse:.3f} val_HR_r={m.pearson:.3f}")
        print_eval_summary(ev, prefix="val")

        if m.mae < best - 1e-4:
            best = m.mae
            bad_epochs = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_class": "HRQualityNetV2" if args.model_version == "v2" else "HRQualityNet",
                "model_version": args.model_version,
                "in_channels": len(channel_names),
                "channel_names": channel_names,
                "candidate_names": (["RGB_G"] if len(channel_names) > 1 else []) + prior_names,
                "priors": prior_names,
                "fs": args.fs,
                "hr_bins": model.hr_bins.detach().cpu().numpy(),
                "args": vars(args),
                "best_val_hr_mae": best,
            }, out_path)
            print("saved best:", out_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs. Best val_HR_MAE={best:.3f}")
                break


if __name__ == "__main__":
    main()
