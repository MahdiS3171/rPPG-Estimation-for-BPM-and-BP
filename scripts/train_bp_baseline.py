#!/usr/bin/env python
"""Train a feature-level BP baseline with participant-wise train/val/test splits.

This script is intended for the group's future synchronized face--hand pilot
recordings. It does not establish that blood pressure can be inferred from face
video alone. Participant partitioning is completed before imputation,
standardization, or model fitting. Feature and target normalization statistics
are estimated from training participants only. The locked test split is evaluated
once after validation-based model selection.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.datasets import SessionBPDataset
from rppg_lab.models import BPFeatureMLP
from rppg_lab.metrics import aami_summary, regression_metrics
from rppg_lab.splits import split_subjects


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_feature_preprocessor(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return training-set feature medians, means, and standard deviations."""
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_imp = np.where(np.isfinite(x_train), x_train, med[None, :])
    mean = x_imp.mean(axis=0)
    std = x_imp.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return med.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def transform_features(x: np.ndarray, med: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_imp = np.where(np.isfinite(x), x, med[None, :])
    return ((x_imp - mean[None, :]) / std[None, :]).astype(np.float32)


def fit_target_preprocessor(y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(y_train, axis=0)
    std = np.std(y_train, axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_targets(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((y - mean[None, :]) / std[None, :]).astype(np.float32)


def inverse_targets(y_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return y_scaled * std[None, :] + mean[None, :]


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y_scaled in loader:
            pred_scaled = model(x.to(device)).cpu().numpy()
            ys.append(inverse_targets(y_scaled.numpy(), target_mean, target_std))
            ps.append(inverse_targets(pred_scaled, target_mean, target_std))
    return np.vstack(ys), np.vstack(ps)


def print_metrics(prefix: str, y: np.ndarray, p: np.ndarray) -> None:
    sbp = regression_metrics(y[:, 0], p[:, 0])
    dbp = regression_metrics(y[:, 1], p[:, 1])
    print(
        f"{prefix}: "
        f"SBP MAE={sbp.mae:.2f} RMSE={sbp.rmse:.2f} ME={sbp.me:.2f} SD={sbp.sd:.2f} r={sbp.pearson:.3f} | "
        f"DBP MAE={dbp.mae:.2f} RMSE={dbp.rmse:.2f} ME={dbp.me:.2f} SD={dbp.sd:.2f} r={dbp.pearson:.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="checkpoints/bp_feature_mlp.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    ds = SessionBPDataset(args.data_dir, target="both")
    train_subjects, val_subjects, test_subjects = split_subjects(
        ds.subject_ids, args.val_fraction, args.test_fraction, args.seed
    )
    train_set, val_set, test_set = set(train_subjects), set(val_subjects), set(test_subjects)
    train_idx = [i for i, sid in enumerate(ds.subject_ids) if sid in train_set]
    val_idx = [i for i, sid in enumerate(ds.subject_ids) if sid in val_set]
    test_idx = [i for i, sid in enumerate(ds.subject_ids) if sid in test_set]
    if not train_idx or not val_idx or (args.test_fraction > 0 and not test_idx):
        raise RuntimeError("Participant split produced an empty requested partition.")

    # Discover the legacy JSON feature schema from training rows only, so a
    # feature present exclusively in validation/test cannot alter model inputs.
    ds.feature_keys = sorted({k for i in train_idx for k,v in ds.rows[i]["features"].items()
                             if isinstance(v, (int,float)) and not isinstance(v,bool)})
    if not ds.feature_keys:
        raise RuntimeError("No numeric features in training subjects")
    print("Legacy cuff timing is unchecked; use run_bp_pilot.py and train_bp_models.py for provenance-aware research.")
    print("Dataset exclusions:", ds.exclusions)

    x_raw = np.stack([ds[i]["x_raw"].numpy() for i in range(len(ds))])
    y_raw = np.stack([ds[i]["y"].numpy() for i in range(len(ds))]).astype(np.float32)

    feat_med, feat_mean, feat_std = fit_feature_preprocessor(x_raw[train_idx])
    x_all = transform_features(x_raw, feat_med, feat_mean, feat_std)
    target_mean, target_std = fit_target_preprocessor(y_raw[train_idx])
    y_all = transform_targets(y_raw, target_mean, target_std)

    def make_loader(indices: list[int], shuffle: bool) -> DataLoader:
        x = torch.from_numpy(x_all[indices])
        y = torch.from_numpy(y_all[indices])
        return DataLoader(
            TensorDataset(x, y),
            batch_size=min(args.batch_size, len(indices)),
            shuffle=shuffle,
        )

    tr = make_loader(train_idx, True)
    va = make_loader(val_idx, False)
    te = make_loader(test_idx, False) if test_idx else None

    print("Train subjects:", train_subjects)
    print("Validation subjects:", val_subjects)
    print("Locked test subjects:", test_subjects)
    print(
        f"Train sessions={len(train_idx)} | validation sessions={len(val_idx)} | "
        f"test sessions={len(test_idx)} | features={len(ds.feature_keys)}"
    )

    # Transparent non-learned reference: the training-cohort mean BP.
    mean_bp = y_raw[train_idx].mean(axis=0)
    val_mean_pred = np.repeat(mean_bp[None, :], len(val_idx), axis=0)
    print_metrics("Validation population-mean baseline", y_raw[val_idx], val_mean_pred)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPFeatureMLP(
        in_features=len(ds.feature_keys),
        hidden=args.hidden,
        dropout=args.dropout,
        out_dim=2,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    best = np.inf
    bad_epochs = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))

        y_val, p_val = evaluate(model, va, device, target_mean, target_std)
        sbp_mae = float(np.mean(np.abs(p_val[:, 0] - y_val[:, 0])))
        dbp_mae = float(np.mean(np.abs(p_val[:, 1] - y_val[:, 1])))
        score = 0.5 * (sbp_mae + dbp_mae)
        if ep == 1 or ep % 10 == 0:
            print(f"epoch={ep} train_scaled_mse={np.mean(losses):.3f} val_mean_MAE={score:.3f}")
            print_metrics("  validation", y_val, p_val)

        if score < best - 1e-4:
            best = score
            bad_epochs = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_class": "BPFeatureMLP",
                "model_config": {
                    "in_features": len(ds.feature_keys),
                    "hidden": args.hidden,
                    "dropout": args.dropout,
                    "out_dim": 2,
                },
                "feature_keys": ds.feature_keys,
                "impute_median": feat_med,
                "feature_mean": feat_mean,
                "feature_std": feat_std,
                "target_mean": target_mean,
                "target_std": target_std,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects,
                "population_mean_bp": mean_bp.astype(np.float32),
                "best_val_mean_mae": best,
                "epoch": ep,
                "args": vars(args),
            }, out_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {args.patience} epochs without validation improvement.")
                break

    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    y_val, p_val = evaluate(model, va, device, target_mean, target_std)
    print_metrics("Best validation model", y_val, p_val)

    if te is not None:
        y_test, p_test = evaluate(model, te, device, target_mean, target_std)
        test_mean_pred = np.repeat(mean_bp[None, :], len(test_idx), axis=0)
        print_metrics("Locked test population-mean baseline", y_raw[test_idx], test_mean_pred)
        print_metrics("Locked test model", y_test, p_test)
        print("Test SBP agreement sanity summary:", aami_summary(y_test[:, 0], p_test[:, 0]))
        print("Test DBP agreement sanity summary:", aami_summary(y_test[:, 1], p_test[:, 1]))
    else:
        print("No test split requested; only validation metrics were produced.")

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
