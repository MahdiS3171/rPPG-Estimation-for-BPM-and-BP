#!/usr/bin/env python
"""Phase 2 architecture smoke training. Loss and HR selection are PROVISIONAL.

This does not freeze Waveform v1 or establish morphology, BP or hand accuracy.
The existing rppg_training_loss is reused unchanged pending Part 3.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.artifacts import provenance, write_json
from rppg_lab.classical import DEFAULT_PRIORS
from rppg_lab.config import PipelineConfig
from rppg_lab.datasets import UBFCMultiROIRPPGDataset, find_ubfc_subjects, subject_split
from rppg_lab.extraction_cache import DEFAULT_WAVEFORM_ROIS, EXTRACTION_CACHE_VERSION
from rppg_lab.losses import rppg_training_loss
from rppg_lab.metrics import regression_metrics, waveform_corr, waveform_corr_aligned
from rppg_lab.models import MultiROIPriorResidualWaveformNet
from rppg_lab.signals import estimate_hr_welch

LOSS_NAME = "rppg_training_loss (PROVISIONAL Phase 2)"


def forward_batch(model: MultiROIPriorResidualWaveformNet, batch: dict, device: torch.device) -> dict:
    return model(batch["x"].to(device), roi_quality=batch["roi_quality"].to(device),
                 roi_valid=batch["roi_valid"].to(device), prior_valid=batch["prior_valid"].to(device),
                 return_dict=True)


def evaluate_model(model: MultiROIPriorResidualWaveformNet, loader: DataLoader,
                   device: torch.device, fs: float) -> dict:
    """Window metrics and descriptive weights; attention is not causal evidence.

    ROI attention means include invalid windows (zero attention). Prior means
    are conditional on that ROI being valid; an entirely absent ROI reports
    None rather than a fabricated probability. Invalid priors contribute zero.
    """
    model.eval()
    cors, aligned, gt_hr, pred_hr = [], [], [], []
    attention = np.zeros(model.num_rois, dtype=np.float64)
    weights = np.zeros((model.num_rois, model.num_priors), dtype=np.float64)
    valid_count = np.zeros(model.num_rois, dtype=np.float64)
    residual_max, windows = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            out = forward_batch(model, batch, device)
            pred = out["ppg"].cpu().numpy()
            target = batch["y_ppg"].cpu().numpy()
            valid = batch["roi_valid"].cpu().numpy()
            windows += len(pred)
            attention += out["roi_attention"].cpu().numpy().sum(axis=0)
            weights += out["prior_weights"].cpu().numpy().sum(axis=0)
            valid_count += valid.sum(axis=0)
            residual_max = max(residual_max, float((model.residual_scale * out["residuals"]).abs().max()))
            for i in range(len(pred)):
                cors.append(waveform_corr(target[i], pred[i]))
                aligned.append(waveform_corr_aligned(target[i], pred[i], max_lag=int(0.5 * fs)))
                gt_hr.append(float(batch["y_hr"][i]))
                pred_hr.append(estimate_hr_welch(pred[i], fs).hr_bpm)
    if not windows:
        raise ValueError("Evaluation loader has no usable windows")

    def mean_finite(values: list[float]) -> float:
        arr = np.asarray(values)
        return float(arr[np.isfinite(arr)].mean()) if np.isfinite(arr).any() else float("nan")

    return dict(
        windows=windows, hr_metrics=asdict(regression_metrics(gt_hr, pred_hr)),
        wave_corr=mean_finite(cors), wave_corr_aligned=mean_finite(aligned),
        roi_attention_mean={name: float(attention[r] / windows) for r, name in enumerate(model.roi_names)},
        roi_valid_fraction={name: float(valid_count[r] / windows) for r, name in enumerate(model.roi_names)},
        prior_weights_mean={name: {prior: float(weights[r, k] / valid_count[r]) if valid_count[r] else None
                                   for k, prior in enumerate(model.prior_names)}
                            for r, name in enumerate(model.roi_names)},
        residual_contribution_max_abs=residual_max,
    )


def print_evaluation(epoch: int, train_loss: float, metrics: dict) -> None:
    print(f"epoch={epoch} train_loss={train_loss:.4f} val_HR_MAE={metrics['hr_metrics']['mae']:.3f} "
          f"val_wave_corr={metrics['wave_corr']:.3f} val_wave_corr_aligned={metrics['wave_corr_aligned']:.3f} "
          f"residual_contribution_max_abs={metrics['residual_contribution_max_abs']:.8g}")
    print("Mean ROI attention:", metrics["roi_attention_mean"])
    print("ROI valid-window fraction:", metrics["roi_valid_fraction"])
    for roi, weights in metrics["prior_weights_mean"].items():
        print(f"Mean prior weights ({roi}, conditional on valid ROI):", weights)


def save_checkpoint(path: Path, model: MultiROIPriorResidualWaveformNet, optimizer: torch.optim.Optimizer,
                    metadata: dict, epoch: int, metrics: dict, criterion: str) -> None:
    torch.save({**metadata, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "validation_metrics": metrics, "checkpoint_selection_criterion": criterion}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ubfc-root", required=True)
    parser.add_argument("--cache-dir", default="cache_roi_multi_phase1")
    parser.add_argument("--out-dir", default="checkpoints/waveform_multi_roi_phase2")
    parser.add_argument("--config", help="Optional Phase 1 JSON; dataset forces face-only and requested ROIs")
    parser.add_argument("--rois", default=",".join(DEFAULT_WAVEFORM_ROIS))
    parser.add_argument("--priors", default=",".join(DEFAULT_PRIORS))
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument("--win-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=2.0)
    parser.add_argument("--min-valid-fraction", type=float, help="Defaults to extraction config (0.9)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--init-prior", default="auto")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-best-hr", action="store_true", help="Save LAST only; HR selection is provisional")
    parser.add_argument("--max-frames", type=int, help="Extraction limit for smoke runs (part of cache identity)")
    parser.add_argument("--max-subjects", type=int, help="Use first N discovered subjects for a small smoke run")
    args = parser.parse_args()
    if args.epochs < 0 or args.batch_size < 1:
        parser.error("epochs must be nonnegative and batch-size positive")
    if args.max_subjects is not None and args.max_subjects < 2:
        parser.error("max-subjects must be at least two for a participant split")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = find_ubfc_subjects(args.ubfc_root)
    if args.max_subjects is not None:
        subjects = subjects[:args.max_subjects]
    train_subjects, val_subjects = subject_split(subjects, val_fraction=0.2, seed=args.seed)
    print("PROVISIONAL Phase 2 architecture training; loss and HR-based selection await Part 3.")
    print("Loss:", LOSS_NAME)
    print("Train subjects:", [s.subject_id for s in train_subjects])
    print("Validation subjects:", [s.subject_id for s in val_subjects])
    roi_names = tuple(name.strip() for name in args.rois.split(",") if name.strip())
    prior_names = tuple(name.strip() for name in args.priors.split(",") if name.strip())
    config = PipelineConfig.load(args.config) if args.config else PipelineConfig(sample_rate=args.fs, seed=args.seed)
    dataset_args = dict(fs_target=args.fs, win_sec=args.win_sec, stride_sec=args.stride_sec,
                        roi_names=roi_names, prior_methods=prior_names, cache_dir=args.cache_dir,
                        extraction_config=config, min_valid_fraction=args.min_valid_fraction, max_frames=args.max_frames)
    train_ds = UBFCMultiROIRPPGDataset(train_subjects, **dataset_args)
    val_ds = UBFCMultiROIRPPGDataset(val_subjects, **dataset_args)
    if not len(train_ds) or not len(val_ds):
        raise ValueError(f"Need usable train/validation windows; got {len(train_ds)}/{len(val_ds)}. Check extraction/reference support.")
    print(f"Train windows={len(train_ds)} validation windows={len(val_ds)} device={device}")
    print("ROI order:", train_ds.roi_names, "channel order:", train_ds.channel_names)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = MultiROIPriorResidualWaveformNet(len(train_ds.channel_names), prior_names, roi_names,
        base_channels=args.base_channels, num_blocks=args.num_blocks, dropout=args.dropout,
        residual_scale=args.residual_scale, init_prior=args.init_prior).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        phase="Phase 2 PROVISIONAL", model_class=model.__class__.__name__, model_config=model.model_config,
        roi_names=train_ds.roi_names, prior_names=train_ds.prior_names, channel_names=train_ds.channel_names,
        channel_ordering=train_ds.channel_names, extraction_config=train_ds.extraction_config.to_dict(),
        extraction_cache_version=EXTRACTION_CACHE_VERSION, fs=args.fs, win_sec=args.win_sec,
        window_length_samples=round(args.fs * args.win_sec), stride_sec=args.stride_sec,
        min_valid_fraction=train_ds.min_valid_fraction, seed=args.seed,
        reference_diagnostics={"train": train_ds.reference_diagnostics, "val": val_ds.reference_diagnostics},
        train_subjects=[s.subject_id for s in train_subjects], val_subjects=[s.subject_id for s in val_subjects],
        loss_name=LOSS_NAME, args=vars(args), provenance=provenance(),
    )
    last = output / "waveform_multi_roi_last.pt"
    best_path = output / "waveform_multi_roi_best_hr_PROVISIONAL.pt"
    history = []
    best = float("inf")
    for epoch in range(args.epochs + 1):
        train_losses = []
        if epoch:
            model.train()
            for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
                out = forward_batch(model, batch, device)
                # Reuse the existing loss; its residual penalty sees valid ROI
                # residuals only. No morphology-specific loss is added here.
                loss = rppg_training_loss(out["ppg"], batch["y_ppg"].to(device), batch["y_hr"].to(device),
                    fs=args.fs, epoch=epoch, total_epochs=args.epochs,
                    residual=out["residuals"][batch["roi_valid"].to(device)])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite provisional training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                train_losses.append(float(loss.detach()))
        metrics = evaluate_model(model, val_loader, device, args.fs)
        if epoch == 0 and metrics["residual_contribution_max_abs"] != 0.0:
            raise RuntimeError("Conservative epoch-0 check failed: residual contribution is not zero")
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        print_evaluation(epoch, train_loss, metrics)
        save_checkpoint(last, model, optimizer, metadata, epoch, metrics, "LAST completed epoch (PROVISIONAL)")
        score = metrics["hr_metrics"]["mae"]
        if not args.no_best_hr and np.isfinite(score) and score < best:
            best = score
            save_checkpoint(best_path, model, optimizer, metadata, epoch, metrics,
                            "minimum validation HR MAE (PROVISIONAL; not final waveform selection)")
            print("Saved provisional best by HR:", best_path)
        history.append(dict(epoch=epoch, train_loss=train_loss, **metrics))
        write_json(output / "training_log_PROVISIONAL.json", {"metadata": metadata, "epochs": history,
                   "train_exclusions": train_ds.exclusions, "val_exclusions": val_ds.exclusions})
    print("Saved LAST:", last)


if __name__ == "__main__":
    main()
