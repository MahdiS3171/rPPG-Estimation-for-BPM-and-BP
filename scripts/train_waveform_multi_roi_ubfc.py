#!/usr/bin/env python
"""Train a morphology-oriented Waveform v1 candidate; locked test is never loaded."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.artifacts import provenance, write_json
from rppg_lab.classical import DEFAULT_PRIORS
from rppg_lab.config import PipelineConfig
from rppg_lab.datasets import UBFCMultiROIRPPGDataset, find_ubfc_subjects
from rppg_lab.extraction_cache import DEFAULT_WAVEFORM_ROIS, EXTRACTION_CACHE_VERSION
from rppg_lab.models import MultiROIPriorResidualWaveformNet
from rppg_lab.losses import (WaveformV1LossConfig, waveform_spectral_band, waveform_v1_training_loss,
                             rppg_training_loss, time_shifted_negative_pearson_loss,
                             spectral_shape_loss, hr_label_distribution_loss, smoothness_loss)
from rppg_lab.splits import waveform_v1_split_manifest, validate_waveform_split_manifest
from rppg_lab.waveform_metrics import forward_batch, evaluate_waveform_model
from rppg_lab.waveform_training import (WaveformCheckpointSelector, MORPHOLOGY_SCORE_FORMULA,
                                       HR_ELIGIBILITY_RULE, TIE_BREAKING_RULE, split_manifest_hash)

# Preserve Phase 2 public helper imports for analysis callers.
evaluate_model = evaluate_waveform_model
LOSS_NAME = "waveform_v1_training_loss (PROVISIONAL candidate_not_frozen)"


def training_components(out, batch, device, args, loss_config):
    pred, target, hr = out["ppg"], batch["y_ppg"].to(device), batch["y_hr"].to(device)
    valid = batch["roi_valid"].to(device)
    if args.objective == "waveform_v1":
        return waveform_v1_training_loss(pred, target, hr, args.fs, out["residuals"], valid,
                                         loss_config, return_components=True)
    residual = out["residuals"][valid]
    # Legacy function/defaults unchanged; diagnostics include its old smoothing.
    return dict(total=rppg_training_loss(pred, target, hr, args.fs, residual=residual),
                waveform_corr=time_shifted_negative_pearson_loss(pred, target, round(0.5 * args.fs)),
                derivative_corr=pred.new_tensor(float("nan")),
                spectral=spectral_shape_loss(pred, target, args.fs),
                hr=hr_label_distribution_loss(pred, hr, args.fs),
                residual=residual.square().mean(), smoothness_optional_legacy=smoothness_loss(pred))


def print_evaluation(epoch, train_components, metrics):
    print(f"epoch={epoch} train_components={train_components}")
    print("Window-weighted validation:", metrics["window_metrics"])
    print("Subject-balanced validation:", metrics["subject_balanced_metrics"])
    print(f"morphology_score={metrics['morphology_score']:.6f} HR_eligible={metrics['hr_eligible']} "
          f"HR_gate_limit_bpm={metrics['hr_gate_limit_bpm']:.6f}")
    print("Mean ROI attention:", metrics["roi_attention_mean"])
    print("ROI valid-window fraction:", metrics["roi_valid_fraction"])
    print("Valid windows per subject:", metrics["valid_windows_per_subject"])
    print("Mean prior weights conditional on valid ROI:", metrics["prior_weights_mean"])
    print("Residual contribution mean abs:", metrics["residual_contribution_mean_abs"],
          "max abs:", metrics["residual_contribution_max_abs"])


def save_checkpoint(path, model, optimizer, metadata, epoch, metrics, criterion):
    torch.save({**metadata, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "validation_metrics": metrics, "checkpoint_selection_criterion": criterion}, path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ubfc-root", required=True)
    parser.add_argument("--cache-dir", default="cache_roi_multi_phase1")
    parser.add_argument("--out-dir", default="checkpoints/waveform_multi_roi_v1_candidate")
    parser.add_argument("--config", help="Phase 1 JSON; dataset forces face-only and requested ROIs")
    parser.add_argument("--split-manifest", help="Reuse an existing locked participant split")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.20, help="Fraction of remaining non-test participants")
    parser.add_argument("--rois", default=",".join(DEFAULT_WAVEFORM_ROIS))
    parser.add_argument("--priors", default=",".join(DEFAULT_PRIORS))
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument("--win-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=2.0)
    parser.add_argument("--min-valid-fraction", type=float)
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
    parser.add_argument("--objective", choices=("waveform_v1", "phase2_rppg"), default="waveform_v1")
    for name, value in asdict(WaveformV1LossConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=value)
    parser.add_argument("--hr-regression-tolerance-bpm", type=float, default=1.0,
                        help="Engineering HR tolerance relative to epoch 0")
    parser.add_argument("--no-best-hr", action="store_true", help="Legacy flag: disable HR diagnostic checkpoint")
    parser.add_argument("--max-frames", type=int, help="Smoke limit; checkpointed and part of cache identity")
    parser.add_argument("--max-subjects", type=int, help="First N participants before partitioning; smoke only")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.epochs < 0 or args.batch_size < 1 or (args.max_subjects is not None and args.max_subjects < 2):
        parser.error("Require nonnegative epochs, positive batch size and max-subjects >= 2")
    try:
        loss_config = WaveformV1LossConfig(**{name: getattr(args, name) for name in asdict(WaveformV1LossConfig())})
        effective_band = waveform_spectral_band(args.fs, loss_config.spectral_fmin_hz, loss_config.spectral_fmax_hz)
        selector = WaveformCheckpointSelector(args.hr_regression_tolerance_bpm)
    except ValueError as error:
        parser.error(str(error))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = find_ubfc_subjects(args.ubfc_root)
    if args.max_subjects is not None:
        subjects = subjects[:args.max_subjects]
    ids = [s.subject_id for s in subjects]
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "waveform_v1_split.json"
    if args.split_manifest:
        manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
        validate_waveform_split_manifest(manifest)
        if sorted(ids) != manifest["all_subject_ids"]:
            raise ValueError("Discovered IDs do not match the supplied locked split")
    else:
        manifest = waveform_v1_split_manifest(ids, args.validation_fraction, args.test_fraction, args.seed)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if split_manifest_hash(previous) != split_manifest_hash(manifest):
            raise ValueError("Refusing to replace an existing locked split; use a separate run directory")
    write_json(manifest_path, manifest)
    if manifest["smoke_only"]:
        warnings.warn("Two-participant smoke only: no locked test possible; not a development cohort", RuntimeWarning)
    by_id = {s.subject_id: s for s in subjects}
    train_subjects = [by_id[s] for s in manifest["train_ids"]]
    val_subjects = [by_id[s] for s in manifest["validation_ids"]]
    print("Waveform v1 candidate training; quality head uncalibrated; test remains locked.")
    print("Objective:", args.objective, "loss config:", asdict(loss_config))
    print("Morphology score:", MORPHOLOGY_SCORE_FORMULA)
    print("HR eligibility:", HR_ELIGIBILITY_RULE, "tolerance:", selector.tolerance)
    print("Tie breaking:", TIE_BREAKING_RULE)
    print("Train IDs:", manifest["train_ids"], "Validation IDs:", manifest["validation_ids"],
          "Locked test IDs:", manifest["test_ids"])
    roi_names = tuple(name.strip() for name in args.rois.split(",") if name.strip())
    prior_names = tuple(name.strip() for name in args.priors.split(",") if name.strip())
    config = PipelineConfig.load(args.config) if args.config else PipelineConfig(sample_rate=args.fs, seed=args.seed)
    dataset_args = dict(fs_target=args.fs, win_sec=args.win_sec, stride_sec=args.stride_sec,
                        roi_names=roi_names, prior_methods=prior_names, cache_dir=args.cache_dir,
                        extraction_config=config, min_valid_fraction=args.min_valid_fraction, max_frames=args.max_frames)
    # Test IDs are metadata, never dataset input.
    train_ds = UBFCMultiROIRPPGDataset(train_subjects, **dataset_args)
    val_ds = UBFCMultiROIRPPGDataset(val_subjects, **dataset_args)
    if not len(train_ds) or not len(val_ds):
        raise ValueError(f"Need usable train/validation windows; got {len(train_ds)}/{len(val_ds)}")
    print(f"Train windows={len(train_ds)} validation windows={len(val_ds)} device={device}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = MultiROIPriorResidualWaveformNet(len(train_ds.channel_names), prior_names, roi_names,
        base_channels=args.base_channels, num_blocks=args.num_blocks, dropout=args.dropout,
        residual_scale=args.residual_scale, init_prior=args.init_prior).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    legacy_weights = dict(w_corr=0.5, w_d1=0.0, w_spec=0.1, w_hr=0.5, w_res=0.02, w_smooth=0.001)
    metadata = dict(
        checkpoint_schema_version="waveform_multi_roi_part3_v1", phase="Part 3 Waveform v1 candidate",
        waveform_v1_status="candidate_not_frozen", model_class=model.__class__.__name__, model_config=model.model_config,
        roi_names=train_ds.roi_names, prior_names=train_ds.prior_names, channel_names=train_ds.channel_names,
        channel_ordering=train_ds.channel_names, extraction_config=train_ds.extraction_config.to_dict(),
        extraction_cache_version=EXTRACTION_CACHE_VERSION, fs=args.fs, win_sec=args.win_sec,
        window_length_samples=round(args.fs * args.win_sec), stride_sec=args.stride_sec,
        min_valid_fraction=train_ds.min_valid_fraction, max_frames=args.max_frames, seed=args.seed,
        reference_diagnostics={"train": train_ds.reference_diagnostics, "val": val_ds.reference_diagnostics},
        train_subjects=manifest["train_ids"], val_subjects=manifest["validation_ids"], test_subjects=manifest["test_ids"],
        split_manifest_path=str(manifest_path.resolve()), split_manifest=manifest,
        split_manifest_sha256=split_manifest_hash(manifest), objective_name=args.objective,
        loss_name=LOSS_NAME if args.objective == "waveform_v1" else "rppg_training_loss (PROVISIONAL Phase 2 ablation)",
        loss_config=asdict(loss_config), loss_weights={k: v for k, v in asdict(loss_config).items() if k.startswith("w_")}
            if args.objective == "waveform_v1" else legacy_weights,
        objective_loss_config=asdict(loss_config) if args.objective == "waveform_v1" else dict(
            **legacy_weights, max_lag_sec=0.5, spectral_fmin_hz=0.7, spectral_fmax_hz=3.5,
            spectral_definition="legacy normalized PSD cross-entropy", derivative_internal_weight=0.0),
        max_lag_sec=loss_config.max_lag_sec, max_lag_samples=round(loss_config.max_lag_sec * args.fs),
        spectral_band_hz=effective_band, morphology_score_formula=MORPHOLOGY_SCORE_FORMULA,
        hr_eligibility_rule=HR_ELIGIBILITY_RULE, hr_regression_tolerance_bpm=selector.tolerance,
        selection_tie_breaking=TIE_BREAKING_RULE, quality_head_status="uncalibrated; not supervised or selected",
        args=vars(args), provenance=provenance())
    criteria = {"best_waveform_candidate.pt": "maximum subject-balanced morphology_score among HR-eligible epochs; " + TIE_BREAKING_RULE,
                "best_morphology_unconstrained.pt": "maximum subject-balanced morphology_score without HR gate; " + TIE_BREAKING_RULE,
                "best_hr_diagnostic.pt": "minimum subject-balanced validation HR MAE; diagnostic only",
                "last.pt": "LAST completed epoch; candidate_not_frozen"}
    history = []
    for epoch in range(args.epochs + 1):
        totals, train_windows = {}, 0
        if epoch:
            model.train()
            for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
                out = forward_batch(model, batch, device)
                components = training_components(out, batch, device, args, loss_config)
                loss = components["total"]
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite waveform training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                count = len(batch["x"])
                train_windows += count
                for key, value in components.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach()) * count
        components_mean = {k: totals[k] / train_windows if train_windows else float("nan")
                           for k in ("total", "waveform_corr", "derivative_corr", "spectral", "hr", "residual")}
        if "smoothness_optional_legacy" in totals:
            components_mean["smoothness_optional_legacy"] = totals["smoothness_optional_legacy"] / train_windows
        metrics = evaluate_model(model, val_loader, device, args.fs, loss_config.max_lag_sec,
                                 loss_config.spectral_fmin_hz, loss_config.spectral_fmax_hz)
        if epoch == 0 and metrics["residual_contribution_max_abs"] != 0.0:
            raise RuntimeError("Conservative epoch-0 check failed: residual contribution is not zero")
        decision = selector.consider(epoch, metrics)
        if epoch == 0:
            metadata["epoch0_reference_metrics"] = json.loads(json.dumps(selector.epoch0_reference_metrics))
        metrics.update(decision)
        print_evaluation(epoch, components_mean, metrics)
        save_checkpoint(output / "last.pt", model, optimizer, metadata, epoch, metrics, criteria["last.pt"])
        # Deprecated aliases retain Phase 2 analysis/test compatibility while
        # carrying explicit Part 3 metadata; best-HR is diagnostic only.
        save_checkpoint(output / "waveform_multi_roi_last.pt", model, optimizer, metadata, epoch, metrics, criteria["last.pt"])
        for filename in decision["updated_checkpoints"]:
            if filename == "best_hr_diagnostic.pt" and args.no_best_hr:
                continue
            save_checkpoint(output / filename, model, optimizer, metadata, epoch, metrics, criteria[filename])
            if filename == "best_hr_diagnostic.pt":
                save_checkpoint(output / "waveform_multi_roi_best_hr_PROVISIONAL.pt", model, optimizer,
                                metadata, epoch, metrics, criteria[filename])
        history.append(dict(epoch=epoch, train_loss=components_mean["total"], train_components=components_mean, **metrics))
        log = dict(metadata=metadata, epochs=history, train_exclusions=train_ds.exclusions, val_exclusions=val_ds.exclusions)
        write_json(output / "training_log.json", log)
        write_json(output / "training_log_PROVISIONAL.json", log)
    print("Saved Waveform v1 candidate artifacts:", output)


if __name__ == "__main__":
    main()
