#!/usr/bin/env python
"""Explicit val or locked-test evaluation of a selected Part 3 checkpoint.

Run test after model selection and before a later explicit freeze. The trainer
never calls this entry point. Undefined metrics are NaN in memory, null in JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.artifacts import file_sha256, provenance, write_json
from rppg_lab.datasets import UBFCMultiROIRPPGDataset, find_ubfc_subjects
from rppg_lab.waveform_metrics import evaluate_waveform_model
from rppg_lab.waveform_training import checkpoint_dataset_config, reconstruct_waveform_model
from rppg_lab.waveform_review import reserve_locked_test


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-manifest", help="Defaults to the checkpoint manifest path")
    parser.add_argument("--split", choices=("val", "test"), required=True,
                        help="Explicitly choose val or test; test is a post-selection action")
    parser.add_argument("--ubfc-root", required=True)
    parser.add_argument("--cache-dir", default="cache_roi_multi_phase1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-baselines", action="store_true",
                        help="Validation-only classical comparisons on identical Part 2 windows")
    parser.add_argument("--validation-review", help="Accepted saved review; required before locked test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    manifest_path = Path(args.split_manifest or checkpoint["split_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_config = checkpoint_dataset_config(checkpoint, manifest)
    receipt = None
    if args.split == "test":
        if not args.validation_review or args.include_baselines:
            parser.error("Locked test requires --validation-review and forbids baseline comparisons")
        receipt = reserve_locked_test(args.checkpoint, manifest_path, args.validation_review, args.out)
    ids = manifest["validation_ids" if args.split == "val" else "test_ids"]
    if not ids:
        raise ValueError("Requested split has no participants (two-subject smoke has no test set)")
    by_id = {s.subject_id: s for s in find_ubfc_subjects(args.ubfc_root)}
    missing = set(ids) - by_id.keys()
    if missing:
        raise ValueError(f"Missing requested split participants: {sorted(missing)}")
    model = reconstruct_waveform_model(checkpoint, device)
    torch.manual_seed(checkpoint["seed"])
    dataset = UBFCMultiROIRPPGDataset([by_id[s] for s in ids], cache_dir=args.cache_dir, **dataset_config)
    if not len(dataset):
        raise ValueError("Requested split has no usable windows")
    for attribute in ("roi_names", "prior_names", "channel_names"):
        if tuple(getattr(dataset, attribute)) != tuple(checkpoint[attribute]):
            raise ValueError(f"Incompatible dataset {attribute}")
    cfg = checkpoint["loss_config"]
    metrics = evaluate_waveform_model(model, DataLoader(dataset, batch_size=args.batch_size, shuffle=False),
        device, checkpoint["fs"], cfg["max_lag_sec"], cfg["spectral_fmin_hz"], cfg["spectral_fmax_hz"],
        include_windows=True, include_baselines=args.include_baselines)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, dict(split=args.split, subject_ids=ids, checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=file_sha256(args.checkpoint), epoch=checkpoint["epoch"],
        waveform_v1_status=checkpoint["waveform_v1_status"], split_manifest=manifest,
        split_manifest_sha256=checkpoint["split_manifest_sha256"],
        split_manifest_file_sha256=file_sha256(manifest_path), locked_test_consumed=args.split == "test",
        locked_test_receipt=receipt, metrics=metrics,
        exclusions=dataset.exclusions, reference_diagnostics=dataset.reference_diagnostics, provenance=provenance()))
    print("Subject-balanced metrics:", metrics["subject_balanced_metrics"])
    print("Saved:", output)


if __name__ == "__main__":
    main()
