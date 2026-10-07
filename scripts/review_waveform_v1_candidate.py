#!/usr/bin/env python
"""Review saved validation artifacts only; never run inference or open locked test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.waveform_review import build_validation_review, write_review


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-evaluation", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    review = build_validation_review(read(args.training_log),
        torch.load(args.checkpoint, map_location="cpu", weights_only=True),
        read(args.validation_evaluation), read(args.split_manifest), args.checkpoint,
        args.validation_evaluation, args.training_log, args.split_manifest,
        args.bootstrap_resamples, args.bootstrap_seed)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "locked_test_consumption.json").exists():
        raise ValueError("Preserve the pre-test review after locked-test consumption")
    write_review(review, output / "waveform_v1_review.json", output / "waveform_v1_review.md")
    print(review["validation_decision"])
    print(review["reasons"])


if __name__ == "__main__":
    main()
