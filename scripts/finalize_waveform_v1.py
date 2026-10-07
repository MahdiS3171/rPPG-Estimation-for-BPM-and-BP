#!/usr/bin/env python
"""Review a saved locked test and optionally freeze its already-selected weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.waveform_freeze import freeze_waveform_candidate
from rppg_lab.waveform_review import build_locked_test_review, write_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-review", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--locked-test-evaluation", required=True)
    parser.add_argument("--generalization-review", required=True,
                        help="JSON with three reasoned engineering judgments; see finalization documentation")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    validation = read(args.validation_review)
    bootstrap = validation["validation_bootstrap"]
    review = build_locked_test_review(validation, checkpoint, read(args.locked_test_evaluation),
        args.checkpoint, args.split_manifest, args.locked_test_evaluation, read(args.generalization_review),
        bootstrap["resamples"], bootstrap["seed"])
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.freeze and review["freeze_decision"] == "FREEZE_WAVEFORM_V1":
        freeze_manifest = freeze_waveform_candidate(args.checkpoint, args.split_manifest, review, output)
        review.update(waveform_v1_status="frozen_v1", frozen_checkpoint=freeze_manifest["model_path"],
                      frozen_checkpoint_sha256=freeze_manifest["model_sha256"],
                      freeze_manifest_path=str((output / "waveform_v1_freeze_manifest.json").resolve()))
    write_review(review, output / "waveform_v1_review.json", output / "waveform_v1_review.md")
    print(review["freeze_decision"])
    print(review["freeze_reasons"])


if __name__ == "__main__":
    main()
