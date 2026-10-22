#!/usr/bin/env python
"""Visual validation of the official frozen Waveform v1 on its saved validation split.

Always rank all validation windows, then render the requested examples.
The consumed locked test cannot be requested. No training/extraction fallback.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.waveform_visualization import DEFAULT_CHECKPOINT, DEFAULT_SPLIT, run_visual_validation


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ubfc-root", required=True, type=Path)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--freeze-manifest", type=Path, help="Defaults to mandatory adjacent waveform_v1_freeze_manifest.json")
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT, type=Path)
    parser.add_argument("--cache-dir", default="cache_roi_multi_phase1", type=Path)
    parser.add_argument("--out-dir", default="outputs/waveform_visual_v1", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--subject", help="Exact saved validation ID; selects figures, while all validation windows are ranked")
    group.add_argument("--all-validation-subjects", action="store_true", help="Render all validation subjects (also the default)")
    choose = parser.add_mutually_exclusive_group()
    choose.add_argument("--start-sec", type=float, help="Exact retained window start; requires --subject")
    choose.add_argument("--selection", choices=("best", "median", "worst", "random"), help="Default: best, median, worst per subject")
    parser.add_argument("--include-random", action="store_true", help="Add a seeded example distinct from best/median/worst when possible")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", help="Optional additional checkpoint prior's multi-ROI average (e.g. GREEN)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start_sec is not None and args.subject is None:
        parser.error("--start-sec requires --subject")
    if args.start_sec is not None and args.include_random:
        parser.error("--include-random cannot be combined with the single --start-sec selection")
    try:
        result = run_visual_validation(ubfc_root=args.ubfc_root, checkpoint_path=args.checkpoint,
            split_path=args.split_manifest, freeze_path=args.freeze_manifest, cache_dir=args.cache_dir,
            out_dir=args.out_dir, subject=args.subject, selection=args.selection, start_sec=args.start_sec,
            include_random=args.include_random, seed=args.seed, baseline=args.baseline)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"visualize_waveform_v1: {exc}\n")
    print(f"Saved {len(result['selected_windows'])} real validation figures; ranked {result['validation_windows']} windows: {args.out_dir}")


if __name__ == "__main__":
    main()
