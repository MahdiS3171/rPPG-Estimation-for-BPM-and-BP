#!/usr/bin/env python
"""Estimate HR from one video using a classical rPPG method."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.roi import extract_rgb_trace
from rppg_lab.classical import METHOD_FUNCS
from rppg_lab.signals import estimate_hr_welch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=str)
    p.add_argument("--roi", default="face", choices=["face", "forehead", "left_cheek", "right_cheek"])
    p.add_argument("--method", default="POS_WIN", choices=sorted(METHOD_FUNCS))
    p.add_argument("--fs", type=float, default=30.0)
    p.add_argument("--cache-dir", default="cache_roi")
    p.add_argument("--min-quality", type=float, default=0.0)
    args = p.parse_args()

    rgb, fs, t, q = extract_rgb_trace(args.video, roi=args.roi, cache_dir=args.cache_dir, fs_target=args.fs, min_quality=args.min_quality)
    rppg = METHOD_FUNCS[args.method](rgb, fs)
    hr = estimate_hr_welch(rppg, fs)
    print(f"frames={len(rgb)} fs={fs:.2f} roi={args.roi} method={args.method}")
    print(f"HR={hr.hr_bpm:.2f} bpm | confidence={hr.confidence:.3f} | SNR={hr.snr_db:.2f} dB")


if __name__ == "__main__":
    main()
