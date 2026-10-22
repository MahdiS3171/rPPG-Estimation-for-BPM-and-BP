#!/usr/bin/env python
"""Read-only Pilot-6 data audit, dual-ROI extraction, and feasibility report."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key,"1")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root",default="Dataset")
    parser.add_argument("--metadata-file")
    parser.add_argument("--metadata-mapping",default="configs/pilot6_metadata.json")
    parser.add_argument("--google-sheet-id")
    parser.add_argument("--google-token-env",default="PILOT6_GOOGLE_ACCESS_TOKEN",help="Environment variable with authorized read-only bearer token; never persisted")
    parser.add_argument("--checkpoint",default="checkpoints/waveform_v1_frozen.pt")
    parser.add_argument("--freeze-manifest")
    parser.add_argument("--out-dir",default="outputs/pilot6_dual_roi/run01")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--recording",help="Exact filename or recording stem")
    selection.add_argument("--all-recordings",action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run",action="store_true",help="Data/metadata/checkpoint/timing audit only")
    mode.add_argument("--metadata-only",action="store_true")
    inference = parser.add_mutually_exclusive_group()
    inference.add_argument("--classical-only",action="store_true")
    inference.add_argument("--frozen",action="store_true",help="Attempt verified face inference; blocked paths remain reported")
    parser.add_argument("--max-frames",type=int,help="Explicit partial run; never treated as full BP session")
    parser.add_argument("--max-dimension",type=int,default=640,help="Classical spatial processing resolution, 0=native; never drops temporal frames")
    parser.add_argument("--private-overlays",action="store_true",help="Save participant images locally in ignored output directory")
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--reuse",action="store_true",help="Reuse only matching raw intermediates; regenerate analysis/report")
    parser.add_argument("--reports",action="store_true",help="Reports are always generated, including failures")
    parser.add_argument("--seed",type=int,default=42)
    args = parser.parse_args()
    if args.workers < 1 or args.max_dimension < 0 or args.max_frames is not None and args.max_frames < 1:
        parser.error("Invalid workers/frame limit/spatial dimension")
    from rppg_lab.pilot6 import run
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
