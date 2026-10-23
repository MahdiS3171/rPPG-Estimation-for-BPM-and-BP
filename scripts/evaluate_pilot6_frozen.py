#!/usr/bin/env python
"""Native-resolution Pilot-6 frozen facial inference with explicit 60-to-30 adaptation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from rppg_lab.pilot_frozen import DEFAULT_CHECKPOINT, run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="Dataset")
    parser.add_argument("--metadata-file", default="sheets/rPPG BP Data Collection final.xlsx")
    parser.add_argument("--metadata-mapping", default="configs/pilot6_metadata.json")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--freeze-manifest")
    parser.add_argument("--adaptation-config", default="configs/pilot6_frozen_timebase_v2.json")
    parser.add_argument("--out-dir", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--recording", help="Exact filename or recording stem; begin with S004_V01")
    selection.add_argument("--all-recordings", action="store_true", help="S004 gate runs before the remaining five")
    parser.add_argument("--dry-run", action="store_true", help="Metadata/checkpoint/PTS audit only")
    parser.add_argument("--reuse", action="store_true", help="Reuse matching native RGB; regenerate this run's analysis/report")
    parser.add_argument("--reports-only", action="store_true", help="Audit and summarize existing per-recording summaries")
    parser.add_argument("--workers", type=int, default=1, help="Parallel recordings after S004 succeeds")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
