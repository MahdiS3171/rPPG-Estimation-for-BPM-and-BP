"""Read-only PURE registry adapter. No training, inference or frame decoding."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rppg_lab.waveform_v2_registry import (inventory_pure, draft_participant_split, write_new_json)
from rppg_lab.waveform_v2_benchmark import benchmark_protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory PURE; only explicit --out-dir writes new JSON")
    parser.add_argument("--pure-root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, help="Optional new output directory; no overwrite")
    parser.add_argument("--reference-timestamp-unit", default="unknown",
                        choices=["unknown", "seconds", "milliseconds", "nanoseconds"])
    parser.add_argument("--hash-images", action="store_true",
                        help="Read all PNG bytes to create content hash (can be very slow)")
    parser.add_argument("--access-status", default="unverified", choices=["unverified", "verified"])
    parser.add_argument("--license-scope", default="unknown")
    parser.add_argument("--license-evidence")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    root = args.pure_root.resolve()
    output = args.out_dir.resolve() if args.out_dir else None
    if output and (output == root or root in output.parents):
        parser.error("Output must be outside the PURE source tree")
    if output and output.exists():
        parser.error("Output directory already exists; never overwrite evidence or manifests")
    inventory = inventory_pure(root, reference_timestamp_unit=args.reference_timestamp_unit,
                               hash_images=args.hash_images, access_status=args.access_status,
                               license_scope=args.license_scope, license_evidence=args.license_evidence)
    splits = draft_participant_split(inventory, seed=args.seed)
    protocol = benchmark_protocol()
    if output:
        output.mkdir(parents=True, exist_ok=False)
        write_new_json(output / "dataset_manifest.json", inventory)
        write_new_json(output / "split_manifest.json", splits)
        write_new_json(output / "benchmark_protocol.json", protocol)
        print(f"Wrote 3 draft JSON files under {output}")
    print(f"PURE inventory: {inventory['participant_count']} subjects, "
          f"{inventory['recording_count']} recordings; all scoring eligibility flags remain FALSE")
    print("Split and benchmark protocol status: DRAFT_NOT_FROZEN. No dataset admitted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
