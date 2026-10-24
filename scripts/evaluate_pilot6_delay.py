"""Run the isolated protocol-aware feasibility study; existing runs never replaced."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from rppg_lab.pilot6_delay_runner import run
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata-path", default="sheets/rPPG_BP_Data_Collection_2026-10-24.xlsx")
    p.add_argument("--previous-metadata-path", default="sheets/rPPG BP Data Collection final.xlsx")
    p.add_argument("--config", default="configs/pilot6_delay_v1.json")
    p.add_argument("--video-dir", default="Dataset")
    p.add_argument("--out-dir", default="outputs/pilot6_protocol_aware_delay_v1/run01")
    p.add_argument("--face-cache-root", default="outputs/pilot6_frozen_native_aa_v1/run01")
    p.add_argument("--stage", choices=("all", "acquire", "analyze", "metadata"), default="all")
    p.add_argument("--resume", action="store_true", help="Resume this experiment only; identity checks mandatory")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--frozen-face", action="store_true", help="Run original facial API + accepted V2 adapter separately; never use for delay")
    run(p.parse_args())


if __name__ == "__main__":
    main()
