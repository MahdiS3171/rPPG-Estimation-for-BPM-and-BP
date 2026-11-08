"""Generate deterministic SYNTHETIC timing controls; never opens user videos.

Usage: python scripts/audit_waveform_v2_timing_synthetic.py --out-dir outputs/new_gate03_synthetic
An existing output path is refused to avoid changing research artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from rppg_lab.waveform_v2_timing_contract import CONTRACT_VERSION, demonstration_cases


def main(argv=None):
    ap=argparse.ArgumentParser(description="Synthetic-only signed phase-transfer audit (no model/data loading)")
    ap.add_argument("--out-dir",required=True,type=Path)
    args=ap.parse_args(argv)
    destination=args.out_dir.resolve()
    if destination.exists():
        ap.error("Refusing existing output directory")
    destination.mkdir(parents=True)
    rows=[]
    for fs in (30.,60.):
        for r in demonstration_cases(fs):
            rows.append({"synthetic_fs_hz":fs,**r})
    payload={"contract_version":CONTRACT_VERSION,
             "warning":"Engineering QA ONLY: no evidence of biological pulse delay, waveform reconstruction accuracy or BP.",
             "scenario_results":rows}
    (destination/"synthetic_timing_contract_report.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    summary={"scenario_count":len(rows),"outcomes":{},"limitations":[
        "Unknown independent physiological ground truth", "No video, optical extraction, or model inference",
        "Subframe estimates must not be interpreted as acquired timing precision"]}
    for r in rows:
        summary["outcomes"][r["outcome"]]=summary["outcomes"].get(r["outcome"],0)+1
    (destination/"synthetic_timing_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__ == "__main__":
    main()
