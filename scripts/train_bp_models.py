#!/usr/bin/env python
"""Compare mean/linear/ridge/forest baselines on subject-independent pilot data."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import logging
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rppg_lab.bp_models import train_baselines
from rppg_lab.splits import split_subjects
from rppg_lab.artifacts import reserve_run,write_json,provenance,file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables",nargs="+",help="One or more bp_table.csv files")
    parser.add_argument("--out",required=True)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--val-fraction",type=float,default=0.2)
    parser.add_argument("--test-fraction",type=float,default=0.2)
    parser.add_argument("--split-manifest",help="Reuse an existing splits.json to keep the same locked subjects")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    rows = []
    for table in args.tables:
        with Path(table).open(encoding="utf-8",newline="") as f:
            rows.extend(csv.DictReader(f))
    if args.split_manifest:
        partitions = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
        if set(partitions) != {r["subject_id"] for r in rows}:
            raise ValueError("Saved partition subjects must match the supplied tables exactly")
    else:
        tr,va,te = split_subjects([r["subject_id"] for r in rows],args.val_fraction,args.test_fraction,args.seed)
        if not te:
            raise ValueError("BP baseline comparison requires a locked test partition")
        partitions = {sid:split for split,ids in (("train",tr),("validation",va),("test",te)) for sid in ids}
    output = reserve_run(args.out)
    write_json(output/"splits.json",partitions)
    write_json(output/"provenance.json",{**provenance(),"args":vars(args),"input_tables":{str(Path(t).resolve()):file_sha256(t) for t in args.tables}})
    try:
        model,report = train_baselines(rows,partitions,args.seed)
        model.save(output/"model.pkl")
        write_json(output/"metrics.json",report)
    except Exception as exc:
        write_json(output/"failure.json",{"reason":type(exc).__name__,"detail":str(exc)})
        raise
    logging.info("Selected %s on validation; saved model and locked-test report to %s",report["selected_model"],output)


if __name__ == "__main__":
    main()
