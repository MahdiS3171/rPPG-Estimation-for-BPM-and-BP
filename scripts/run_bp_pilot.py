#!/usr/bin/env python
"""Generate signal-only windows and an explicitly cuff-associated pilot table."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_dual_roi import add_arguments,run
from rppg_lab.config import PipelineConfig
from rppg_lab.study import Recording
from rppg_lab.artifacts import write_json,json_safe,file_sha256
from rppg_lab.bp import FEATURE_KEYS


def associate_windows(result, recording: Recording, fingerprint: str) -> list[dict]:
    """Only wholly contained windows receive explicitly approximate labels."""
    refs = {r.measurement_index:r for r in recording.references}
    start = datetime.fromisoformat(recording.video_start_time)
    end = start+timedelta(seconds=float(result.shared.original_timestamps[-1]))
    for ref in recording.references:
        a,b = datetime.fromisoformat(ref.start_time),datetime.fromisoformat(ref.end_time)
        relation_valid = {"before":b <= start,"after":a >= end,"during":max(a,start) < min(b,end)}
        if not relation_valid[ref.relation_to_video]:
            raise ValueError("Cuff relation_to_video contradicts acquisition times")
    for association in recording.associations:
        nominal = result.video.fps_nominal
        step = 1/nominal if nominal > 0 else 1/result.face.rppg.sample_rate
        if association.interval_end_sec > float(result.shared.original_timestamps[-1])+step:
            raise ValueError("Cuff association interval exceeds decoded recording")
    rows = []
    for window in result.bp_features:
        row = {**window,"reasons":list(window["reasons"]),"subject_id":recording.subject_id,
               "session_id":recording.session_id,"recording_id":recording.recording_id,"input_sha256":fingerprint,
               "sbp_mmHg":None,"dbp_mmHg":None,"reference_index":None,"reference_start_time":None,
               "reference_end_time":None,"reference_relation":None,"reference_device":None,
               "association_method":None,"association_notes":None}
        matching = [a for a in recording.associations if window["start_sec"] >= a.interval_start_sec-1e-8 and window["end_sec"] <= a.interval_end_sec+1e-8]
        if matching:
            association = matching[0]
            ref = refs[association.reference_index]
            row.update(sbp_mmHg=ref.sbp,dbp_mmHg=ref.dbp,reference_index=ref.measurement_index,
                       reference_start_time=ref.start_time,reference_end_time=ref.end_time,
                       reference_relation=ref.relation_to_video,reference_device=ref.device,
                       association_method=association.method,association_notes=association.notes)
        else:
            row["status"] = "excluded"
            row["reasons"].append("invalid_bp_reference_no_explicit_association")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording",help="Recording JSON manifest")
    add_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    recording = Recording.load(args.recording)
    config = PipelineConfig.load(args.config)
    if set(config.regions) != {"face","hand"}:
        raise ValueError("BP pilot requires both regions")
    result,output,info = run(recording.video_path,args,config)
    write_json(output/"recording.json",asdict(recording))
    info["recording_manifest_sha256"] = file_sha256(args.recording)
    write_json(output/"provenance.json",info)
    try:
        rows = associate_windows(result,recording,info["input_sha256"])
    except ValueError as exc:
        write_json(output/"bp_reference_failure.json",{"reason":"invalid_bp_reference","detail":str(exc)})
        raise
    columns = ["subject_id","session_id","recording_id","input_sha256","window_index","start_sec","end_sec","status","reasons",
               *FEATURE_KEYS,"sbp_mmHg","dbp_mmHg","reference_index","reference_start_time","reference_end_time",
               "reference_relation","reference_device","association_method","association_notes"]
    with (output/"bp_table.csv").open("w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**json_safe(row),"reasons":"|".join(row["reasons"])})
    write_json(output/"bp_exclusions.json",[{"window_index":r["window_index"],"reasons":r["reasons"]} for r in rows if r["status"] != "accepted"])
    logging.info("Saved %d pilot windows (%d accepted) to %s",len(rows),sum(r["status"] == "accepted" for r in rows),output)


if __name__ == "__main__":
    main()
