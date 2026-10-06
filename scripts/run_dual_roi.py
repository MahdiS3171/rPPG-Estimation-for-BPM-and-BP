#!/usr/bin/env python
"""Extract face/hand with one clock, saved quality and optional debug plots."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import logging
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rppg_lab.config import PipelineConfig
from rppg_lab.pipeline import process_recording
from rppg_lab.artifacts import reserve_run, provenance, save_result, plot_result, overlay_writer, write_json, file_sha256


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config",default="configs/bp_pilot.json")
    parser.add_argument("--out",required=True,help="New run directory; existing paths are refused")
    parser.add_argument("--timestamps",help="NPY timestamps in video-relative seconds, one per decoded frame")
    parser.add_argument("--max-frames",type=int)
    parser.add_argument("--debug",action="store_true",help="Save sparse ROI overlays and signal/timing plots")


def run(video: str | Path, args, config: PipelineConfig):
    output = reserve_run(args.out)
    info = {"input_path":str(video)}
    try:
        info = provenance(video)
        if args.timestamps:
            info["timestamp_file"] = str(Path(args.timestamps).resolve())
            info["timestamp_file_sha256"] = file_sha256(args.timestamps)
        info["detector_model_sha256"] = {site:file_sha256(path) for site,path in (("face",config.face_model),("hand",config.hand_model)) if site in config.regions and Path(path).is_file()}
        timestamps = np.load(args.timestamps,allow_pickle=False) if args.timestamps else None
        result = process_recording(video,config,timestamps=timestamps,max_frames=args.max_frames,
                    debug_callback=overlay_writer(output/"roi_debug") if args.debug else None)
        save_result(result,output,info)
        if args.debug:
            plot_result(result,output)
    except Exception as exc:
        write_json(output/"failure.json",{"reason":type(exc).__name__,"detail":str(exc),"config":config.to_dict(),"provenance":info})
        raise
    return result,output,info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    add_arguments(parser)
    parser.add_argument("--regions",choices=["face","hand","both"],help="Override regions from the config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(name)s: %(message)s")
    config = PipelineConfig.load(args.config)
    if args.regions:
        config = replace(config,regions=("face","hand") if args.regions == "both" else (args.regions,))
    result,output,_ = run(args.video,args,config)
    logging.info("HR face=%s hand=%s bpm; delay=%s; saved %s",result.hr["face"],result.hr["hand"],result.delay,output)


if __name__ == "__main__":
    main()
