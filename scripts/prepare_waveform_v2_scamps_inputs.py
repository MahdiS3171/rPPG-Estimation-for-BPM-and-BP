"""Gate 06: bounded SCAMPS ROI/input engineering diagnostics; no rPPG model calls."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import tracemalloc

import cv2
import h5py
import numpy as np
import psutil

from rppg_lab.waveform_v2_scamps import ScampsRecording, discover_scamps, source_sha256
from rppg_lab.waveform_v2_scamps_roi import (ScampsROIConfig, ROI_NAMES, create_scamps_face_detector,
                                           extract_scamps_roi_traces)
from rppg_lab.waveform_v2_scamps_inputs import prepare_scamps_v1_inputs, as_torch_inputs, CHANNEL_NAMES
from rppg_lab.waveform_v2_scamps_mask_audit import audit_mask_candidates


class RSSBudget:
    def __init__(self, limit):
        self.limit, self.peak = limit, psutil.Process().memory_info().rss
        self.start_rss = self.peak
        self.exceeded = False
        self.stop = threading.Event()
        self.process = psutil.Process()

    def check(self):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.exceeded |= self.peak > self.limit
        if self.exceeded:
            raise MemoryError("Explicit process RSS budget exceeded")

    def monitor(self):
        while not self.stop.wait(.002):
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.exceeded |= self.peak > self.limit

    def __enter__(self):
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _debug_callback(output, filename):
    def callback(index, image, masks):
        if index not in (0, 300, 599):
            return
        canvas = image.copy()
        colors = ((255, 210, 0), (0, 225, 220), (240, 75, 140))
        for name, color in zip(ROI_NAMES, colors):
            contours, _ = cv2.findContours(masks[name].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 1)
        from PIL import Image
        Image.fromarray(canvas).save(output / f"roi_{filename[:-4]}_frame{index:04d}.png")
    return callback


def _mask_preview(recording, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    index = 299
    with h5py.File(recording.path, "r") as f:
        rgb = f["Xsub"][:, :, :, index-1:index+2].transpose(3, 2, 1, 0)
        mask = f["skin_mask"][:, :, index].T
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), constrained_layout=True)
    for k in range(3):
        axes[k].imshow(rgb[k])
        axes[k].set_title(f"Xsub frame {index - 1 + k}")
    image = axes[3].imshow(mask, cmap="gray", vmin=min(0, float(mask.min())), vmax=max(1, float(mask.max())))
    axes[3].set_title(f"Stored skin-mask index {index}")
    fig.colorbar(image, ax=axes[3], fraction=.045)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"SCAMPS {recording.path.stem}: mask/frame relation UNVERIFIED\n"
                 "Separate original indices; no threshold or shift applied. McDuff et al., 2022 / R-UDA.", fontsize=11)
    fig.savefig(output / f"mask_candidates_{recording.path.stem}.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scamps-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--gate05-inventory", type=Path, default=Path("docs/waveform_v2/scamps/dataset_inventory.json"))
    ap.add_argument("--face-model", type=Path, default=Path("models/face_landmarker.task"))
    ap.add_argument("--chunk-frames", type=int, default=32)
    ap.add_argument("--array-budget-mib", type=int, default=128)
    ap.add_argument("--rss-budget-mib", type=int, default=1024)
    args = ap.parse_args(argv)
    paths = discover_scamps(args.scamps_root)
    if [p.name for p in paths] != [f"P{i:06d}.mat" for i in range(1, 11)]:
        ap.error("Gate 06 diagnostics are restricted to the ten admitted example filenames")
    output = args.out_dir.resolve()
    if output.exists():
        ap.error("Refusing existing output; use a new Gate 06 directory")
    source = args.scamps_root.resolve()
    if output == source or source in output.parents:
        ap.error("Output must be outside SCAMPS source files")
    if args.rss_budget_mib < 1:
        ap.error("RSS budget must be positive")
    config = ScampsROIConfig(chunk_frames=args.chunk_frames, array_budget_bytes=args.array_budget_mib * 1024**2)
    previous = json.loads(args.gate05_inventory.read_text(encoding="utf-8"))
    expected = {r["filename"]: r["source_sha256"] for r in previous["recordings"]}
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile("docs/waveform_v2/scamps/LICENSE_SCAMPS.txt", output / "LICENSE_SCAMPS.txt")
    rows, mask_reports, memory = [], [], []
    representative = None
    with RSSBudget(args.rss_budget_mib * 1024**2) as budget:
        for path in paths:
            print(f"{path.name}: streaming Xsub / CPU landmarks / two fixed input windows", flush=True)
            recording = ScampsRecording(source, path.name, max_window_frames=config.chunk_frames,
                                        max_window_bytes=64 * 1024**2, expected_sha256=expected[path.name])
            detector = create_scamps_face_detector(config, args.face_model)
            gc.collect()
            baseline = psutil.Process().memory_info().rss
            started = time.perf_counter()
            tracemalloc.start()
            try:
                traces = extract_scamps_roi_traces(recording, detector, config=config, resource_check=budget.check,
                    debug_callback=_debug_callback(output, path.name) if path.stem in ("P000001", "P000005") else None)
                _, allocation_peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
                detector.close()
            extraction_seconds = time.perf_counter() - started
            batch = prepare_scamps_v1_inputs(traces, (0, 300))
            assert batch["x"].shape == (2, 3, 9, 300) and np.isfinite(batch["x"]).all()
            assert np.array_equal(batch["frame_indices"], batch["label_indices"])
            if representative is None:
                representative = {key: batch[key][:1].copy() for key in ("x", "roi_valid", "roi_quality", "prior_valid")}
            coverage = {name: dict(valid_frames=int(traces.roi_valid[r].sum()), intended_frames=600,
                valid_fraction=float(traces.roi_valid[r].mean()), min_pixels=int(traces.pixel_count[r].min()),
                max_pixels=int(traces.pixel_count[r].max()),
                failure_reasons=dict(Counter(reason for reason in traces.failure_reason[r] if reason)))
                       for r, name in enumerate(ROI_NAMES)}
            rows.append(dict(filename=path.name, source_sha256=recording.info["source_sha256"],
                extraction=traces.provenance, landmark_success_frames=int(traces.detection_valid.sum()),
                intended_frames=600, roi_coverage=coverage, intended_windows=2,
                windows=[dict(start_index=s, stop_index=s+300, shape=list(batch["x"][k:k+1].shape),
                    status=batch["status"][k], roi_valid=batch["roi_valid"][k].tolist(),
                    roi_quality=batch["roi_quality"][k].tolist(), prior_valid=batch["prior_valid"][k].tolist(),
                    interpolated_samples_per_roi=batch["interpolated"][k].sum(-1).tolist(),
                    original_valid_samples_per_roi=batch["source_roi_valid"][k].sum(-1).tolist(),
                    input_sha256=hashlib.sha256(batch["x"][k].tobytes()).hexdigest(),
                    label_indices_identical=True, capture_timestamps=None, calibration_indices=[])
                         for k, s in enumerate((0, 300))],
                preparation_provenance=batch["provenance"]))
            mask_reports.append(audit_mask_candidates(recording))
            if path.stem in ("P000001", "P000005"):
                _mask_preview(recording, output)
            memory.append(dict(filename=path.name, extraction_seconds=extraction_seconds,
                baseline_rss_bytes=baseline, observed_process_peak_so_far_bytes=budget.peak,
                traced_python_numpy_extraction_peak_bytes=allocation_peak,
                estimated_streaming_array_peak_bytes=traces.provenance["estimated_streaming_array_peak_bytes"],
                max_decoded_chunk_frames=config.chunk_frames, frame_collection_materialized=False))
            assert source_sha256(path) == expected[path.name]
            print(f"{path.name}: detections {int(traces.detection_valid.sum())}/600; "
                  f"ROI valid {[int(v.sum()) for v in traces.roi_valid]}; windows {batch['status']}", flush=True)
            del traces, batch, recording, detector
            budget.check()
        tensors = as_torch_inputs(representative)
        assert list(tensors["x"].shape) == [1, 3, 9, 300]
        budget.check()
        memory_report = dict(array_budget_bytes=config.array_budget_bytes, process_rss_budget_bytes=budget.limit,
            process_start_rss_bytes=budget.start_rss, sampled_process_peak_rss_bytes=budget.peak,
            rss_budget_exceeded=budget.exceeded, rss_sampling_interval_s=.002, recordings=memory,
            tensor_shape_without_network_call=list(tensors["x"].shape), tensor_device=str(tensors["x"].device),
            limitations=["Native RSS sampled every 2 ms may miss brief peaks",
                         "Tracemalloc excludes some native detector/HDF5 allocations; RSS includes them",
                         "Array estimate is conservative workspace accounting, not an OS allocation limiter"])
    write_json(output / "roi_coverage_and_inputs.json", dict(scope="synthetic_engineering_only",
        roi_names=ROI_NAMES, channel_names=CHANNEL_NAMES, recordings=rows,
        recording_count=10, intended_frames=6000, intended_windows=20,
        usable_windows=sum(w["status"] == "READY_ENGINEERING" for r in rows for w in r["windows"]),
        no_neural_waveform_model_invoked=True, allow_training=False))
    write_json(output / "mask_alignment_diagnostic.json", dict(decision="UNVERIFIED", recordings=mask_reports,
        extraction_uses_scamps_segmentation=False, ppg_scores_computed=False,
        conclusion="Adjacent static/similar candidates and an unvalidated edge descriptor do not certify an offset"))
    write_json(output / "memory_timebase_integrity.json", dict(memory_report,
        source_indices_preserved=True, synthetic_grid_kind="documented_nominal_30Hz_frame_grid",
        source_timestamps=None, independent_sensor_clock_verified=False, source_ppg_shift_applied=0,
        subject_adaptation_enabled=False, calibration_requirement=False))
    print(json.dumps(dict(recordings=10, intended_frames=6000, intended_windows=20,
                         usable_windows=sum(w["status"] == "READY_ENGINEERING" for r in rows for w in r["windows"]),
                         rss_peak_bytes=budget.peak, allow_training=False)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
