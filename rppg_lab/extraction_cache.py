"""Pickle-free raw face extraction cache using the Phase 1 array schema.

This is a subset of signals.npz, not a legacy FaceROIExtractor cache or a
cache of classical/neural waveforms. It reuses RGBTrace and SharedTimebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import os
import tempfile
import warnings
import zipfile

import numpy as np

from .artifacts import file_sha256
from .config import PipelineConfig
from .pipeline import process_recording
from .types import RGBTrace, SharedTimebase, validate_timestamps

EXTRACTION_CACHE_VERSION = "phase1_raw_face_rois_v1"
DEFAULT_WAVEFORM_ROIS = ("forehead", "left_cheek", "right_cheek")


@dataclass
class FaceROIExtraction:
    shared: SharedTimebase
    traces: dict[str, RGBTrace]


def extraction_cache_identity(video: str | Path, config: PipelineConfig, max_frames: int | None) -> dict:
    """Key source metadata, complete config, model content, code and libraries.

    Code hashes automatically invalidate geometry, clock or interpolation edits;
    the explicit schema version handles representation changes. No target or
    prior/window data enters the extraction cache.
    """
    path = Path(video).resolve()
    stat = path.stat()
    model = Path(config.face_model).resolve()
    libraries = {}
    for package in ("mediapipe", "opencv-python", "numpy", "scipy"):
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            libraries[package] = None
    source = Path(__file__).parent
    modules = ("pipeline", "detection", "region_masks", "roi", "video", "processing",
               "quality", "types", "config", "classical", "extraction_cache")
    return {
        "version": EXTRACTION_CACHE_VERSION,
        "video": {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "config": json.loads(json.dumps(config.to_dict())),
        "roi_names": list(config.requested_face_rois),
        "max_frames": max_frames,
        "face_model": {"path": str(model), "sha256": file_sha256(model) if model.is_file() else None},
        "source_sha256": {name: file_sha256(source / (name + ".py")) for name in modules},
        "libraries": libraries,
    }


def _validate(extraction: FaceROIExtraction, config: PipelineConfig) -> FaceROIExtraction:
    grid = validate_timestamps(extraction.shared.timestamps)
    source = validate_timestamps(extraction.shared.original_timestamps)
    if not len(grid) or not len(source):
        raise ValueError("Extraction clocks must be nonempty")
    if len(grid) > 1 and not np.allclose(np.diff(grid), 1 / config.sample_rate, atol=1e-7, rtol=1e-5):
        raise ValueError("Extraction grid does not match configured sample rate")
    if grid[0] < source[0] - 1e-8 or grid[-1] > source[-1] + 1e-8:
        raise ValueError("Uniform grid lies outside decoded video support")
    if tuple(extraction.traces) != config.requested_face_rois:
        raise ValueError("Extraction ROI order does not match config")
    for trace in extraction.traces.values():
        if not np.array_equal(trace.timestamps, source):
            raise ValueError("Every ROI must share the original decoded clock")
        if any(np.asarray(value).shape != source.shape for value in trace.quality.values()):
            raise ValueError("Raw extraction diagnostics must align with the source clock")
    return extraction


def load_face_roi_extraction(
    video: str | Path,
    config: PipelineConfig,
    cache_dir: str | Path | None = None,
    max_frames: int | None = None,
) -> FaceROIExtraction:
    """Run process_recording once for all ROIs, or read its cached raw traces.

    Cache writes are atomic. A malformed/truncated artifact is warned about and
    rebuilt. Reading always uses allow_pickle=False and checks clocks/ROI order.
    """
    if config.regions != ("face",):
        raise ValueError("Face extraction cache requires regions=('face',)")
    identity = extraction_cache_identity(video, config, max_frames) if cache_dir is not None else None
    cache_path = None
    if identity is not None:
        root = Path(cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        cache_path = root / ("phase1_face_" + key + ".npz")
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as arrays:
                    metadata = json.loads(str(arrays["metadata_json"].item()))
                    if metadata["identity"] != identity:
                        raise ValueError("Extraction cache identity mismatch")
                    source = arrays["timestamps_original"]
                    grid = arrays["timestamps_uniform"]
                    names = arrays["face_roi_names"].tolist()
                    traces = {}
                    for name in names:
                        prefix = "face_roi_" + name + "_"
                        quality = {metric: arrays[prefix + metric] for metric in metadata["quality_keys"][name]}
                        traces[name] = RGBTrace(source, arrays[prefix + "rgb"], metadata["trace_roi_names"][name],
                                                arrays[prefix + "valid"], quality)
                    return _validate(FaceROIExtraction(SharedTimebase(grid, source), traces), config)
            except (ValueError, KeyError, OSError, EOFError, TypeError, zipfile.BadZipFile) as exc:
                warnings.warn(f"Rebuilding unreadable Phase 1 extraction cache {cache_path}: {exc}", RuntimeWarning)

    result = process_recording(video, config, max_frames=max_frames)
    extraction = _validate(FaceROIExtraction(result.shared,
                           {name: result.face_rois[name].rgb for name in config.requested_face_rois}), config)
    if cache_path is not None:
        arrays = {
            "timestamps_original": extraction.shared.original_timestamps,
            "timestamps_uniform": extraction.shared.timestamps,
            "face_roi_names": np.asarray(list(extraction.traces), dtype=str),
            "metadata_json": np.asarray(json.dumps({"identity": identity,
                "trace_roi_names": {name: trace.roi_name for name, trace in extraction.traces.items()},
                "quality_keys": {name: list(trace.quality) for name, trace in extraction.traces.items()}}, sort_keys=True)),
        }
        for name, trace in extraction.traces.items():
            prefix = "face_roi_" + name + "_"
            arrays[prefix + "rgb"] = trace.values
            arrays[prefix + "valid"] = trace.validity_mask
            arrays.update({prefix + metric: value for metric, value in trace.quality.items()})
        # A sibling temporary file keeps os.replace atomic on the same volume.
        fd, temp_name = tempfile.mkstemp(prefix=cache_path.stem + "_", suffix=".npz", dir=cache_path.parent)
        os.close(fd)
        try:
            np.savez_compressed(temp_name, **arrays)
            os.replace(temp_name, cache_path)
        finally:
            Path(temp_name).unlink(missing_ok=True)
    return extraction
