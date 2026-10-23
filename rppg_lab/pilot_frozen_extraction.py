"""Separate native-pixel facial acquisition for the frozen Pilot-6 experiment.

No historical extraction code or cache is modified. Temporal adaptation is a
subsequent operation; every decoded frame is measured on its MP4 display PTS.
Participant pixels are never written. Saved polygon vertices reconstruct the
exact rounded convex-hull masks using the historical region_masks utilities.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .artifacts import file_sha256, write_json
from .detection import FaceRegionDetector
from .extraction_cache import extraction_cache_identity
from .pipeline import _observation_quality
from .pilot_video import mp4_presentation_times
from .region_masks import get_face_masks, extract_frame_rgb
from .roi import FACE_ROIS
from .video import VideoReader

VERSION = "pilot6_native_frozen_face_v1"


def measure_native_masks(frame, masks, min_pixels):
    """Same pixels/order/formula as the historical mean, avoiding background scans.

    Cropping here only bounds an already constructed full-resolution mask; no
    detector input, geometry, pixel or spatial sampling is changed.
    """
    import cv2
    result = {}
    for name, mask in masks.items():
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        w, h = max(1, w), max(1, h)
        rgb, quality = extract_frame_rgb(frame[y:y+h, x:x+w], mask[y:y+h, x:x+w], min_pixels)
        quality["mask_coverage"] = quality["mask_pixel_count"] / mask.size
        result[name] = rgb, quality
    return result


def native_face_identity(video, config):
    return dict(version=VERSION, historical_identity=extraction_cache_identity(video, config, None),
                input_sha256=file_sha256(video), spatial_resize=False,
                source_sha256=file_sha256(Path(__file__)))


def extract_native_face(video, out, config, reuse=False):
    """Frozen geometry/config on full resolution, with no rate conversion here."""
    import cv2
    cv2.setNumThreads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "native_face_rgb.npz"
    identity = native_face_identity(video, config)
    if cache.exists():
        if not reuse:
            raise FileExistsError("Native extraction exists; use --reuse or a fresh output directory")
        with np.load(cache, allow_pickle=False) as data:
            if json.loads(str(data["identity_json"].item())) != identity:
                raise ValueError("Native extraction identity changed; use a fresh output directory")
            return {key: data[key].copy() for key in data.files}
    pts, clock = mp4_presentation_times(video)
    reader = VideoReader(video, timestamp_mode=config.timestamp_mode, timestamps=pts)
    detector = FaceRegionDetector(config.detector_backend, config.face_model,
                                  config.min_detection_confidence, config.min_tracking_confidence)
    names = config.requested_face_rois
    values, metrics, polygons = ({name: [] for name in names} for _ in range(3))
    reasons, repeated, previous = [], [], {}
    last = None
    started = time.monotonic()
    try:
        for index, t, frame in reader.frames():
            observation = detector.detect(frame, index, t)
            masks = get_face_masks(frame, observation, names, config.skin_mask,
                                   config.skin_cr_range, config.skin_cb_range)
            common = _observation_quality(observation, previous, "face", config.max_gap_sec)
            for name, (rgb, quality) in measure_native_masks(frame, masks, config.min_pixels).items():
                values[name].append(rgb)
                metrics[name].append(dict(quality, **common))
                vertices = np.full((len(FACE_ROIS[name]), 2), np.nan)
                if observation.valid and observation.landmarks is not None:
                    vertices = observation.landmarks[FACE_ROIS[name]].copy()
                polygons[name].append(vertices)
            reasons.append(observation.reason or "ok")
            repeated.append(last is not None and np.array_equal(last, frame))
            last = frame.copy()
            if index % 600 == 0:
                progress = dict(recording_id=Path(video).stem, decoded_frames=index + 1,
                                expected_frames=len(pts), last_pts_sec=t,
                                elapsed_sec=time.monotonic() - started, native_shape=list(frame.shape),
                                detector_backend=detector.backend)
                write_json(out / "extraction_progress.json", progress)
                print(json.dumps(progress), flush=True)
        arrays = dict(camera_pts=reader.metadata.timestamps,
                      decoder_pts=np.asarray(reader.metadata.backend_timestamps),
                      roi_names=np.asarray(names), frame_shape=np.asarray(last.shape),
                      repeated_decoded_frame=np.asarray(repeated), tracking_reason=np.asarray(reasons),
                      identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
                      clock_json=np.asarray(json.dumps(clock)),
                      actual_backend=np.asarray(detector.backend))
        for name in names:
            arrays[name + "_rgb"] = np.asarray(values[name])
            arrays[name + "_vertices"] = np.asarray(polygons[name])
            for key in metrics[name][0]:
                arrays[name + "_" + key] = np.asarray([q[key] for q in metrics[name]])
        temporary = cache.with_suffix(".partial.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(cache)
        write_json(out / "native_extraction.json", dict(identity=identity,
            frames=len(pts), shape=arrays["frame_shape"], backend=detector.backend,
            clock=clock, elapsed_sec=time.monotonic() - started,
            timestamps_preserved=np.array_equal(pts, arrays["camera_pts"]),
            pixel_artifacts="none; full-resolution mask vertices only",
            confidence="not exposed by FaceMesh/FaceLandmarker; retained as missing"))
        return arrays
    finally:
        detector.close()
        reader.close()
