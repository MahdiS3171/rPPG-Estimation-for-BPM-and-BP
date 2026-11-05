"""Bounded Xsub -> V1 landmark-polygon RGB, without SCAMPS segmentation.

SCAMPS (McDuff et al., 2022) is used for non-commercial engineering only.
See docs/waveform_v2/scamps/ATTRIBUTION.md. No waveform model is loaded here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
import hashlib

import numpy as np

from .detection import FaceRegionDetector
from .region_masks import extract_frame_rgb_by_roi, get_face_masks
from .roi import FACE_ROIS

ROI_NAMES = tuple(FACE_ROIS)
RGB_CHANNELS = ("RGB_R", "RGB_G", "RGB_B")
EXTRACTION_SCHEMA = "scamps_v1_landmark_rgb_v0.1"


@dataclass(frozen=True)
class ScampsROIConfig:
    chunk_frames: int = 32
    min_pixels: int = 50
    nominal_rate_hz: float = 30.
    array_budget_bytes: int = 128 * 1024 * 1024
    min_detection_confidence: float = .5
    min_tracking_confidence: float = .5

    def __post_init__(self):
        if (type(self.chunk_frames) is not int or not 1 <= self.chunk_frames <= 64
                or type(self.min_pixels) is not int or self.min_pixels < 1
                or type(self.array_budget_bytes) is not int or self.array_budget_bytes < 1
                or self.nominal_rate_hz != 30.
                or not 0 <= self.min_detection_confidence <= 1
                or not 0 <= self.min_tracking_confidence <= 1):
            raise ValueError("Invalid fixed 30-Hz engineering extraction configuration")


@dataclass
class ScampsROITraces:
    rgb: np.ndarray                       # [R,N,3], continuous V1 intensity units
    roi_valid: np.ndarray                 # bool [R,N], ORIGINAL measurement support
    frame_indices: np.ndarray            # original synthetic indices, never compressed
    label_indices: np.ndarray
    raw_d_ppg: np.ndarray
    pixel_count: np.ndarray               # [R,N]
    mask_coverage: np.ndarray             # [R,N], landmark geometry only
    failure_reason: np.ndarray            # [R,N]
    detection_valid: np.ndarray           # bool [N]
    nominal_grid_seconds: np.ndarray      # processing coordinate, NOT capture timestamps
    provenance: dict
    roi_names: tuple[str, ...] = ROI_NAMES
    rgb_channel_names: tuple[str, ...] = RGB_CHANNELS


def preprocessing_provenance() -> dict:
    source = Path(__file__).parent
    names = ("roi", "region_masks", "detection", "signals", "processing", "classical",
             "window_priors", "waveform_v2_scamps", "waveform_v2_scamps_roi", "waveform_v2_scamps_inputs")
    return dict(preprocessing_source_sha256={name: hashlib.sha256((source / (name + ".py")).read_bytes()).hexdigest()
                              for name in names},
                source_hash_basis="working_tree_bytes",
                software={name: version(name) for name in
                          ("numpy", "scipy", "scikit-learn", "opencv-python", "mediapipe", "h5py")})


def create_scamps_face_detector(config: ScampsROIConfig, model_path="models/face_landmarker.task"):
    """Explicit existing CPU Tasks path; no download or automatic ROI/backend fallback."""
    detector = FaceRegionDetector("tasks", model_path, config.min_detection_confidence,
                                  config.min_tracking_confidence)
    detector.scamps_provenance = dict(kind="mediapipe_face_landmarker", backend="tasks_cpu",
        model_filename=Path(model_path).name, model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        confidence_kind="not_exposed_per_frame", tracking_reset="new_detector_per_recording")
    return detector


def _measure_frame(frame, detector, index, coordinate, config, debug_callback=None):
    colors = np.full((3, 3), np.nan, dtype=np.float64)
    counts, coverage = np.zeros(3, dtype=np.int32), np.zeros(3, dtype=np.float64)
    valid = np.zeros(3, dtype=bool)
    reasons = np.full(3, "invalid_rgb_frame", dtype="U80")
    if not np.isfinite(frame).all() or (frame < 0).any() or (frame > 1).any():
        return colors, valid, counts, coverage, reasons, False
    # Geometry uses an explicit uint8 COPY. Measurements use unchanged float
    # intensities *255, with no rounding, thresholding or stored-frame mutation.
    geometry_frame = np.rint(frame * 255.).astype(np.uint8)
    try:
        observation = detector.detect(geometry_frame, index, coordinate)
    except Exception as exc:
        reasons[:] = "detector_error:" + type(exc).__name__
        return colors, valid, counts, coverage, reasons, False
    if observation.frame_index != index or observation.timestamp != coordinate:
        raise ValueError("Detector changed synthetic frame/coordinate identity")
    if not observation.valid:
        reasons[:] = observation.reason or "face_not_detected"
        return colors, valid, counts, coverage, reasons, False
    points = np.asarray(observation.landmarks)
    if (points.ndim != 2 or points.shape[1] != 2
            or len(points) <= max(max(v) for v in FACE_ROIS.values())
            or not np.isfinite(points).all()):
        reasons[:] = "invalid_landmarks"
        return colors, valid, counts, coverage, reasons, False
    masks = get_face_masks(geometry_frame, observation, ROI_NAMES, use_skin_mask=False)
    # Geometry is reused verbatim from V1. Never silently coerce malformed masks.
    if tuple(masks) != ROI_NAMES or any(m.dtype != bool or m.shape != frame.shape[:2] for m in masks.values()):
        raise ValueError("Invalid landmark ROI mask shape, dtype or ordering")
    if debug_callback is not None:
        debug_callback(index, geometry_frame, masks)
    measured = extract_frame_rgb_by_roi(frame * 255., masks, config.min_pixels)
    for r, name in enumerate(ROI_NAMES):
        color, quality = measured[name]
        counts[r], coverage[r] = quality["skin_pixel_count"], quality["mask_coverage"]
        if quality["extraction_valid"] and np.isfinite(color).all():
            colors[r], valid[r], reasons[r] = color, True, ""
        else:
            reasons[r] = "insufficient_roi_pixels"
    return colors, valid, counts, coverage, reasons, True


def extract_scamps_roi_traces(recording, detector, *, config=ScampsROIConfig(),
                              resource_check=None, debug_callback=None) -> ScampsROITraces:
    """One pass, one detection per frame. Caller owns/reset/closes the detector.

    No segmentation stream is consumed by ROI extraction. Failure rows retain NaNs,
    masks and original indices; no geometric fallback or track reuse is invented.
    Only O(N*R) RGB/quality/label arrays persist between bounded frame chunks.
    """
    frame_info = recording.info["datasets"]["Xsub"]
    c, w, h, n = frame_info["shape"]
    if c != 3 or n != recording.layout.frames or recording.max_window_frames < config.chunk_frames:
        raise ValueError("Reader layout/window limit incompatible with extraction configuration")
    itemsize = np.dtype(frame_info["dtype"]).itemsize
    chunk_bytes = config.chunk_frames * h * w * 3 * itemsize
    # Gate 05 read + transpose copies, boolean scratch and per-frame geometry/RGB.
    estimated_bytes = 2 * chunk_bytes + 8 * config.chunk_frames * h * w + 4 * h * w * 3 * itemsize
    persistent_bytes = n * (3 * 3 * 8 + 3 * (1 + 4 + 8 + 80 * 4) + 8 * 4 + 1)
    if estimated_bytes + persistent_bytes > config.array_budget_bytes:
        raise ValueError("Streaming array estimate exceeds explicit budget; reduce chunk_frames")
    rgb = np.full((3, n, 3), np.nan)
    validity = np.zeros((3, n), dtype=bool)
    counts, coverage = np.zeros((3, n), dtype=np.int32), np.zeros((3, n), dtype=np.float64)
    reasons = np.full((3, n), "unprocessed", dtype="U80")
    detected, ppg = np.zeros(n, dtype=bool), np.full(n, np.nan)
    indices = np.arange(n, dtype=np.int64)
    nominal = indices / config.nominal_rate_hz
    for start in range(0, n, config.chunk_frames):
        if resource_check is not None:
            resource_check()
        stop = min(start + config.chunk_frames, n)
        block = recording.read_window(start, stop, frames="Xsub", require_segmentation=False)
        p = block["provenance"]
        if (p["channel_order"] != "RGB" or p["decoded_axes"] != "T,H,W,C" or p["frames_variable"] != "Xsub"
                or block["rgb"].shape != (stop - start, h, w, 3)
                or not np.issubdtype(block["rgb"].dtype, np.floating)
                or block["d_ppg"].shape != (stop - start,)
                or not np.array_equal(block["frame_indices"], indices[start:stop])
                or not np.array_equal(block["label_indices"], indices[start:stop])):
            raise ValueError("Malformed frame/channel/index contract from SCAMPS reader")
        ppg[start:stop] = block["d_ppg"]
        for offset, index in enumerate(range(start, stop)):
            rgb[:, index], validity[:, index], counts[:, index], coverage[:, index], reasons[:, index], detected[index] = (
                _measure_frame(block["rgb"][offset], detector, index, nominal[index], config, debug_callback))
        # Release BEFORE allocating the next chunk; no frame view escapes _measure_frame.
        del block
        if resource_check is not None:
            resource_check()
    provenance = dict(schema_version=EXTRACTION_SCHEMA, filename=recording.path.name,
                      source_sha256=recording.info["source_sha256"], dataset_id="SCAMPS",
                      extraction_method="v1_landmark_polygons_no_skin_gate",
                      config=asdict(config), roi_landmark_indices={k: list(v) for k, v in FACE_ROIS.items()},
                      source_variable="Xsub", stored_rgb_scale="float_[0,1]_unchanged",
                      measured_rgb_scale="continuous_source_times_255_no_rounding",
                      detector_rgb_conversion="rint(source_times_255)_uint8_copy",
                      segmentation_used=False, mask_alignment="UNVERIFIED",
                      geometric_fallback_used=False, frame_or_label_shift_applied=0,
                      coordinate_kind="documented_nominal_synthetic_frame_grid",
                      nominal_rate_hz=30., capture_timestamps=None, physical_clock_verified=False,
                      subject_adaptation_enabled=False, calibration_frame_indices=[], participant_identity=None,
                      allow_training=False, allow_physiological_timing=False,
                      estimated_streaming_array_peak_bytes=estimated_bytes + persistent_bytes,
                      persistent_trace_bytes=persistent_bytes,
                      detector=getattr(detector, "scamps_provenance", dict(kind="injected_detector",
                          python_type=type(detector).__name__, confidence_kind="not_exposed")),
                      **preprocessing_provenance())
    return ScampsROITraces(rgb, validity, indices, indices.copy(), ppg, counts, coverage,
                          reasons, detected, nominal, provenance)
