"""Single decoded pass for face and hand; timing uses paired segment boundaries."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping
import logging
import numpy as np
from .config import PipelineConfig
from .detection import BaseRegionDetector, FaceRegionDetector, HandRegionDetector
from .region_masks import get_region_mask, extract_frame_rgb
from .types import RGBTrace, RegionObservation, RegionResult, SharedTimebase, DualROIResult, PhysiologicalSignal
from .video import VideoReader
from .processing import shared_grid, resample_trace, extract_signal
from .quality import region_quality
from .bp import FEATURE_KEYS, extract_features

LOG = logging.getLogger(__name__)


def _detectors(config: PipelineConfig) -> dict[str, BaseRegionDetector]:
    detectors = {}
    try:
        if "face" in config.regions:
            detectors["face"] = FaceRegionDetector(config.detector_backend, config.face_model,
                config.min_detection_confidence, config.min_tracking_confidence)
        if "hand" in config.regions:
            detectors["hand"] = HandRegionDetector(config.detector_backend, config.hand_model,
                config.expected_hand, config.input_mirrored, config.max_hand_jump_fraction,
                config.min_detection_confidence, config.min_tracking_confidence)
    except Exception:
        for detector in detectors.values():
            detector.close()
        raise
    return detectors


def _feature_windows(face: RegionResult, hand: RegionResult, rgb: dict, imputed: dict,
                     metadata, grid: np.ndarray, config: PipelineConfig) -> list[dict]:
    joint = np.isfinite(rgb["face"]).all(axis=1) & np.isfinite(rgb["hand"]).all(axis=1)
    # Re-extract with identical paired boundaries to prevent different gap-edge
    # filters from becoming an apparent inter-site lag. Individual HR signals
    # remain available even when the other site is absent.
    paired = {site: extract_signal(rgb[site], grid, site, config, allowed=joint) for site in ("face", "hand")}
    count = max(1, int(round(config.window_sec * config.sample_rate)))
    stride = max(1, int(round(config.stride_sec * config.sample_rate)))
    rows = []
    nominal = any(s.startswith("nominal") for s in metadata.timestamp_sources)
    for start in range(0, len(grid)-count+1, stride):
        end = start+count
        a,b = float(grid[start]), float(grid[end-1]+1/config.sample_rate)
        row = {"window_index": len(rows), "start_sec": a, "end_sec": b,
               "status": "excluded", "reasons": [], **dict.fromkeys(FEATURE_KEYS, float("nan"))}
        for site, region in (("face",face), ("hand",hand)):
            source = (region.rgb.timestamps >= a) & (region.rgb.timestamps < b)
            fraction = float(np.mean(region.rgb.validity_mask[source])) if source.any() else 0.0
            row[site+"_valid_frame_fraction"] = fraction
            row[site+"_interpolated_fraction"] = float(np.mean(imputed[site][start:end]))
            if fraction < config.min_valid_fraction:
                row["reasons"].append(site+"_insufficient_valid_frames")
            if not np.isfinite(paired[site].values[start:end]).all():
                row["reasons"].append(site+"_missing_signal_or_filter_edge")
        if nominal and not config.allow_nominal_timing:
            row["reasons"].append("unverified_nominal_timestamps")
        # Compute inspectable signal features when finite, even if quality or
        # timestamp provenance later excludes this window from model training.
        if all(np.isfinite(paired[site].values[start:end]).all() for site in ("face", "hand")):
            signals = [PhysiologicalSignal(paired[site].values[start:end], grid[start:end], config.sample_rate,
                        site, config.method) for site in ("face", "hand")]
            features = extract_features(*signals, max_delay_sec=config.max_delay_sec,
                                        fmin=config.hr_min_hz, fmax=config.hr_max_hz)
            for key,value in features.items():
                if not key.endswith(("valid_frame_fraction", "interpolated_fraction")):
                    row[key] = value
            for site in ("face", "hand"):
                if not np.isfinite(row[site+"_snr_db"]) or row[site+"_snr_db"] < config.min_snr_db:
                    row["reasons"].append(site+"_low_signal_quality")
            if not np.isfinite(row["delay_xcorr_sec"]):
                row["reasons"].append("unresolved_delay")
            if not np.isfinite(row["delay_xcorr_score"]) or row["delay_xcorr_score"] < config.min_delay_correlation:
                row["reasons"].append("low_delay_correlation")
            if not np.isfinite(row["face_hand_hr_difference_bpm"]) or row["face_hand_hr_difference_bpm"] > config.max_hr_difference_bpm:
                row["reasons"].append("face_hand_hr_inconsistent")
        row["status"] = "accepted" if not row["reasons"] else "excluded"
        rows.append(row)
    return rows


def process_recording(video: str | Path, config: PipelineConfig | None = None,
                      *, detectors: Mapping[str, BaseRegionDetector] | None = None,
                      timestamps: np.ndarray | None = None, max_frames: int | None = None,
                      debug_callback: Callable | None = None) -> DualROIResult:
    """Extract synchronized sites. Injected detectors are owned by their caller.

    Stored RGB uses every original decoded timestamp. Stored rPPG uses one shared
    uniform grid with explicit missing samples. Offline phase/timing fidelity
    remains a hypothesis to validate, especially for adaptive classical methods.
    """
    config = config or PipelineConfig()
    reader = VideoReader(video, config.timestamp_mode, timestamps)
    owned = detectors is None
    try:
        active = _detectors(config) if owned else dict(detectors)
    except Exception:
        reader.close()
        raise
    if not set(config.regions) <= set(active):
        reader.close()
        raise ValueError("A detector is required for every selected region")
    observations = {site: [] for site in ("face", "hand")}
    values = {site: [] for site in observations}
    metrics = {site: [] for site in observations}
    previous = {}
    try:
        for index,t,frame in reader.frames(max_frames):
            masks = {}
            for site in observations:
                obs = active[site].detect(frame,index,t) if site in config.regions else RegionObservation(index,t,reason="region_not_requested")
                if obs.frame_index != index or obs.timestamp != t:
                    raise ValueError("Detector changed the original frame/time identity")
                strategy = config.face_roi if site == "face" else config.hand_roi
                mask = get_region_mask(frame, obs, site, strategy, config.skin_mask, config.skin_cr_range, config.skin_cb_range)
                color, q = extract_frame_rgb(frame, mask, config.min_pixels)
                q["detector_confidence"] = obs.confidence
                q["motion_px_per_sec"] = float("nan")
                if obs.valid and obs.landmarks is not None:
                    center = obs.landmarks.mean(axis=0)
                    if site in previous:
                        pt,pc = previous[site]
                        # Do not pretend motion over a lost-tracking gap is a
                        # continuously observed trajectory.
                        if t-pt <= config.max_gap_sec:
                            q["motion_px_per_sec"] = float(np.linalg.norm(center-pc)/(t-pt))
                    previous[site] = t,center
                if obs.valid and not np.isfinite(color).all():
                    obs.reason = "insufficient_skin_pixels"
                values[site].append(color)
                metrics[site].append(q)
                observations[site].append(obs)
                masks[site] = mask
            if debug_callback is not None and config.debug_every and index % config.debug_every == 0:
                debug_callback(index,t,frame,masks,{site: observations[site][-1] for site in observations})
    finally:
        reader.close()
        if owned:
            for detector in active.values():
                detector.close()
    metadata = reader.metadata
    source_t = metadata.timestamps
    if len(source_t) > 1:
        acquired_fs = 1 / np.median(np.diff(source_t))
        if config.sample_rate < 0.95*acquired_fs:
            raise ValueError(f"Downsampling {acquired_fs:.3f} to {config.sample_rate} Hz needs antialiasing; set sample_rate to capture rate or higher")
    grid = shared_grid(source_t, config.sample_rate)
    regions, uniform, imputed = {}, {}, {}
    exclusions = []
    if metadata.decoded_frame_shortfall:
        exclusions.append({"reason": "decoded_frame_count_below_container_report", "count": metadata.decoded_frame_shortfall,
                           "note": "Reported frame count can be inaccurate; cannot infer original capture drops from this alone"})
    for site in observations:
        colors = np.asarray(values[site])
        raw_q = {k: np.asarray([q[k] for q in metrics[site]]) for k in metrics[site][0]}
        trace = RGBTrace(source_t, colors, site, np.isfinite(colors).all(axis=1), raw_q)
        uniform[site], imputed[site] = resample_trace(trace, grid, config.max_gap_sec)
        signal = extract_signal(uniform[site], grid, site, config)
        quality = region_quality(trace, signal, imputed[site], config.min_valid_fraction,
                                 config.min_snr_db, config.hr_min_hz, config.hr_max_hz)
        signal.quality = quality
        regions[site] = RegionResult(trace, signal, quality, observations[site])
        if site in config.regions:
            exclusions.extend({"region": site, "reason": reason} for reason in quality["reasons"])
            exclusions.extend({"region": site, **failure} for failure in signal.preprocessing["failures"])
    rows = _feature_windows(regions["face"],regions["hand"],uniform,imputed,metadata,grid,config) if set(config.regions) == {"face","hand"} else []
    exclusions.extend({"window_index": row["window_index"], "start_sec": row["start_sec"],
                       "end_sec": row["end_sec"], "reasons": row["reasons"]} for row in rows if row["status"] == "excluded")
    if set(config.regions) == {"face", "hand"} and not rows:
        exclusions.append({"reason": "insufficient_duration_for_feature_window"})
    delays = [row["delay_xcorr_sec"] for row in rows if row["status"] == "accepted"]
    delay = {"face_hand_delay_median_sec": float(np.median(delays)) if delays else float("nan"),
             "face_hand_delay_std_sec": float(np.std(delays)) if delays else float("nan"),
             "accepted_window_count": len(delays), "positive_means": "hand_lags_face",
             "interpretation": "experimental_optical_inter_site_delay", "acquisition_interval_median_sec":
             float(np.median(np.diff(source_t))) if len(source_t)>1 else None}
    LOG.info("Decoded %d frames; face valid %.1f%%, hand valid %.1f%%; accepted %d timing windows",
             len(source_t),100*regions["face"].quality["valid_frame_fraction"],
             100*regions["hand"].quality["valid_frame_fraction"],len(delays))
    return DualROIResult(regions["face"],regions["hand"],SharedTimebase(grid,source_t),metadata,
                         {site: regions[site].quality["hr_bpm"] for site in regions},delay,rows,exclusions,config.to_dict())
