"""Native-pixel hand acquisition with identity-checked, read-only native face reuse."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .artifacts import file_sha256, write_json
from .pilot_detection import PilotHandDetector
from .pilot_frozen_extraction import extract_native_face
from .pilot_video import mp4_presentation_times
from .pipeline import _observation_quality
from .region_masks import _polygon, extract_frame_rgb
from .video import VideoReader

ROIS = ("forehead", "left_cheek", "right_cheek", "combined", "hand_palm", "hand_center")
VERSION = "native_pilot6_protocol_delay_rgb_v1"


def polygon_rgb(frame, polygons, min_pixels=50):
    """Exact original rounded hull union, bounded spatial work; no resizing."""
    h, w = frame.shape[:2]
    if not polygons or not all(np.isfinite(p).all() for p in polygons):
        return extract_frame_rgb(frame[:1, :1], np.zeros((1, 1), bool), min_pixels)
    rounded = [np.rint(p).astype(np.int32) for p in polygons]
    for p in rounded:
        p[:, 0] = np.clip(p[:, 0], 0, w - 1)
        p[:, 1] = np.clip(p[:, 1], 0, h - 1)
    all_points = np.concatenate(rounded)
    lo, hi = all_points.min(axis=0), all_points.max(axis=0) + 1
    x, y = lo
    mask = np.zeros((hi[1] - y, hi[0] - x), np.uint8)
    for p in rounded:
        _polygon(mask, p - lo)
    rgb, metrics = extract_frame_rgb(frame[y:hi[1], x:hi[0]], mask.astype(bool), min_pixels)
    metrics["mask_coverage"] = metrics["mask_pixel_count"] / (h * w)
    return rgb, metrics


def acquire(video, out, frozen_config, face_cache_root, resume=False):
    cv2.setNumThreads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "native_rgb.npz"
    config = replace(frozen_config, regions=("face", "hand"), expected_hand="any")
    identity = dict(version=VERSION, video_sha256=file_sha256(video), spatial_resize=False,
                    source_sha256={name: file_sha256(Path(__file__).with_name(name + ".py")) for name in
                                   ("pilot6_delay_acquisition", "pilot_detection", "detection", "region_masks", "roi", "video")},
                    extraction_config=config.to_dict(),
                    hand_asset_sha256=file_sha256(config.hand_model))
    if cache.exists():
        if not resume:
            raise FileExistsError(f"Native acquisition exists: {cache}")
        with np.load(cache, allow_pickle=False) as z:
            if json.loads(str(z["identity_json"].item())) != json.loads(json.dumps(identity)):
                raise ValueError("Native RGB cache identity differs")
            return {k: z[k].copy() for k in z.files}
    face_path = Path(face_cache_root) / Path(video).stem / "native_face_rgb.npz"
    # The historical extractor validates its full original extraction identity,
    # model assets, source/dependency hashes and video bytes before returning.
    # Missing caches are acquired ONLY inside this new run.
    face_out = face_path.parent if face_path.is_file() else out / "face_acquisition"
    face = extract_native_face(video, face_out, frozen_config, reuse=face_path.is_file())
    t, clock = mp4_presentation_times(video)
    if not np.array_equal(t, face["camera_pts"]):
        raise ValueError("Face/container PTS mismatch")
    reader, detector = VideoReader(video, timestamps=t), PilotHandDetector(config)
    arrays = dict(timestamps=t, identity_json=np.asarray(json.dumps(identity)),
                  face_tracking_reason=face["tracking_reason"],
                  repeated_analysis_frame=face["repeated_decoded_frame"],
                  face_backend=face["actual_backend"])
    for roi in ROIS[:3]:
        arrays.update({k: v for k, v in face.items() if k.startswith(roi + "_")})
    values, metrics = {r: [] for r in ROIS[3:]}, {r: [] for r in ROIS[3:]}
    identities, reasons, landmarks, previous = [], [], [], {}
    started = time.monotonic()
    count = 0
    try:
        for index, pts, frame in reader.frames():
            observation = detector.detect(frame, index, pts)
            common = _observation_quality(observation, previous, "hand", config.max_gap_sec)
            polygons = [face[r + "_vertices"][index] for r in ROIS[:3]]
            rgb, q = polygon_rgb(frame, polygons, config.min_pixels)
            values["combined"].append(rgb)
            metrics["combined"].append(q)
            points = observation.landmarks
            for roi, scale in (("hand_palm", 1.), ("hand_center", .65)):
                poly = []
                if observation.valid and points is not None:
                    p = points.mean(axis=0) + scale * (points - points.mean(axis=0))
                    poly = [p[[0, 1, 2, 5, 9, 13, 17]]]
                rgb, q = polygon_rgb(frame, poly, config.min_pixels)
                values[roi].append(rgb)
                metrics[roi].append(dict(q, **common))
            landmarks.append(points.copy() if points is not None else np.full((21, 2), np.nan))
            identities.append(observation.identity or "missing")
            reasons.append(observation.reason or "ok")
            count += 1
            if index % 600 == 0:
                progress = dict(session_id=Path(video).stem, frames=count, total_frames=len(t),
                                pts_sec=pts, elapsed_sec=time.monotonic() - started, native_shape=frame.shape,
                                hand_backend=detector.backend, face_cache=str(face_out))
                write_json(out / "acquisition_progress.json", progress)
                print(json.dumps(progress), flush=True)
        if count != len(t):
            raise ValueError(f"Incomplete decoding: {count}/{len(t)}")
        arrays.update(backend_timestamps=np.asarray(reader.metadata.backend_timestamps),
                      hand_identity=np.asarray(identities), hand_tracking_reason=np.asarray(reasons),
                      hand_landmarks_native=np.asarray(landmarks), native_shape=np.asarray(frame.shape),
                      actual_hand_backend=np.asarray(detector.backend))
        for roi in values:
            arrays[roi + "_rgb"] = np.asarray(values[roi])
            for key in metrics[roi][0]:
                arrays[roi + "_" + key] = np.asarray([r[key] for r in metrics[roi]])
        backend = arrays["backend_timestamps"]
        error = float(np.max(np.abs((backend - backend[0]) - (t - t[0]))))
        if error > 1e-5:
            raise ValueError(f"Decoder/container relative PTS discrepancy {error}")
        tmp = cache.with_suffix(".partial.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(cache)
        write_json(out / "acquisition.json", dict(identity=identity, frames=count, native_shape=frame.shape,
            original_pts_preserved=True, clock=clock, decoder_relative_pts_max_error_sec=error,
            face_cache_path=str(face_path if face_path.is_file() else face_out / "native_face_rgb.npz"),
            face_cache_sha256=file_sha256(face_path if face_path.is_file() else face_out / "native_face_rgb.npz"),
            elapsed_sec=time.monotonic() - started, hand_backend=detector.backend,
            hand_side_note="Protocol right hand. Detector identity locked; mirror convention unverified; classification is not anatomical confirmation.",
            acquisition_note="Full native frames, every PTS; no participant image pixels saved; geometry is not semantic skin validation."))
        return arrays
    finally:
        reader.close()
        detector.close()
