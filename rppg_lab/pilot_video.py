"""Pilot-only raw extraction. Historical extraction/model contracts are untouched.

Classical diagnostics may use spatially resized frames, but never temporal
decimation. Native-resolution extraction is mandatory for frozen inference.
"""
from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
from pathlib import Path
import struct
import time

import cv2
import numpy as np

from .artifacts import file_sha256, write_json
from .config import PipelineConfig
from .detection import FaceRegionDetector, HandRegionDetector
from .pilot_detection import PilotHandDetector
from .pipeline import _observation_quality
from .region_masks import get_face_masks, get_region_mask, extract_frame_rgb_by_roi
from .types import RGBTrace, validate_timestamps
from .video import VideoReader

FACE_NAMES = ("forehead", "left_cheek", "right_cheek", "combined")
ROI_NAMES = (*FACE_NAMES, "cheeks", "hand_palm", "hand_center")


def _boxes(data, start=0, end=None):
    end = len(data) if end is None else end
    while start + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, start)
        head = 8
        if size == 1:
            size = struct.unpack_from(">Q", data, start + 8)[0]
            head = 16
        elif size == 0:
            size = end - start
        if size < head or start + size > end:
            raise ValueError("Malformed MP4 box")
        yield kind, start + head, start + size
        start += size


def _child(data, begin, end, name):
    return next(((a,b) for kind,a,b in _boxes(data, begin, end) if kind == name), None)


def mp4_presentation_times(path):
    """Nonfragmented MP4 stts+ctts clock in display order, normalized at first PTS.

    Only a unique video track is supported. Complex edits/fragments are rejected
    rather than interpreted using guessed FPS. Original PTS ticks are retained.
    """
    moov = None
    with Path(path).open("rb") as stream:
        total = Path(path).stat().st_size
        while stream.tell() + 8 <= total:
            offset = stream.tell()
            size, kind = struct.unpack(">I4s", stream.read(8))
            head = 8
            if size == 1:
                size = struct.unpack(">Q", stream.read(8))[0]
                head = 16
            elif size == 0:
                size = total - offset
            if size < head or offset + size > total:
                raise ValueError("Malformed MP4 top-level box")
            if kind == b"moof":
                raise ValueError("Fragmented MP4 timing not supported by native parser")
            if kind == b"moov":
                if size > 100_000_000:
                    raise ValueError("Oversized MP4 metadata")
                moov = stream.read(size - head)
            stream.seek(offset + size)
    if moov is None:
        raise ValueError("No nonfragmented MP4 movie table")
    mvhd = _child(moov,0,len(moov),b"mvhd")
    movie_scale = None
    if mvhd:
        p = mvhd[0]
        movie_scale = struct.unpack_from(">I",moov,p+(20 if moov[p] == 1 else 12))[0]
    tracks = []
    for kind,a,b in _boxes(moov):
        if kind != b"trak":
            continue
        mdia = _child(moov,a,b,b"mdia")
        if mdia is None:
            continue
        hdlr = _child(moov,*mdia,b"hdlr")
        if hdlr is None or moov[hdlr[0]+8:hdlr[0]+12] != b"vide":
            continue
        mdhd = _child(moov,*mdia,b"mdhd")
        p = mdhd[0]
        scale = struct.unpack_from(">I",moov,p+(20 if moov[p] == 1 else 12))[0]
        minf = _child(moov,*mdia,b"minf")
        stbl = _child(moov,*minf,b"stbl")
        stts = _child(moov,*stbl,b"stts")
        if stts is None or not scale:
            raise ValueError("No complete MP4 sample clock")
        p = stts[0]
        count = struct.unpack_from(">I",moov,p+4)[0]
        durations = []
        for index in range(count):
            n, delta = struct.unpack_from(">II",moov,p+8+8*index)
            if n > 10_000_000 or delta == 0:
                raise ValueError("Invalid MP4 sample durations")
            durations.extend([delta]*n)
        durations = np.asarray(durations,dtype=np.int64)
        if not len(durations):
            raise ValueError("Empty MP4 sample table")
        dts = np.r_[0,np.cumsum(durations[:-1])]
        offsets = np.zeros(len(dts),dtype=np.int64)
        ctts = _child(moov,*stbl,b"ctts")
        if ctts:
            p = ctts[0]
            count = struct.unpack_from(">I",moov,p+4)[0]
            parts = []
            for index in range(count):
                n, delta = struct.unpack_from(">Ii" if moov[p] == 1 else ">II",moov,p+8+8*index)
                parts.extend([delta]*n)
            if len(parts) != len(dts):
                raise ValueError("MP4 composition/sample count mismatch")
            offsets = np.asarray(parts,dtype=np.int64)
        edits = []
        edts = _child(moov,a,b,b"edts")
        if edts:
            elst = _child(moov,*edts,b"elst")
            if elst:
                p = elst[0]
                n = struct.unpack_from(">I",moov,p+4)[0]
                width = 20 if moov[p] == 1 else 12
                for index in range(n):
                    values = struct.unpack_from(">Qqhh" if moov[p] == 1 else ">Iihh",moov,p+8+width*index)
                    edits.append(values)
                simple = n == 1 and edits[0][1] >= 0
                empty_lead = n == 2 and edits[0][1] == -1 and edits[1][1] == 0 and movie_scale
                if not (simple or empty_lead) or any(v[2:] != (1,0) for v in edits):
                    raise ValueError(f"Complex MP4 edits require ffprobe/PyAV timing validation: {edits}")
        ticks = np.sort(dts + offsets)
        lead = edits[0][0]/movie_scale if len(edits) == 2 else 0.
        media_start = edits[-1][1] if edits else int(ticks[0])
        if media_start != ticks[0]:
            raise ValueError("Trimmed media edit needs decoder/frame-count validation")
        relative = (ticks - media_start)/scale+lead
        validate_timestamps(relative)
        tracks.append((relative, dict(timescale=scale, first_pts_tick=int(ticks[0]),
                                      edit_list=edits, empty_lead_sec=lead, movie_timescale=movie_scale, sample_count=len(ticks),
                                      sample_duration_sec=float(durations.sum()/scale))))
    if len(tracks) != 1:
        raise ValueError("Require one unambiguous MP4 video track")
    return tracks[0]


def inspect_video(path):
    reader = VideoReader(path)
    try:
        m = reader.metadata
        info = dict(filename=Path(path).name, bytes=Path(path).stat().st_size, width=m.width,
                    height=m.height, codec=m.codec, nominal_fps=m.fps_nominal,
                    frame_count_reported=m.frame_count_reported, rotation_degrees=m.orientation_degrees,
                    nominal_duration_sec=m.frame_count_reported/m.fps_nominal if m.fps_nominal > 0 else None,
                    repeated_pixel_frames="not_assessed_in_audit", capture_drops="not_observable_without_capture_log")
        try:
            pts, details = mp4_presentation_times(path)
            dt = np.diff(pts)
            info.update(pts_source="MP4 stts+ctts; display order; first PTS normalized to zero",
                        timing=details, pts_duration_sec=float(pts[-1]),
                        median_frame_interval_sec=float(np.median(dt)),
                        min_frame_interval_sec=float(dt.min()), max_frame_interval_sec=float(dt.max()),
                        variable_frame_intervals=bool(np.ptp(dt) > 1e-5),
                        long_intervals_over_1_5_median=int(np.sum(dt > 1.5*np.median(dt))))
        except (ValueError,struct.error,TypeError) as exc:
            info.update(pts_source="unavailable", pts_error=str(exc))
        return info
    finally:
        reader.close()


def raw_cache_identity(video, max_frames, max_dimension, config):
    return dict(schema="pilot6_raw_v1", input_sha256=file_sha256(video), max_frames=max_frames,
                max_dimension=max_dimension, config=config.to_dict(),
                source_sha256={name:file_sha256(Path(__file__).with_name(name + ".py")) for name in
                               ("pilot_video", "pilot_detection", "detection", "region_masks", "roi", "pipeline", "video", "types")},
                libraries={name:version(name) for name in ("numpy","opencv-python","mediapipe")},
                model_sha256={s:file_sha256(p) for s,p in (("face",config.face_model),("hand",config.hand_model))})


def extract_raw(video, out, config, max_frames=None, max_dimension=640, reuse=False, private_overlays=False):
    """Every decoded frame on its original display clock; one pass for all ROIs."""
    cv2.setNumThreads(1)
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    identity = raw_cache_identity(video,max_frames,max_dimension,config)
    cache = out/"raw_rgb.npz"
    if reuse and cache.is_file():
        with np.load(cache,allow_pickle=False) as data:
            if json.loads(str(data["identity_json"].item())) != json.loads(json.dumps(identity)):
                raise ValueError("Raw cache identity changed; use a new output directory")
            return {key:data[key].copy() for key in data.files}
    if cache.exists():
        raise FileExistsError("Raw extraction exists; use --reuse or a new output namespace")
    pts, clock_details = mp4_presentation_times(video)
    reader = VideoReader(video,timestamps=pts)
    face = FaceRegionDetector(config.detector_backend,config.face_model,
                              config.min_detection_confidence,config.min_tracking_confidence)
    try:
        hand = PilotHandDetector(config)
    except Exception:
        face.close(); reader.close()
        raise
    values, metrics = {n:[] for n in ROI_NAMES}, {n:[] for n in ROI_NAMES}
    coordinates = {"face":[],"hand":[]}
    reasons, identities = {"face":[],"hand":[]}, []
    previous = {}
    repeated, last = [], None
    began = time.monotonic()
    detection_pool = ThreadPoolExecutor(max_workers=2)
    try:
        for index,t,frame in reader.frames(max_frames):
            original_shape = frame.shape[:2]
            if max_dimension and max(frame.shape[:2]) > max_dimension:
                scale = max_dimension/max(frame.shape[:2])
                frame = cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            scale_xy = np.array([original_shape[1]/frame.shape[1],original_shape[0]/frame.shape[0]])
            # Exact repeated decoded analysis frame; does not imply capture drop.
            repeated.append(last is not None and np.array_equal(frame,last))
            last = frame.copy()
            pending = {"face":detection_pool.submit(face.detect,frame,index,t),
                       "hand":detection_pool.submit(hand.detect,frame,index,t)}
            obs = {site:future.result() for site,future in pending.items()}
            masks = get_face_masks(frame,obs["face"],FACE_NAMES,config.skin_mask,
                                   config.skin_cr_range,config.skin_cb_range)
            masks["cheeks"] = masks["left_cheek"] | masks["right_cheek"]
            masks["hand_palm"] = get_region_mask(frame,obs["hand"],"hand","palm",config.skin_mask,
                                                  config.skin_cr_range,config.skin_cb_range)
            central = obs["hand"]
            if central.landmarks is not None:
                points = central.landmarks.copy()
                points = points.mean(axis=0)+0.65*(points-points.mean(axis=0))
                central = replace(central,landmarks=points)
            masks["hand_center"] = get_region_mask(frame,central,"hand","palm",config.skin_mask,
                                                    config.skin_cr_range,config.skin_cb_range)
            common = {site:_observation_quality(o,previous,site,config.max_gap_sec) for site,o in obs.items()}
            for name,(color,q) in extract_frame_rgb_by_roi(frame,masks,config.min_pixels).items():
                site = "hand" if name.startswith("hand_") else "face"
                q.update(common[site])
                values[name].append(color)
                metrics[name].append(q)
            for site,o in obs.items():
                count = 478 if site == "face" else 21
                points = np.full((count,2),np.nan)
                if o.landmarks is not None:
                    points[:min(count,len(o.landmarks))] = o.landmarks[:count]*scale_xy
                coordinates[site].append(points)
                reasons[site].append(o.reason or "ok")
            identities.append(obs["hand"].identity or "missing")
            if index % 1800 == 0:
                # Default artifact shows exact contributing masks without participant pixels.
                canvas = np.zeros_like(frame)
                colors = [(220,60,60),(60,220,60),(60,60,220),(200,140,60),(80,160,200),(200,200,50),(200,50,200)]
                for (name,mask),color in zip(masks.items(),colors):
                    cv2.imwrite(str(out/f"mask_{name}_{index:06d}.png"),mask.astype(np.uint8)*255)
                    canvas[mask] = color
                cv2.imwrite(str(out/f"roi_geometry_{index:06d}.png"),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
                if private_overlays:
                    private = out/"private_overlays"
                    private.mkdir(exist_ok=True)
                    overlay = frame.copy()
                    active = np.logical_or.reduce(list(masks.values()))
                    overlay[active] = (0.5*overlay[active]+0.5*canvas[active]).astype(np.uint8)
                    cv2.imwrite(str(private/f"overlay_{index:06d}.png"),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
                print(f"{Path(video).stem}: {index+1} frames, t={t:.1f}s, elapsed={time.monotonic()-began:.1f}s",flush=True)
    finally:
        detection_pool.shutdown(wait=True)
        face.close(); hand.close(); reader.close()
    source = reader.metadata.timestamps
    arrays = dict(timestamps=source, backend_timestamps=np.asarray(reader.metadata.backend_timestamps),
                  repeated_analysis_frame=np.asarray(repeated), hand_identity=np.asarray(identities),
                  identity_json=np.asarray(json.dumps(identity)),
                  analysis_shape=np.asarray(frame.shape[:2]), original_shape=np.asarray(original_shape))
    for name in ROI_NAMES:
        arrays[name+"_rgb"] = np.asarray(values[name])
        for key in metrics[name][0]:
            arrays[name+"_"+key] = np.asarray([m[key] for m in metrics[name]])
    for site in coordinates:
        arrays[site+"_landmarks_original_pixels"] = np.asarray(coordinates[site],dtype=np.float32)
        arrays[site+"_tracking_reason"] = np.asarray(reasons[site])
    np.savez_compressed(cache,**arrays)
    write_json(out/"extraction.json",dict(decoded_frames=len(source), truncated=reader.metadata.truncated,
               elapsed_sec=time.monotonic()-began, timestamp_source="MP4 presentation times",
               clock=clock_details, analysis_shape=frame.shape[:2], original_shape=original_shape,
               spatial_resize=max_dimension, temporal_decimation=False,
               hand_candidates=1, detector_execution="Independent face/hand graphs run concurrently; frame order serial within each graph",
               repeated_analysis_frames=int(np.sum(repeated)),
               detector_confidence_note="Face score unavailable; hand score is handedness classification, not tracking confidence",
               geometry="Existing facial polygons; central-hand hull and contracted central-hand hull. Semantic skin/hair/occlusion exclusion is unvalidated.",
               masks="Sparse exact analysis masks; all-frame landmarks in original pixel coordinates; no participant pixels by default"))
    return arrays


def traces_from_arrays(arrays):
    t = arrays["timestamps"]
    traces = {}
    for name in ROI_NAMES:
        rgb = arrays[name+"_rgb"]
        quality = {key[len(name)+1:]:value for key,value in arrays.items()
                   if key.startswith(name+"_") and key != name+"_rgb"}
        traces[name] = RGBTrace(t,rgb,name,np.isfinite(rgb).all(axis=1),quality)
    return traces
