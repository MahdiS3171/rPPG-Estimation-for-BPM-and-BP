"""Consistent every-frame, native-pixel acquisition for the authorized pilot.

Decode once; bound pixel-mask calculations spatially without changing pixels.
Historical extractors/caches stay read-only. No participant image is written.
"""
from dataclasses import replace
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .artifacts import file_sha256,write_json
from .config import PipelineConfig
from .detection import FaceRegionDetector
from .pilot_detection import PilotHandDetector
from .pilot_video import mp4_presentation_times
from .video import VideoReader
from .roi import FACE_ROIS
from .pipeline import _observation_quality
from .pilot6_delay_acquisition import polygon_rgb,ROIS

VERSION='pilot11_native_every_frame_single_decode_v1'


def face_pixels(frame,observation,names,min_pixels):
    return {name:polygon_rgb(frame,[observation.landmarks[FACE_ROIS[name]]] if observation.valid and observation.landmarks is not None else [],min_pixels) for name in names}


def acquire_session(video,out,config_dict,resume=False):
    cv2.setNumThreads(1);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    c=dict(config_dict)
    for key in ('regions','skin_cr_range','skin_cb_range','face_rois'):c[key]=tuple(c[key])
    config=replace(PipelineConfig(**c),regions=('face','hand'),detector_backend='tasks',expected_hand='any')
    if config.skin_mask:raise ValueError('This exact polygon equivalence contract requires skin_mask=False')
    video=Path(video);digest=file_sha256(video);t,clock=mp4_presentation_times(video)
    identity=dict(version=VERSION,video_sha256=digest,extraction_config=config.to_dict(),spatial_resize=False,every_frame_detection=True,
        face_asset_sha256=file_sha256(config.face_model),hand_asset_sha256=file_sha256(config.hand_model),
        source_sha256={name:file_sha256(Path(__file__).with_name(name+'.py')) for name in
            ('pilot11_acquisition','pilot6_delay_acquisition','pilot_detection','detection','roi','region_masks','pipeline','video','pilot_video')})
    identity=json.loads(json.dumps(identity));path=out/'native_rgb.npz'
    if path.exists():
        if not resume:raise FileExistsError('Existing acquisition needs explicit resume')
        with np.load(path,allow_pickle=False) as archive:
            if json.loads(str(archive['identity_json'].item()))!=identity:raise ValueError('Existing acquisition identity differs')
        return dict(session_id=video.stem,status='reused_complete',path=str(path),sha256=file_sha256(path))
    reader=VideoReader(video,timestamps=t)
    face=FaceRegionDetector('tasks',config.face_model,config.min_detection_confidence,config.min_tracking_confidence)
    hand=PilotHandDetector(config)
    values={name:[] for name in ROIS};qualities={name:[] for name in ROIS};vertices={name:[] for name in ROIS[:3]}
    face_reasons=[];hand_reasons=[];hand_ids=[];hand_points=[];repeated=[];previous_face={};previous_hand={};last=None
    started=time.monotonic();count=0
    try:
        for index,pts,frame in reader.frames():
            f=face.detect(frame,index,pts);h=hand.detect(frame,index,pts)
            fq=_observation_quality(f,previous_face,'face',config.max_gap_sec);hq=_observation_quality(h,previous_hand,'hand',config.max_gap_sec)
            facial=face_pixels(frame,f,ROIS[:3],config.min_pixels);polygons=[]
            for name in ROIS[:3]:
                rgb,q=facial[name];values[name].append(rgb);qualities[name].append(dict(q,**fq))
                points=f.landmarks[FACE_ROIS[name]].copy() if f.valid and f.landmarks is not None else np.full((len(FACE_ROIS[name]),2),np.nan)
                vertices[name].append(points)
                if np.isfinite(points).all():polygons.append(points)
            rgb,q=polygon_rgb(frame,polygons if len(polygons)==3 else [],config.min_pixels)
            values['combined'].append(rgb);qualities['combined'].append(q)
            for name,scale in (('hand_palm',1.),('hand_center',.65)):
                polygon=[]
                if h.valid and h.landmarks is not None:
                    center=h.landmarks.mean(0);points=center+scale*(h.landmarks-center);polygon=[points[[0,1,2,5,9,13,17]]]
                rgb,q=polygon_rgb(frame,polygon,config.min_pixels);values[name].append(rgb);qualities[name].append(dict(q,**hq))
            face_reasons.append(f.reason or 'ok');hand_reasons.append(h.reason or 'ok');hand_ids.append(h.identity or 'missing')
            hand_points.append(h.landmarks.copy() if h.landmarks is not None else np.full((21,2),np.nan))
            repeated.append(last is not None and np.array_equal(last,frame));last=frame.copy();count+=1
            if count%600==0 or count==1:
                elapsed=time.monotonic()-started
                progress=dict(session_id=video.stem,frames=count,total_frames=len(t),pts_seconds=pts,elapsed_seconds=elapsed,
                    processing_fps=count/max(elapsed,1e-8),frame_shape=list(frame.shape),status='acquiring_native_every_frame')
                write_json(out/'acquisition_progress.json',progress);print(json.dumps(progress),flush=True)
        if count!=len(t):raise ValueError('Decoded frame/PTS count mismatch')
        backend=np.asarray(reader.metadata.backend_timestamps)
        discrepancy=float(abs((backend-backend[0])-(t-t[0])).max())
        if discrepancy>1e-5:raise ValueError('Decoder/container relative timestamps differ')
        if file_sha256(video)!=digest:raise ValueError('Video bytes changed during acquisition')
        arrays=dict(timestamps=t,backend_timestamps=backend,identity_json=np.asarray(json.dumps(identity)),
            face_tracking_reason=np.asarray(face_reasons),hand_tracking_reason=np.asarray(hand_reasons),hand_identity=np.asarray(hand_ids),
            hand_landmarks_native=np.asarray(hand_points),repeated_analysis_frame=np.asarray(repeated),native_shape=np.asarray(last.shape),
            face_backend=np.asarray('tasks'),actual_hand_backend=np.asarray('tasks'))
        for name in ROIS:
            arrays[name+'_rgb']=np.asarray(values[name])
            if name in vertices:arrays[name+'_vertices']=np.asarray(vertices[name])
            for key in qualities[name][0]:arrays[name+'_'+key]=np.asarray([q[key] for q in qualities[name]])
        temporary=path.with_suffix('.partial.npz');np.savez_compressed(temporary,**arrays);temporary.replace(path)
        receipt=dict(identity=identity,frames=count,native_shape=list(last.shape),clock=clock,original_pts_preserved=True,
            decoder_relative_pts_max_error_sec=discrepancy,elapsed_seconds=time.monotonic()-started,
            no_participant_image_pixels_written=True,sha256=file_sha256(path))
        write_json(out/'acquisition.json',receipt);write_json(out/'acquisition_progress.json',dict(session_id=video.stem,frames=count,total_frames=len(t),status='complete',elapsed_seconds=receipt['elapsed_seconds']))
        return dict(session_id=video.stem,status='complete',path=str(path),sha256=receipt['sha256'],seconds=receipt['elapsed_seconds'])
    finally:
        reader.close();face.close();hand.close()
