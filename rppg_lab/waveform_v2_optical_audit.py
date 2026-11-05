"""Same-frame A/B collection with no reference-PPG dataset reads."""
from dataclasses import asdict
import time

import h5py
import numpy as np

from .waveform_v2_scamps_roi import (ROI_NAMES, RGB_CHANNELS, ScampsROIConfig, ScampsROITraces,
                                    _measure_frame, preprocessing_provenance)
from .waveform_v2_optical_quality import METRICS, OpticalQualityConfig, optical_measurement
from .waveform_v2_micro_rois import partition_polygon

SAMPLE_INDICES=(0,149,299,449,599)


def collect_optical_evidence(recording, detector, *, chunk_frames=32, array_budget_bytes=192*1024**2,
                             quality_config=OpticalQualityConfig(), sample_indices=SAMPLE_INDICES,
                             resource_check=None):
    """Original Gate 06 Baseline A measurements plus separate patch observations.

    Only Xsub values are read. Reference labels are not decoded. NaN target slots
    in the legacy-compatible trace object explicitly mean NOT LOADED; only x
    hashes are subsequently checked, never labels or prediction performance.
    """
    c,w,h,n=recording.info['datasets']['Xsub']['shape']
    if c!=3 or chunk_frames<1 or chunk_frames>recording.max_window_frames:
        raise ValueError('Invalid source/chunk contract')
    if any(type(i) is not int or not 0<=i<n for i in sample_indices):
        raise ValueError('Contact-sheet sample indices outside source')
    if 3*chunk_frames*w*h*3*8 + n*3*6*(3*8+len(METRICS)*8+1) > array_budget_bytes:
        raise ValueError('Declared array budget too small before frame allocation')
    rgb=np.full((3,n,3),np.nan);valid=np.zeros((3,n),dtype=bool)
    counts=np.zeros((3,n),dtype=np.int32);coverage=np.zeros((3,n))
    reasons=np.full((3,n),'unprocessed',dtype='U80');detected=np.zeros(n,dtype=bool)
    patch_rgb=np.full((3,6,n,3),np.nan);patch_valid=np.zeros((3,6,n),dtype=bool)
    roi_metrics={k:np.full((3,n),np.nan) for k in METRICS}
    patch_metrics={k:np.full((3,6,n),np.nan) for k in METRICS}
    contacts={}
    clocks=dict(frame_read_seconds=0.,landmarks_and_A_measurement_seconds=0.,
                A_optical_descriptors_seconds=0.,B_patch_descriptors_seconds=0.)
    config=ScampsROIConfig(chunk_frames=chunk_frames,array_budget_bytes=array_budget_bytes)
    for start in range(0,n,chunk_frames):
        if resource_check:resource_check()
        stop=min(start+chunk_frames,n)
        recording._bounds(start,stop,n,w*h*3*8)
        read_start=time.perf_counter()
        with h5py.File(recording.path,'r',rdcc_nbytes=1024**2) as f:
            block=f['Xsub'][:,:,:,start:stop].transpose(3,2,1,0).copy()
        clocks['frame_read_seconds']+=time.perf_counter()-read_start
        if block.shape!=(stop-start,h,w,3):raise ValueError('Unexpected decoded frame shape')
        for offset,index in enumerate(range(start,stop)):
            callback_seconds=[0.]
            def observe(i,geometry,masks):
                callback_start=time.perf_counter()
                original=block[offset]
                frame255=original*255.
                if i in sample_indices:
                    contacts[i]=dict(rgb=original.copy(),masks={k:v.copy() for k,v in masks.items()})
                for r,name in enumerate(ROI_NAMES):
                    a_start=time.perf_counter()
                    _,_,metrics=optical_measurement(frame255,masks[name],50)
                    for k in METRICS:roi_metrics[k][r,i]=metrics[k]
                    clocks['A_optical_descriptors_seconds']+=time.perf_counter()-a_start
                    b_start=time.perf_counter()
                    for p,mask in enumerate(partition_polygon(masks[name])):
                        color,available,metrics=optical_measurement(frame255,mask,quality_config.patch_min_pixels)
                        patch_rgb[r,p,i],patch_valid[r,p,i]=color,available
                        for k in METRICS:patch_metrics[k][r,p,i]=metrics[k]
                    clocks['B_patch_descriptors_seconds']+=time.perf_counter()-b_start
                callback_seconds[0]=time.perf_counter()-callback_start
            measured_start=time.perf_counter()
            rgb[:,index],valid[:,index],counts[:,index],coverage[:,index],reasons[:,index],detected[index]=(
                _measure_frame(block[offset],detector,index,index/30.,config,observe))
            clocks['landmarks_and_A_measurement_seconds']+=time.perf_counter()-measured_start-callback_seconds[0]
            if index in sample_indices and index not in contacts:
                contacts[index]=dict(rgb=block[offset].copy(),masks=None,missing_reason=str(reasons[0,index]))
        del block
        if resource_check:resource_check()
    provenance=dict(filename=recording.path.name,source_sha256=recording.info['source_sha256'],
        coordinate_kind='documented_nominal_synthetic_frame_grid',nominal_rate_hz=30.,capture_timestamps=None,
        physical_clock_verified=False,segmentation_used=False,subject_adaptation_enabled=False,allow_training=False,
        reference_values_decoded=False,quality_config=asdict(quality_config),
        detector=getattr(detector,'scamps_provenance',{'kind':'injected_test_detector'}),**preprocessing_provenance())
    traces=ScampsROITraces(rgb,valid,np.arange(n),np.arange(n),np.full(n,np.nan),counts,coverage,reasons,
                         detected,np.arange(n)/30.,provenance)
    return dict(baseline=traces,patch_rgb=patch_rgb,patch_valid=patch_valid,roi_metrics=roi_metrics,
                patch_metrics=patch_metrics,contacts=contacts,runtime=clocks,
                quality_schema=dict(acquisition='finite_landmark_polygon_support',
                    anatomical_skin_confidence='UNKNOWN_NO_INDEPENDENT_LABELS',
                    optical_reliability='separate_unvalidated_descriptors',physiological_pulse_quality='UNVERIFIED'))
