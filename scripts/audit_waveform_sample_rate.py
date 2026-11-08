"""Synthetic transfer of the EXISTING 60-to-30 adapter and GREEN filter.

No participant video, checkpoint, learning or physiological efficacy claim.
Single-frequency amplitudes are measured before per-window standardization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rppg_lab.classical import green
from rppg_lab.config import PipelineConfig
from rppg_lab.extraction_cache import FaceROIExtraction
from rppg_lab.frozen_timebase import adapt_native_face
from rppg_lab.processing import resample_trace, shared_grid
from rppg_lab.types import RGBTrace, SharedTimebase


def fitted_gain(t, x, frequency):
    good=np.isfinite(x) & (t>=5) & (t<25)
    matrix=np.column_stack((np.ones(good.sum()),np.sin(2*np.pi*frequency*t[good]),
                            np.cos(2*np.pi*frequency*t[good])))
    coef=np.linalg.lstsq(matrix,x[good],rcond=None)[0]
    return float(np.hypot(coef[1],coef[2]))


def run():
    names=('forehead','left_cheek','right_cheek')
    config=PipelineConfig(regions=('face',),sample_rate=30,face_roi='forehead',face_rois=names)
    t=np.arange(1800,dtype=float)/60
    rows=[]
    for frequency in (0.7,1.2,3.5,8.0,12.0,18.0,24.0):
        wave=np.sin(2*np.pi*frequency*t)
        rgb=100+wave[:,None]*np.array([0.5,1.0,0.2])
        traces={name:RGBTrace(t.copy(),rgb.copy(),name,np.ones(len(t),bool)) for name in names}
        extraction=FaceROIExtraction(SharedTimebase(shared_grid(t,30),t.copy()),traces)
        adapted,_=adapt_native_face(extraction,config)
        low_t=adapted.shared.timestamps
        low_rgb=resample_trace(adapted.traces['forehead'],low_t,config.max_gap_sec)[0]
        alias=frequency if frequency<15 else 30-frequency
        row=dict(input_frequency_hz=frequency,input_native_rate_hz=60,output_adapter_rate_hz=30,
            fitted_output_frequency_hz=alias,native60_raw_rgb_gain=fitted_gain(t,rgb[:,1],frequency),
            adapted30_raw_rgb_gain=fitted_gain(low_t,low_rgb[:,1],alias),
            native60_default_green_gain=fitted_gain(t,green(rgb,60),frequency))
        if frequency<=3.5 and abs(row['adapted30_raw_rgb_gain']-1)>.02:
            raise AssertionError('Existing adapter unexpectedly loses its clean low-band input')
        if frequency>=18 and row['adapted30_raw_rgb_gain']>.001:
            raise AssertionError('Existing adapter unexpectedly aliases its declared stopband')
        rows.append(row)
    return dict(evidence_type='SYNTHETIC_TRANSFER_ONLY', duration_s=30,retained_interval_s=[5,25],
        native_samples=1800, model_weights_loaded=False, training=False,
        scope='No neural performance or real waveform bandwidth measurement; no standardization applied',rows=rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parents[1]
    out=args.out.absolute()
    allowed=root/'outputs/waveform_sample_rate'
    if allowed.resolve() not in out.resolve().parents or out.exists():
        parser.error('Use a NEW directory inside outputs/waveform_sample_rate/')
    report=run()
    paths=('scripts/audit_waveform_sample_rate.py','rppg_lab/frozen_timebase.py','rppg_lab/classical.py')
    report['source_sha256']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    out.mkdir(parents=True,exist_ok=False)
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
