"""Verified Gate 06 Baseline A cache reuse and isolated reference-window reads."""
from pathlib import Path
import json

import h5py
import numpy as np

from .waveform_v2_scamps import _check_path, source_sha256
from .waveform_v2_classical_study import STARTS, array_sha256, predict_recording


def load_verified_baseline(root, filename, expected_cache_hash, expected_input_hashes):
    root = Path(root)
    _check_path(root)
    path = root / "retained_arrays" / (Path(filename).stem + ".npz")
    _check_path(path)
    if source_sha256(path) != expected_cache_hash:
        raise ValueError("Baseline cache checksum mismatch")
    with np.load(path, allow_pickle=False) as z:
        # Candidate B and reference arrays are deliberately not read.
        rgb = z["baseline_rgb"].copy()
        valid = z["baseline_acquisition_valid"].copy()
        indices = z["frame_indices"].copy()
        clipping = z["roi_metric_clipped_fraction"].copy()
    if (indices.dtype.kind not in "iu" or not np.array_equal(indices, np.arange(600))
            or clipping.shape != (3, 600)):
        raise ValueError("Baseline indices/optical shape changed")
    prediction = predict_recording(rgb, valid)
    hashes = [array_sha256(x) for x in prediction["batch"]["x"]]
    if hashes != expected_input_hashes:
        raise ValueError("Rebuilt inputs differ from Gate 06")
    return dict(rgb=rgb, valid=valid, clipping=clipping, prediction=prediction,
                cache_sha256=expected_cache_hash, input_sha256=hashes)


def cache_manifest(path):
    path = Path(path)
    _check_path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["relative_path"]: r["sha256"] for r in data["files"]}


def read_reference_windows(recording):
    """Only same-index d_ppg values, after the caller locks blind predictions."""
    windows = []
    for start in STARTS:
        recording._bounds(start, start+300, recording.layout.frames, 8)
        with h5py.File(recording.path, "r", rdcc_nbytes=1024**2) as f:
            reference = f["d_ppg"][start:start+300, 0].copy()
        if reference.shape != (300,):
            raise ValueError("Reference index support changed")
        windows.append(reference)
    return np.stack(windows)
