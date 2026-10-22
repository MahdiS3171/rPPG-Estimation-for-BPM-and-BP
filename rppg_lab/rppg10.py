"""Read-only rPPG-10 discovery, precropped RGB adapter and ECG-only reference.

No landmarks, optical reference targets, learned parameters or ECG-dependent
model inputs live here. See docs/WAVEFORM_V1_RPPG10_EXTERNAL_VALIDATION.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re

import cv2
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from .artifacts import file_sha256
from .extraction_cache import DEFAULT_WAVEFORM_ROIS, FaceROIExtraction, _validate
from .processing import resample_trace
from .types import RGBTrace, SharedTimebase

EXPECTED_IDS = tuple(f"Subject_{i}" for i in range(1, 28) if i != 4)
ROI_MAP = {"forehead": "Forehead", "left_cheek": "Cheek2", "right_cheek": "Cheek1"}
DATA_DOI = "10.17632/bx8982xgwt.1"
PAPER = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12311941/"
ECG_FS = 1000.0  # Paper equipment section; never inferred from video duration.
SYNC_BOUND_SEC = 0.5  # Paper limitations, manual operator reaction delay.


def normalized_subject_id(value) -> str:
    match = re.fullmatch(r"Subject[ _](\d+)", str(value).strip(), re.IGNORECASE)
    if not match or not 1 <= int(match[1]) <= 27:
        raise ValueError(f"Expected an exact Subject_X ID, got {value!r}")
    return f"Subject_{int(match[1])}"


def validate_roi_map(mapping=None) -> dict:
    mapping = dict(ROI_MAP if mapping is None else mapping)
    if tuple(mapping) != DEFAULT_WAVEFORM_ROIS or set(mapping.values()) != set(ROI_MAP.values()):
        raise ValueError("roi-map must map forehead,left_cheek,right_cheek in frozen order to Forehead,Cheek1,Cheek2 once each")
    return mapping


def scan_dataset(root, roi_map=None) -> dict:
    """Inspect exact paths and content hashes; unrelated/duplicate files are QC.

    Expected root/Subject_X/Subject_X_{Forehead,Cheek1,Cheek2}[_].avi and
    Subject_X_ECG.npy. No positional or cross-subject fallback is permitted.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"rPPG-10 root is unavailable: {root}")
    mapping = validate_roi_map(roi_map)
    folders, unrelated = {}, []
    for path in sorted(root.iterdir()):
        if path.is_dir() and re.fullmatch(r"Subject_\d+", path.name):
            sid = normalized_subject_id(path.name)
            if sid in folders:
                raise ValueError(f"Duplicate normalized subject folder: {sid}")
            folders[sid] = path
        elif path.suffix.lower() != ".xlsx":
            unrelated.append(path.name)
    records = []
    for sid in EXPECTED_IDS:
        folder = folders.get(sid)
        files, qc, recognized = {}, [], set()
        for key, suffix in {**mapping, "ecg": "ECG"}.items():
            pattern = (rf"{re.escape(sid)}_{suffix}\.npy" if key == "ecg" else
                       rf"{re.escape(sid)}_{suffix}_?\.avi")
            matches = [] if folder is None else [p for p in folder.iterdir() if p.is_file() and re.fullmatch(pattern, p.name)]
            recognized.update(p.name for p in matches)
            if len(matches) != 1:
                qc.append(f"{key}: {'missing' if not matches else 'ambiguous_duplicate'}")
                continue
            path = matches[0]
            files[key] = dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                              sha256=file_sha256(path))
            if not path.stat().st_size:
                qc.append(f"{key}: zero_sized_file")
        if folder:
            qc.extend(f"unrelated_file: {p.name}" for p in sorted(folder.iterdir()) if p.name not in recognized)
        records.append(dict(subject_id=sid, present=folder is not None, files=files,
                            structure_usable=len(files) == 4 and all(f["bytes"] for f in files.values())
                            and not any("ambiguous" in q for q in qc), qc=qc))
    excluded = folders.get("Subject_4")
    exclusion = dict(subject_id="Subject_4", present=excluded is not None,
                     reason="Official exclusion: significant video/ECG artifacts; never evaluated",
                     files=[] if excluded is None else [dict(name=p.name, bytes=p.stat().st_size)
                                                         for p in sorted(excluded.iterdir()) if p.is_file()])
    identity = [{"subject_id": r["subject_id"], "files": r["files"]} for r in records]
    return dict(dataset_doi=DATA_DOI, root=str(root), expected_usable_ids=list(EXPECTED_IDS),
                discovered_usable_ids=[r["subject_id"] for r in records if r["present"]],
                missing_ids=[r["subject_id"] for r in records if not r["present"]],
                subjects=records, excluded=[exclusion], unrelated_root_entries=unrelated, roi_map=mapping,
                data_identity_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest())


@dataclass
class DecodedCrop:
    trace: RGBTrace
    qc: dict


def decode_crop(path, roi_name, cache_dir=None) -> DecodedCrop:
    """One full decode, RGB means per frame. AVI clocks remain nominal.

    OpenCV POS_MSEC may be synthesized by the backend. Record its diagnostics
    but do not promote it to a verified presentation/acquisition timestamp.
    Failed tails are never padded. Cached data are RGB/validity only.
    """
    path = Path(path)
    identity = dict(sha256=file_sha256(path), version="rppg10_rgb_v1", opencv=cv2.__version__, roi=roi_name)
    cached = None
    if cache_dir:
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        cached = directory / (hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() + ".npz")
        if cached.exists():
            with np.load(cached, allow_pickle=False) as a:
                qc = json.loads(str(a["qc"].item()))
                return DecodedCrop(RGBTrace(a["time"], a["rgb"], roi_name, a["valid"],
                                           {"brightness": a["brightness"]}), qc)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Malformed/unreadable crop: {path}")
        fps, reported = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or not 29 <= fps <= 31:
            raise ValueError(f"Nonconforming nominal FPS {fps}: {path}; frozen adapter expects approximately 30 fps")
        values, valid, backend, sizes = [], [], [], set()
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            good = bgr is not None and bgr.ndim == 3 and bgr.shape[2] == 3 and bgr.size > 0
            if good:
                sizes.add(tuple(bgr.shape[:2]))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).mean(axis=(0, 1))
                good = bool(np.isfinite(rgb).all())
            else:
                rgb = np.full(3, np.nan)
            values.append(rgb)
            valid.append(good)
            backend.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
    finally:
        cap.release()
    if len(values) < 2:
        raise ValueError(f"Too few decoded frames: {path}")
    time = np.arange(len(values), dtype=float) / fps
    rgb, mask = np.asarray(values), np.asarray(valid, bool)
    brightness = rgb.mean(axis=1)
    backend = np.asarray(backend)
    qc = dict(identity=identity, fps_nominal=fps, frames_reported=reported, frames_decoded=len(time),
              decoded_shortfall=max(0, reported - len(time)), sizes=sorted(sizes),
              nominal_duration_sec=len(time) / fps, timestamp_source="nominal_fps",
              backend_start_sec=float(backend[0]), backend_end_sec=float(backend[-1]),
              backend_monotonic=bool(np.isfinite(backend).all() and (np.diff(backend) > 0).all()),
              valid_fraction=float(mask.mean()), brightness_mean=float(np.nanmean(brightness)),
              brightness_std=float(np.nanstd(brightness)),
              warnings=["nominal_clock; acquisition PTS/common start unverified"])
    if reported != len(time):
        qc["warnings"].append("reported_decoded_frame_count_mismatch; missing tail retained as unsupported")
    if len(time) / fps < 599:
        qc["warnings"].append("short_recording")
    if sizes != {(64, 64)}:
        qc["warnings"].append("nonstandard_crop_dimensions")
    if len(sizes) != 1:
        raise ValueError(f"Crop dimensions change during clip: {path}")
    if cached:
        # Cache contains no targets, priors, neural outputs or spreadsheet data.
        with cached.open("xb") as f:
            np.savez_compressed(f, time=time, rgb=rgb, valid=mask, brightness=brightness, qc=json.dumps(qc))
    return DecodedCrop(RGBTrace(time, rgb, roi_name, mask, {"brightness": brightness}), qc)


def load_subject_crops(root, record, config, cache_dir=None) -> tuple[FaceROIExtraction, dict]:
    """Bound all three crops to their common support, then reuse frozen API.

    Each stream has its own nominal source clock. The canonical 30-Hz clock
    contains only supported sample times; resample_trace supplies bounded gap
    handling. Source validity is conservatively projected onto that clock.
    """
    if config.sample_rate != 30 or config.requested_face_rois != DEFAULT_WAVEFORM_ROIS:
        raise ValueError("Frozen adapter requires 30 Hz and forehead,left_cheek,right_cheek")
    decoded = {name: decode_crop(Path(root) / record["files"][name]["path"], name, cache_dir)
               for name in DEFAULT_WAVEFORM_ROIS}
    fps = [d.qc["fps_nominal"] for d in decoded.values()]
    if max(fps) - min(fps) > .01:
        raise ValueError(f"Divergent crop FPS {fps}; common start/clock cannot be verified")
    end = min(d.trace.timestamps[-1] for d in decoded.values())
    grid = np.arange(int(np.floor(end * 30 + 1e-7)) + 1) / 30
    traces = {}
    for name, crop in decoded.items():
        rgb, _ = resample_trace(crop.trace, grid, config.max_gap_sec)
        # Quality is source acquisition support, not ECG agreement or calibration.
        t = crop.trace.timestamps
        right = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
        left = np.maximum(right - 1, 0)
        exact = np.isclose(t[right], grid, atol=1e-8, rtol=0)
        valid = np.where(exact, crop.trace.validity_mask[right],
                         crop.trace.validity_mask[left] & crop.trace.validity_mask[right])
        valid &= np.isfinite(rgb).all(axis=1)
        traces[name] = RGBTrace(grid, rgb, name, valid, {"brightness": rgb.mean(axis=1)})
    counts = [d.qc["frames_decoded"] for d in decoded.values()]
    qc = dict(crops={n: d.qc for n, d in decoded.items()}, common_samples=len(grid),
              common_duration_sec=len(grid) / 30, frame_counts_match=len(set(counts)) == 1,
              discarded_tail_sec={n: d.qc["nominal_duration_sec"] - len(grid) / 30 for n, d in decoded.items()},
              clock="nominal 30-Hz grid bounded by all three clips; no extrapolation",
              common_start="assumed from official shared-source crops; no acquisition timestamps to independently verify",
              synchronization="official manual ECG trim; up to 0.5 s reaction delay; no shifts estimated")
    return _validate(FaceROIExtraction(SharedTimebase(grid, grid), traces), config), qc


def load_ecg(path, fs=ECG_FS) -> tuple[np.ndarray, dict]:
    if not np.isfinite(fs) or not 100 <= fs <= 5000:
        raise ValueError("ecg-fs must be documented, finite and between 100 and 5000 Hz")
    try:
        raw = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"Unreadable ECG array {path}: {exc}") from exc
    x = np.squeeze(raw)
    if x.ndim != 1 or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError(f"ECG must be a single real channel (N,) or singleton equivalent, got {raw.shape}")
    x = x.astype(float)
    if len(x) < fs * 10:
        raise ValueError("ECG shorter than one 10-second reference window")
    return x, dict(shape=list(raw.shape), dtype=str(raw.dtype), fs_hz=fs, duration_sec=len(x) / fs,
                   finite_fraction=float(np.isfinite(x).mean()),
                   units="stored amplitudes; ADC-like counts in inspected release; no mV conversion assumed",
                   clock="uniform samples from manually trimmed ECG; start=0 nominal video time")


@dataclass
class ECGPeaks:
    times: np.ndarray
    amplitude_ratio: np.ndarray
    filtered: np.ndarray
    raw: np.ndarray
    fs: float
    qc: dict


def detect_r_peaks(ecg, fs=ECG_FS) -> ECGPeaks:
    """Simple ECG-only QRS detector; not a clinical rhythm classifier.

    Filter finite runs (5-20 Hz, third-order SOS), abs-QRS peaks separated by
    300 ms, height >= 5 times local median absolute filtered amplitude.
    Local noise floor is measured in 8-second blocks. No optical data enter.
    """
    if not np.isfinite(fs) or not 100 <= fs <= 5000:
        raise ValueError("Invalid documented ECG sampling rate")
    x = np.asarray(ecg, float)
    if x.ndim != 1:
        raise ValueError("ECG detector requires one channel")
    filtered = np.full(x.shape, np.nan)
    from .processing import valid_runs
    candidates, ratios = [], []
    sos = butter(3, [5, 20], fs=fs, btype="bandpass", output="sos")
    for a, b in valid_runs(np.isfinite(x)):
        if b - a < fs * 3 or np.std(x[a:b]) < 1e-10:
            continue
        y = sosfiltfilt(sos, x[a:b] - np.median(x[a:b]))
        filtered[a:b] = y
        absolute = np.abs(y)
        block = int(8 * fs)
        floor = np.concatenate([np.full(min(block, len(y) - j), max(np.median(absolute[j:j+block]), 1e-10))
                                for j in range(0, len(y), block)])
        p, _ = find_peaks(absolute, distance=int(.30 * fs), height=5 * floor, prominence=3 * float(np.median(absolute)))
        # Filtering at a finite-run boundary cannot provide reliable beat timing.
        p = p[(p >= .25 * fs) & (p < len(y) - .25 * fs)]
        candidates.extend((p + a).tolist())
        ratios.extend((absolute[p] / floor[p]).tolist())
    times = np.asarray(candidates, float) / fs
    rr = np.diff(times)
    return ECGPeaks(times, np.asarray(ratios), filtered, x, fs,
                    dict(detector="abs_QRS_5_20Hz_v1", refractory_sec=.30, local_floor_block_sec=8,
                         height_to_local_median=5, peaks=len(times),
                         median_rr_sec=float(np.median(rr)) if len(rr) else None,
                         plausible_rr_fraction=float(((rr >= .30) & (rr <= 1.5)).mean()) if len(rr) else 0,
                         finite_fraction=float(np.isfinite(x).mean()),
                         limitations="simple detector; false/missed beats possible; irregular and noisy windows rejected; not medical adjudication"))


def _ecg_window(peaks: ECGPeaks, start, end) -> dict:
    result = dict(hr_bpm=float("nan"), valid=False, reason="", rr_count=0)
    if start < 0 or end > len(peaks.raw) / peaks.fs + 1e-8:
        return {**result, "reason": "missing_ecg_support"}
    x = peaks.raw[int(round(start * peaks.fs)):int(round(end * peaks.fs))]
    if not len(x) or not np.isfinite(x).all():
        return {**result, "reason": "nonfinite_ecg"}
    if np.std(x) < 1e-10 or np.mean(np.diff(x) == 0) > .95:
        return {**result, "reason": "flat_ecg"}
    if np.mean((x == 0) | (x == 1023)) > .05:
        return {**result, "reason": "adc_clipping"}
    keep = (peaks.times >= start) & (peaks.times < end)
    t, strength = peaks.times[keep], peaks.amplitude_ratio[keep]
    rr = np.diff(t)  # Both endpoints inside the exact window; no beat-count edge bias.
    result["rr_count"] = len(rr)
    if len(rr) < 5:
        return {**result, "reason": "fewer_than_5_complete_rr"}
    median = float(np.median(rr))
    cv = float(np.std(rr) / np.mean(rr))
    result.update(rr_cv=cv, median_qrs_ratio=float(np.median(strength)))
    if ((rr < .30) | (rr > 1.5)).any() or cv > .20 or np.max(abs(rr / median - 1)) > .30:
        return {**result, "reason": "irregular_or_missed_beats"}
    if t[0] - start > 1.5 * median or end - t[-1] > 1.5 * median:
        return {**result, "reason": "uncovered_ecg_edges"}
    y = peaks.filtered[int(round(start * peaks.fs)):int(round(end * peaks.fs))]
    near_qrs = np.zeros(len(y), bool)
    for beat in t:
        idx = int(round((beat - start) * peaks.fs))
        near_qrs[max(0, idx-int(.08*peaks.fs)):min(len(y), idx+int(.08*peaks.fs)+1)] = True
    concentration = float(np.nansum(y[near_qrs] ** 2) / max(np.nansum(y ** 2), 1e-20))
    result["qrs_energy_fraction"] = concentration
    if not np.isfinite(y).all() or concentration < .5:
        return {**result, "reason": "low_qrs_energy_concentration"}
    return {**result, "hr_bpm": 60 / median, "valid": True, "reason": "valid"}


def ecg_reference(peaks, start, end, sync_bound_sec=SYNC_BOUND_SEC) -> dict:
    """Median complete-RR HR; invalidate if ±manual-sync bound changes HR >3 bpm.

    The nominal label is never selected/shifted against an optical prediction.
    Every label carries timing uncertainty, even when the sensitivity is small.
    """
    if not np.isfinite(sync_bound_sec) or sync_bound_sec < 0:
        raise ValueError("Synchronization uncertainty must be finite and nonnegative")
    nominal = _ecg_window(peaks, start, end)
    offsets = [-sync_bound_sec, 0., sync_bound_sec] if sync_bound_sec else [0.]
    checks = [_ecg_window(peaks, start + shift, end + shift) for shift in offsets]
    spread = max(c["hr_bpm"] for c in checks) - min(c["hr_bpm"] for c in checks) if all(c["valid"] for c in checks) else float("nan")
    timing_ok = all(c["valid"] for c in checks) and spread <= 3
    return dict(nominal, hr_bpm=nominal["hr_bpm"] if nominal["valid"] and timing_ok else float("nan"),
                valid=nominal["valid"] and timing_ok,
                reason=nominal["reason"] if not nominal["valid"] else ("valid" if timing_ok else "timing_uncertain_or_boundary_support"),
                sync_bound_sec=sync_bound_sec, sync_hr_range_bpm=spread,
                timing="manual nominal alignment; ±bound sensitivity only; no exact electrical/optical phase claim")


def load_metadata(path) -> tuple[dict, dict]:
    """Join column- or row-oriented XLSX by exact normalized headers/IDs only.

    Retain only environmental/skin descriptors; no dates, times or personal
    demographic/health fields are copied to artifacts.
    """
    import pandas as pd
    raw = pd.read_excel(path, header=None)
    if str(raw.iloc[0, 0]).strip().lower() == "characteristics":
        ids = [normalized_subject_id(v) for v in raw.iloc[0, 1:]]
        fields = [str(v).strip() for v in raw.iloc[1:, 0]]
        values = raw.iloc[1:, 1:].to_numpy().T
    else:
        fields = [str(v).strip() for v in raw.iloc[0, 1:]]
        ids = [normalized_subject_id(v) for v in raw.iloc[1:, 0]]
        values = raw.iloc[1:, 1:].to_numpy()
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate metadata subject IDs; positional joining is forbidden")
    allowed = {"Skin Tone (FitzPatrick)": "skin_type", "Illumination (Lux)": "illumination",
               "Temperature (Celsius)": "temperature_c", "Humidity (%)": "humidity_percent"}
    joined = {sid: {allowed[field]: str(value).strip() for field, value in zip(fields, row)
                    if field in allowed and not pd.isna(value)} for sid, row in zip(ids, values)}
    return joined, dict(sha256=file_sha256(path), joined_ids=sorted(joined),
                         missing_ids=sorted(set(EXPECTED_IDS) - joined.keys()),
                         retained_fields=list(allowed.values()), join="exact normalized subject ID; no positional joining")
