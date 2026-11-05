"""SCAMPS HDF5 admission and bounded, read-only engineering windows.

SCAMPS: McDuff et al., NeurIPS 2022, https://github.com/danmcduff/scampsdataset
Dataset use is subject to R-UDA v1.0 (non-commercial research); see the retained
license/attribution under docs/waveform_v2/scamps. No V1 or PURE paths are used.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
from itertools import product
from pathlib import Path
import re
from typing import Iterator

import h5py
import numpy as np

SCHEMA = "waveform_v2_scamps_admission_v0.1"
DOC_REVISION = "0d3e1fd9915ea52a66af87caa1508b292555f765"
OFFICIAL_SPLIT_SHA256 = "d0b03f4a22f00c7dcfc106417e252c798274c7eb5fd022de0843b65951fea79e"
NAME = re.compile(r"P[0-9]{6}\.mat\Z")
REQUIRED = ("RawFrames", "Xsub", "skin_mask", "combined_mask", "d_ppg")


class ScampsError(ValueError):
    """Unsupported structure or unsafe/unadmitted requested operation."""


@dataclass(frozen=True)
class ScampsLayout:
    """Explicit profile; defaults are the observed official 600-frame release.

    Small alternate profiles are for unit fixtures, not inferred transpositions.
    Native HDF5 is C,W,H,T; returned RGB is T,H,W,C. Masks are W,H,T -> T,H,W.
    """
    frames: int = 600
    raw_width: int = 320
    crop_width: int = 240
    height: int = 240

    def __post_init__(self):
        if (any(type(v) is not int or v < 2 for v in asdict(self).values())
                or self.crop_width > self.raw_width):
            raise ScampsError("Invalid explicit SCAMPS layout")


def _check_path(path: Path) -> None:
    # Reject private/reserved paths lexically, before stat/open/resolve/enumeration.
    for part in path.parts:
        if part.casefold() == "dataset" or re.match(r"S(?:00[7-9]|01[01])(?:\D|$)", part.upper()):
            raise ScampsError("Private/reserved paths cannot be SCAMPS inputs")
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ScampsError("Symlink SCAMPS paths are refused")


def discover_scamps(root: str | Path) -> list[Path]:
    """Nonrecursive discovery only under the explicit supplied root."""
    base = Path(root)
    _check_path(base)
    if not base.is_dir():
        raise FileNotFoundError("SCAMPS root is not a directory")
    found = []
    for p in sorted(base.iterdir()):
        if p.suffix.lower() != ".mat":
            continue
        if not NAME.fullmatch(p.name):
            raise ScampsError("Unexpected .mat filename in SCAMPS root")
        _check_path(p)
        if not p.is_file():
            raise ScampsError("SCAMPS recording must be a regular .mat file")
        found.append(p)
    if not found:
        raise ScampsError("No SCAMPS PXXXXXX.mat files found directly under root")
    return found


def source_sha256(path: str | Path) -> str:
    p = Path(path)
    _check_path(p)
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.reshape(-1)]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    return value


def _inspect_open(f: h5py.File, layout: ScampsLayout) -> dict:
    datasets, errors, warnings = {}, [], []
    keys = sorted(f.keys())
    for name in keys:
        if not isinstance(f.get(name, getlink=True), h5py.HardLink):
            errors.append(f"{name}:indirect_hdf5_link_refused")
            continue
        ds = f[name]
        if not isinstance(ds, h5py.Dataset):
            errors.append(f"{name}:expected_numeric_dataset")
            continue
        if ds.dtype.kind not in "biuf" or ds.dtype.itemsize > 8 or not ds.shape or 0 in ds.shape:
            errors.append(f"{name}:unsupported_numeric_type_or_empty_shape")
        if ds.external or ds.is_virtual:
            errors.append(f"{name}:external_or_virtual_storage_refused")
        datasets[name] = dict(shape=list(ds.shape), dtype=str(ds.dtype), compression=ds.compression,
                              compression_opts=ds.compression_opts, chunks=list(ds.chunks) if ds.chunks else None,
                              uncompressed_bytes=int(ds.size * ds.dtype.itemsize),
                              attributes={k: _json_value(v) for k, v in ds.attrs.items()})
    missing = [name for name in REQUIRED if name not in datasets]
    errors += [f"{name}:missing_required_variable" for name in missing]
    shapes = {"RawFrames": (3, layout.raw_width, layout.height, layout.frames),
              "Xsub": (3, layout.crop_width, layout.height, layout.frames),
              "d_ppg": (layout.frames, 1)}
    for name, expected in shapes.items():
        if name in datasets and tuple(datasets[name]["shape"]) != expected:
            errors.append(f"{name}:unexpected_shape")
    for name, item in datasets.items():
        if (name.startswith("d_") or re.fullmatch(r"au[0-9]+", name)) and item["shape"] != [layout.frames, 1]:
            errors.append(f"{name}:unexpected_auxiliary_waveform_shape")
    for name in ("skin_mask", "combined_mask", "numerical_mask"):
        if name not in datasets:
            continue
        shape = tuple(datasets[name]["shape"])
        if shape[:2] != (layout.crop_width, layout.height) or len(shape) != 3 or shape[-1] not in (layout.frames, layout.frames - 1):
            errors.append(f"{name}:unexpected_shape")
        elif shape[-1] != layout.frames:
            warnings.append(f"{name}:frame_count_mismatch_mapping_unverified")
    # Explicitly check producer axis annotations instead of trusting dimensions alone.
    for name, item in datasets.items():
        annotated = item["attributes"].get("Python.Shape")
        shape = item["shape"]
        if annotated is not None:
            if name in ("RawFrames", "Xsub") and len(shape) == 4:
                expected = [shape[3], shape[2], shape[1], shape[0]]
            elif name.endswith("mask") and len(shape) == 3:
                expected = list(reversed(shape))
            elif len(shape) == 2 and shape[-1] == 1:
                expected = [shape[0]]
            else:
                expected = list(reversed(shape))
            if annotated != expected:
                errors.append(f"{name}:conflicting_producer_axis_annotation")
    masks_aligned = all(name in datasets and datasets[name]["shape"] ==
                        [layout.crop_width, layout.height, layout.frames]
                        for name in ("skin_mask", "combined_mask"))
    return dict(root_keys=keys, root_attributes={k: _json_value(v) for k, v in f.attrs.items()},
                datasets=datasets, required_present={k: k in datasets for k in REQUIRED},
                missing_variables=missing, errors=errors, warnings=warnings,
                structure_valid=not errors, layout=asdict(layout),
                frame_count=layout.frames if not errors else None,
                waveform_count=datasets.get("d_ppg", {}).get("shape", [None])[0],
                mask_alignment="documented_same_index" if masks_aligned else "UNVERIFIED_LENGTH_MISMATCH",
                embedded_sampling_rate_hz=None, embedded_capture_timestamps=None,
                embedded_dataset_version=None,
                uninterpreted_timing_metadata_keys=[k for k in keys if re.search(r"timestamp|time|fps|rate", k, re.I)],
                allow_training=False, allow_absolute_timing_scoring=False,
                admission_state="ENGINEERING_ONLY" if not errors else "REJECTED_STRUCTURE")


def _tiles(shape, itemsize: int, max_bytes: int, origin=None) -> Iterator[tuple[slice, ...]]:
    """Bound contiguous/chunk subreads in every dimension, not just time."""
    tile = list(shape)
    while np.prod(tile, dtype=np.int64) * itemsize > max_bytes:
        axis = int(np.argmax(tile))
        tile[axis] = max(1, (tile[axis] + 1) // 2)
    origin = origin or (0,) * len(shape)
    for starts in product(*(range(0, n, t) for n, t in zip(shape, tile))):
        yield tuple(slice(o + s, o + min(s + t, n)) for o, s, t, n in zip(origin, starts, tile, shape))


def numeric_audit(ds: h5py.Dataset, *, max_block_bytes: int = 4 * 1024 * 1024) -> dict:
    """Full finite/range scan, bounded reads; no whole-recording allocation."""
    if type(max_block_bytes) is not int or max_block_bytes < max(8, ds.dtype.itemsize):
        raise ScampsError("max_block_bytes must hold at least one numeric value")
    if ds.dtype.kind not in "biuf" or ds.external or ds.is_virtual:
        raise ScampsError("Unsupported numeric audit source")
    if ds.chunks:
        blocks = (tile for chunk in ds.iter_chunks() for tile in _tiles(
            tuple(s.stop - s.start for s in chunk), ds.dtype.itemsize, max_block_bytes,
            tuple(s.start for s in chunk)))
    else:
        blocks = _tiles(ds.shape, ds.dtype.itemsize, max_block_bytes)
    count, invalid, outside, fractional, largest, reads = 0, 0, 0, 0, 0, 0
    minimum, maximum = None, None
    for sl in blocks:
        values = ds[sl]
        largest = max(largest, values.nbytes)
        reads += 1
        count += values.size
        finite = np.isfinite(values)
        invalid += int(values.size - finite.sum())
        usable = values[finite]
        if usable.size:
            low, high = float(usable.min()), float(usable.max())
            minimum = low if minimum is None else min(minimum, low)
            maximum = high if maximum is None else max(maximum, high)
            outside += int(((usable < 0) | (usable > 1)).sum())
            fractional += int((usable != np.trunc(usable)).sum())
    return dict(scope="all_stored_values", count=count, nonfinite_count=invalid,
                all_finite=invalid == 0 and count == ds.size, minimum=minimum, maximum=maximum,
                outside_unit_interval_count=outside, fractional_value_count=fractional,
                largest_read_bytes=largest, read_count=reads)


def inspect_scamps_file(path: str | Path, *, layout: ScampsLayout = ScampsLayout(),
                        scan_values: bool = False, max_block_bytes: int = 4 * 1024 * 1024) -> dict:
    p = Path(path)
    _check_path(p)
    if not NAME.fullmatch(p.name):
        raise ScampsError("Expected a SCAMPS PXXXXXX.mat filename")
    before = p.stat()
    with p.open("rb") as stream:
        header = stream.read(128)
    info = dict(filename=p.name, source_sha256=source_sha256(p), file_bytes=before.st_size,
                schema_version=SCHEMA, documentation_revision=DOC_REVISION,
                matlab_container_header=(header.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
                                         if header.startswith(b"MATLAB 7.3") else None))
    try:
        with h5py.File(p, "r", rdcc_nbytes=1024 * 1024) as f:
            info.update(_inspect_open(f, layout))
            if scan_values and info["structure_valid"]:
                for name, item in info["datasets"].items():
                    item["value_audit"] = numeric_audit(f[name], max_block_bytes=max_block_bytes)
                info["all_values_finite"] = all(v["value_audit"]["all_finite"] for v in info["datasets"].values())
            else:
                info["all_values_finite"] = None
    except OSError as exc:
        info.update(structure_valid=False, errors=[f"invalid_hdf5:{type(exc).__name__}"],
                    admission_state="REJECTED_STRUCTURE", all_values_finite=None,
                    allow_training=False, allow_absolute_timing_scoring=False)
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ScampsError("Source changed during read-only audit")
    return info


class ScampsRecording:
    """Lazy engineering access; this is not a V1 tensor loader or training dataset."""
    def __init__(self, root: str | Path, filename: str, *, layout: ScampsLayout = ScampsLayout(),
                 max_window_frames: int = 8, max_window_bytes: int = 64 * 1024 * 1024,
                 expected_sha256: str | None = None):
        if not isinstance(filename, str) or not NAME.fullmatch(filename):
            raise ScampsError("Filename must be a direct PXXXXXX.mat child")
        self.path = Path(root) / filename
        _check_path(self.path)
        self.info = inspect_scamps_file(self.path, layout=layout)
        if not self.info["structure_valid"]:
            raise ScampsError("Malformed SCAMPS structure: " + ",".join(self.info["errors"]))
        if expected_sha256 is not None and self.info["source_sha256"] != expected_sha256:
            raise ScampsError("SCAMPS source checksum mismatch")
        if type(max_window_frames) is not int or max_window_frames < 1 or type(max_window_bytes) is not int or max_window_bytes < 1:
            raise ScampsError("Window limits must be positive integers")
        self.layout, self.max_window_frames, self.max_window_bytes = layout, max_window_frames, max_window_bytes
        self._identity = (self.path.stat().st_size, self.path.stat().st_mtime_ns)

    def _bounds(self, start: int, stop: int, count: int, bytes_per_frame: int):
        if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= count
                or stop - start > self.max_window_frames):
            raise ScampsError("Invalid/oversized half-open window")
        if (stop - start) * bytes_per_frame > self.max_window_bytes:
            raise ScampsError("Window exceeds configured decoded-byte limit")
        current = self.path.stat()
        if (current.st_size, current.st_mtime_ns) != self._identity:
            raise ScampsError("Source changed since checksum/structure verification")

    def read_window(self, start: int, stop: int, *, frames: str = "Xsub",
                    require_segmentation: bool = False) -> dict:
        if frames not in ("Xsub", "RawFrames"):
            raise ScampsError("Choose Xsub or RawFrames explicitly")
        aligned = self.info["mask_alignment"] == "documented_same_index"
        if require_segmentation and (not aligned or frames != "Xsub"):
            raise ScampsError("Segmentation cannot be mapped to these frames without verified provenance")
        frame_info = self.info["datasets"][frames]
        bytes_per_frame = int(np.prod(frame_info["shape"][:-1])) * np.dtype(frame_info["dtype"]).itemsize
        mask_bytes = sum(int(np.prod(self.info["datasets"][k]["shape"][:-1])) *
                         np.dtype(self.info["datasets"][k]["dtype"]).itemsize for k in ("skin_mask", "combined_mask"))
        self._bounds(start, stop, self.layout.frames, bytes_per_frame + (mask_bytes if aligned and frames == "Xsub" else 0))
        masks = None
        with h5py.File(self.path, "r", rdcc_nbytes=1024 * 1024) as f:
            rgb = f[frames][:, :, :, start:stop].transpose(3, 2, 1, 0).copy()
            target = f["d_ppg"][start:stop, 0].copy()
            if aligned and frames == "Xsub":
                masks = {k: f[k][:, :, start:stop].transpose(2, 1, 0).copy()
                         for k in ("skin_mask", "combined_mask")}
        finite_pixels = np.isfinite(rgb).all(-1)
        unit_pixels = finite_pixels & ((rgb >= 0) & (rgb <= 1)).all(-1)
        frame_valid = unit_pixels.all(axis=(1, 2))
        label_valid = np.isfinite(target)
        segmentation_valid = np.zeros(stop - start, dtype=bool)
        if masks:
            segmentation_valid = np.logical_and.reduce([np.isfinite(v).all(axis=(1, 2)) for v in masks.values()])
        indices = np.arange(start, stop, dtype=np.int64)
        return dict(rgb=rgb, d_ppg=target, frame_indices=indices, label_indices=indices.copy(),
                    masks=masks, mask_alignment=self.info["mask_alignment"],
                    validity=dict(frame=frame_valid, label=label_valid, paired=frame_valid & label_valid,
                                  segmentation_finite=segmentation_valid, pixel_finite=finite_pixels),
                    quality=dict(finite_pixel_fraction=finite_pixels.mean(axis=(1, 2)),
                                 unit_range_pixel_fraction=unit_pixels.mean(axis=(1, 2))),
                    provenance=dict(filename=self.path.name, source_sha256=self.info["source_sha256"],
                                    schema_version=SCHEMA, documentation_revision=DOC_REVISION,
                                    frames_variable=frames, channel_order="RGB",
                                    native_axes="C,W,H,T", decoded_axes="T,H,W,C",
                                    label_alignment="documented_same_synthetic_frame_index",
                                    capture_timestamps=None, physical_synchronization_verified=False,
                                    mask_values="stored_unthresholded_not_probability_or_binary_guarantee",
                                    allow_training=False))

    def read_mask_window(self, start: int, stop: int) -> dict:
        """Independent MASK indices for diagnostics; never invent frame associations."""
        fields = [self.info["datasets"][k] for k in ("skin_mask", "combined_mask")]
        if fields[0]["shape"][-1] != fields[1]["shape"][-1]:
            raise ScampsError("Skin/combined mask counts differ")
        per_frame = sum(int(np.prod(v["shape"][:-1])) * np.dtype(v["dtype"]).itemsize for v in fields)
        self._bounds(start, stop, fields[0]["shape"][-1], per_frame)
        with h5py.File(self.path, "r", rdcc_nbytes=1024 * 1024) as f:
            masks = {k: f[k][:, :, start:stop].transpose(2, 1, 0).copy()
                     for k in ("skin_mask", "combined_mask")}
        return dict(masks=masks, mask_indices=np.arange(start, stop, dtype=np.int64),
                    frame_indices=None if self.info["mask_alignment"] != "documented_same_index"
                    else np.arange(start, stop, dtype=np.int64),
                    mask_alignment=self.info["mask_alignment"], source_sha256=self.info["source_sha256"],
                    finite={k: np.isfinite(v) for k, v in masks.items()})


def read_official_split(path: str | Path, *, expected_sha256: str = OFFICIAL_SPLIT_SHA256) -> dict:
    """Read the pinned public CSV, including its verified malformed two-field header.

    The downloaded artifact has header nk,Split but three data fields: ID, URL,
    split. This explicit profile preserves the anomaly rather than shifting fields.
    URLs are provenance strings only and are never fetched by this parser.
    """
    p = Path(path)
    actual = source_sha256(p)
    if actual != expected_sha256:
        raise ScampsError("Official split snapshot hash mismatch")
    with p.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or rows[0] != ["nk", "Split"]:
        raise ScampsError("Unexpected official split CSV header")
    assignments = {}
    for row in rows[1:]:
        if len(row) != 3:
            raise ScampsError("Expected explicit recording, URL, split fields")
        recording, url, split = row
        if not re.fullmatch(r"P[0-9]{6}", recording) or split not in ("Train", "Val", "Test"):
            raise ScampsError("Invalid recording or official split value")
        if not url.startswith("https://") or url.rsplit("/", 1)[-1] != recording + ".mat":
            raise ScampsError("Official split recording/URL mismatch")
        if recording in assignments:
            raise ScampsError("Duplicate official split recording")
        assignments[recording] = dict(split=split, source_url=url)
    return dict(source_sha256=actual, header=rows[0], header_anomaly="2_header_fields_3_data_fields",
                assignments=assignments, identity_groups_available=False,
                avatar_disjointness="claimed_by_datasheet_not_locally_verifiable")
