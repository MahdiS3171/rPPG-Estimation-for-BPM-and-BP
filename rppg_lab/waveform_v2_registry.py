"""Read-only, fail-closed Waveform v2 dataset inventory and split contracts.

PURE inventory reads only PNG *names* and the companion pulse JSON by default.
It neither decodes images nor makes a waveform/HR/timing validity claim. Existing
Waveform v1 dataset loaders remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import re

import numpy as np

from .splits import assert_subject_disjoint, split_subjects

SCHEMA = "waveform_v2_dataset_manifest_v0.1"
SPLIT_SCHEMA = "waveform_v2_subject_split_v0.1"
PURE_RECORDING = re.compile(r"^(\d{2})-(\d{2})$")
RESERVED_OWN_SUBJECTS = frozenset(f"S{i:03d}" for i in range(7, 12))
REFERENCE_TIME_UNITS = {"unknown", "seconds", "milliseconds", "nanoseconds"}


@dataclass(frozen=True)
class PureSequence:
    participant_id: str
    recording_id: str
    frames_dir: Path
    reference_file: Path


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_sequence_sha256(frames: Sequence[Path], root: Path) -> str:
    """Opt-in full-byte hash; stable over file metadata changes and ordering."""
    digest = sha256()
    for frame in frames:
        relative_name = frame.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative_name).to_bytes(4, "big"))
        digest.update(relative_name)
        with frame.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def discover_pure(root: str | Path) -> list[PureSequence]:
    """Recognize only PURE's explicit sequence folders; never scan a generic Dataset tree."""
    base = Path(root)
    if any(part.casefold() == "dataset" or part.upper() in RESERVED_OWN_SUBJECTS
           for part in base.parts):
        raise ValueError("Private/reserved data paths are not PURE inputs")
    if base.is_symlink():
        raise ValueError("Symlink PURE root refused")
    if not base.is_dir():
        raise FileNotFoundError(f"PURE root not found: {base}")
    sequences: list[PureSequence] = []
    for directory in sorted(base.iterdir()):
        if not directory.is_dir() or directory.is_symlink():
            continue
        match = PURE_RECORDING.fullmatch(directory.name)
        if not match:
            continue
        image_dir = directory / directory.name
        gt = directory / f"{directory.name}.json"
        if image_dir.is_symlink() or gt.is_symlink():
            raise ValueError(f"Symlink input refused: {directory.name}")
        sequences.append(PureSequence(
            participant_id=f"PURE:{match.group(1)}",
            recording_id=f"PURE:{directory.name}",
            frames_dir=image_dir,
            reference_file=gt,
        ))
    if not sequences:
        raise ValueError("No PURE recording folders (NN-NN) found; this is not a PURE root")
    return sequences


def _reference_audit(path: Path, timestamp_unit: str) -> dict[str, Any]:
    if not path.is_file():
        return dict(reference_record_count=0, reference_waveform_count=0,
                    reference_clock_status="missing_reference", reference_timestamp_unit=timestamp_unit,
                    reference_clock_source="unknown", reference_sha256=None)
    digest = sha256_file(path)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        entries = parsed["/FullPackage"]
        if not isinstance(entries, list) or len(entries) < 2:
            raise ValueError("/FullPackage must be a nontrivial array")
        if not all(isinstance(row, dict) for row in entries):
            raise ValueError("/FullPackage entries must be objects")
        valid_waveforms = sum(isinstance(row, dict) and isinstance(row.get("Value"), dict)
                              and isinstance(row["Value"].get("waveform"), (int, float))
                              and np.isfinite(row["Value"]["waveform"])
                              for row in entries)
        timestamps = [row.get("Timestamp") for row in entries]
        if any(t is None for t in timestamps):
            clock = "missing_timestamp"
        elif any(isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t)
                 for t in timestamps):
            clock = "invalid_timestamp"
        else:
            diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
            clock = ("nonmonotonic" if np.any(diffs < 0) else
                     "duplicates" if np.any(diffs == 0) else
                     "strictly_increasing")
        return dict(reference_record_count=len(entries), reference_waveform_count=int(valid_waveforms),
                    reference_clock_status=clock, reference_timestamp_unit=timestamp_unit,
                    reference_clock_source=("PURE:/FullPackage[*].Timestamp"
                                            if clock not in ("missing_timestamp", "invalid_timestamp")
                                            else "unknown"), reference_sha256=digest)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return dict(reference_record_count=0, reference_waveform_count=0,
                    reference_clock_status=f"invalid_reference_json:{type(exc).__name__}",
                    reference_timestamp_unit=timestamp_unit, reference_clock_source="unknown",
                    reference_sha256=digest)


def inventory_pure(root: str | Path, *, reference_timestamp_unit: str = "unknown",
                   hash_images: bool = False, access_status: str = "unverified",
                   license_scope: str = "unknown", license_evidence: str | None = None,
                   sync_evidence: str = "parallel_recorded_not_independently_verified") -> dict[str, Any]:
    """Metadata-only inventory; eligibility is always false pending independent checks.

    Video exposure timestamps are unknown for PNGs. We deliberately do not map
    array index / nominal FPS to an independently verified camera clock.
    """
    if reference_timestamp_unit not in REFERENCE_TIME_UNITS:
        raise ValueError(f"Unsupported reference timestamp unit: {reference_timestamp_unit}")
    if access_status not in ("unverified", "verified"):
        raise ValueError("access_status must be 'unverified' or 'verified'")
    if not isinstance(license_scope, str) or not license_scope.strip():
        raise ValueError("license_scope must be nonempty")
    if access_status == "verified" and not license_evidence:
        raise ValueError("Verified access requires explicit evidence (URL or institutional record)")
    root = Path(root)
    records: list[dict[str, Any]] = []
    for sequence in discover_pure(root):
        frames = (sorted(f for f in sequence.frames_dir.glob("*.png")
                         if f.is_file() and not f.is_symlink())
                  if sequence.frames_dir.is_dir() else [])
        ref = _reference_audit(sequence.reference_file, reference_timestamp_unit)
        issues = []
        if not frames:
            issues.append("no_png_frames")
        if ref["reference_record_count"] < 2:
            issues.append("no_valid_reference_entries")
        if ref["reference_waveform_count"] != ref["reference_record_count"]:
            issues.append("incomplete_reference_waveform")
        if ref["reference_clock_status"] != "strictly_increasing":
            issues.append("reference_timestamps_need_review")
        if reference_timestamp_unit == "unknown":
            issues.append("reference_timestamp_units_unverified")
        if not hash_images:
            issues.append("image_hash_not_computed")
        issues += ["frame_capture_clock_unverified", "video_reference_alignment_unverified"]
        records.append(dict(
            dataset_id="PURE", participant_id=sequence.participant_id,
            recording_id=sequence.recording_id,
            video_format="ordered_png_sequence",
            raw_video_path=sequence.frames_dir.relative_to(root).as_posix(),
            reference_path=sequence.reference_file.relative_to(root).as_posix(),
            raw_video_sha256=image_sequence_sha256(frames, root) if hash_images and frames else None,
            reference_sha256=ref["reference_sha256"],
            fps_nominal=30.0, video_clock_source="unknown", frames_decoded=None,
            image_file_count=len(frames),
            reference_modality="finger_clip_plethysmography",
            reference_body_site="finger", reference_fs_reported=60.0,
            reference_fs_verified=None, reference_fs=None,
            reference_clock_source=ref["reference_clock_source"],
            reference_timestamp_unit=ref["reference_timestamp_unit"],
            reference_clock_status=ref["reference_clock_status"],
            reference_sample_count=ref["reference_record_count"],
            reference_waveform_count=ref["reference_waveform_count"],
            validity_mask_source="not_yet_extracted",
            sync_evidence=sync_evidence,
            allow_shape_supervision=False, allow_hr_scoring=False,
            allow_absolute_timing_scoring=False,
            admission_state="inventory_only", admission_blockers=sorted(set(issues)),
        ))
    return dict(schema_version=SCHEMA, dataset_id="PURE", root_hint=root.name,
                access_status=access_status, access_verified=(access_status == "verified"),
                license_scope=license_scope,
                license_evidence=license_evidence, recording_count=len(records),
                participant_count=len(set(r["participant_id"] for r in records)),
                reserved_own_subjects=sorted(RESERVED_OWN_SUBJECTS),
                recordings=records)


def draft_participant_split(manifest: Mapping[str, Any], *, seed: int = 42,
                            val_fraction: float = .2, test_fraction: float = .2) -> dict[str, Any]:
    """Generate an explicitly provisional split; validation/evaluation remains locked."""
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Unrecognized Waveform v2 dataset manifest")
    rows = manifest["recordings"]
    ids = sorted(set(row["participant_id"] for row in rows))
    if any(row["dataset_id"] != "PURE" for row in rows):
        raise ValueError("PURE split is restricted to PURE data")
    if any(str(row["participant_id"]).removeprefix("PURE:") in RESERVED_OWN_SUBJECTS
           for row in rows):
        raise ValueError("Reserved own subjects cannot participate in Waveform v2")
    train, validation, test = split_subjects(ids, val_fraction=val_fraction,
                                            test_fraction=test_fraction, seed=seed)
    assert_subject_disjoint(train, validation, test)
    assignments = {p: "train" for p in train}
    assignments.update({p: "validation" for p in validation})
    assignments.update({p: "test" for p in test})
    assert set(assignments) == set(ids)
    return dict(schema_version=SPLIT_SCHEMA, status="DRAFT_NOT_FROZEN",
                source_manifest_sha256=canonical_json_sha256(manifest), dataset_id="PURE",
                seed=seed, val_fraction=val_fraction, test_fraction=test_fraction,
                participant_assignments=assignments,
                recording_assignments={row["recording_id"]: assignments[row["participant_id"]]
                                       for row in rows},
                development_warning="This split is a proposal, not a pristine cross-dataset test."
                                    " Freeze splits before any model/preprocessing selection.")


def canonical_json_sha256(data: Mapping[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def write_new_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """No overwrites or writes inside a source-data tree are allowed by CLI contract."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
