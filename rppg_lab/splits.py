"""Subject partitions and checks for mislabeled duplicate/derived samples."""
from __future__ import annotations

from typing import Sequence, Mapping
import numpy as np

WAVEFORM_SPLIT_VERSION = "waveform_v1_sorted_pcg64_test_first_v1"


def waveform_v1_split_manifest(subject_ids: Sequence[str], validation_fraction: float = 0.2,
                               test_fraction: float = 0.2, seed: int = 42) -> dict:
    """Reserve test first, then validation from the remaining participants.

    Sorted unique IDs are permuted with NumPy default_rng (PCG64). Counts use
    Python round, at least one per requested partition, always retaining train.
    Two-participant legacy plumbing runs have no possible three-way split:
    explicitly mark them smoke_only with empty test; never a development split.
    """
    ids = sorted(str(s) for s in subject_ids)
    if len(ids) != len(set(ids)) or any(not s.strip() for s in ids) or len(ids) < 2:
        raise ValueError("Need at least two unique nonempty participant IDs")
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("Validation/test fractions must be in (0,1)")
    order = np.random.default_rng(seed).permutation(ids).tolist()
    smoke = len(ids) == 2
    nt = 0 if smoke else min(len(ids) - 2, max(1, round(len(ids) * test_fraction)))
    remaining = len(ids) - nt
    nv = min(remaining - 1, max(1, round(remaining * validation_fraction)))
    test, val, train = sorted(order[:nt]), sorted(order[nt:nt + nv]), sorted(order[nt + nv:])
    manifest = dict(algorithm_version=WAVEFORM_SPLIT_VERSION, seed=int(seed), all_subject_ids=ids,
                    train_ids=train, validation_ids=val, test_ids=test,
                    test_fraction=float(test_fraction), validation_fraction=float(validation_fraction),
                    validation_fraction_basis="remaining_non_test_participants", smoke_only=smoke)
    validate_waveform_split_manifest(manifest)
    return manifest


def validate_waveform_split_manifest(manifest: Mapping) -> None:
    """Require complete disjoint partitions; refuse ambiguous/unknown schemas."""
    if manifest.get("algorithm_version") != WAVEFORM_SPLIT_VERSION:
        raise ValueError("Incompatible waveform split algorithm/version")
    groups = [manifest[k] for k in ("train_ids", "validation_ids", "test_ids")]
    all_ids = manifest["all_subject_ids"]
    for group in [all_ids, *groups]:
        if not isinstance(group, list) or any(not isinstance(s, str) or not s.strip() for s in group):
            raise ValueError("Split IDs must be lists of nonempty strings")
        if len(group) != len(set(group)):
            raise ValueError("Duplicate participant IDs in split manifest")
    assert_subject_disjoint(*groups)
    if not groups[0] or not groups[1] or set(all_ids) != set().union(*map(set, groups)):
        raise ValueError("Manifest must partition all participants and retain train/validation")
    if not groups[2] and not (manifest.get("smoke_only") is True and len(all_ids) == 2):
        raise ValueError("A development split must reserve locked test participants")
    if not (0 < manifest["validation_fraction"] < 1 and 0 < manifest["test_fraction"] < 1):
        raise ValueError("Invalid manifest fractions")


def assert_subject_disjoint(train: Sequence[str], validation: Sequence[str], test: Sequence[str]) -> None:
    groups = [set(train), set(validation), set(test)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("Subject overlap between partitions")
    if any(not str(s).strip() for group in groups for s in group):
        raise ValueError("Empty subject ID")


def split_subjects(subject_ids: Sequence[str], val_fraction: float = 0.2,
                   test_fraction: float = 0.2, seed: int = 42) -> tuple[list[str], list[str], list[str]]:
    """All sessions/windows of a participant map to one deterministic partition."""
    unique = sorted(set(str(s) for s in subject_ids))
    if not (0 < val_fraction < 1 and 0 <= test_fraction < 1 and val_fraction+test_fraction < 1):
        raise ValueError("Invalid validation/test fractions")
    if len(unique) < (3 if test_fraction else 2) or any(not s.strip() for s in unique):
        raise ValueError("Insufficient or empty subject IDs")
    order = np.random.default_rng(seed).permutation(unique).tolist()
    nv = max(1, round(len(unique)*val_fraction))
    nt = max(1, round(len(unique)*test_fraction)) if test_fraction else 0
    while nv+nt >= len(unique) and nt > 1:
        nt -= 1
    while nv+nt >= len(unique) and nv > 1:
        nv -= 1
    if nv+nt >= len(unique):
        raise ValueError("No training subjects remain")
    result = (sorted(order[nv+nt:]), sorted(order[:nv]), sorted(order[nv:nv+nt]))
    assert_subject_disjoint(*result)
    return result


def validate_sample_partitions(rows: Sequence[Mapping], partitions: Mapping[str, str]) -> None:
    """Check IDs, file fingerprints, derived samples and cross-partition overlap.

    Temporal overlap within a partition is allowed; exact duplicate windows are
    rejected to avoid unintended weighting. Content hashes catch duplicate videos
    assigned a different recording/subject ID.
    """
    if not set(partitions.values()) <= {"train", "validation", "test"}:
        raise ValueError("Unknown partition")
    owners, seen, windows = {}, set(), {}
    for row in rows:
        sid = str(row["subject_id"])
        if sid not in partitions:
            raise ValueError("Sample belongs to an unpartitioned subject")
        split = partitions[sid]
        identity = str(row.get("input_sha256") or row["recording_id"])
        if identity in owners and owners[identity] != sid:
            raise ValueError("Duplicate recording/content assigned to multiple subjects")
        owners[identity] = sid
        a,b = float(row["start_sec"]), float(row["end_sec"])
        if not np.isfinite([a,b]).all() or b <= a:
            raise ValueError("Invalid sample interval")
        key = identity,a,b
        if key in seen:
            raise ValueError("Duplicate derived sample/window")
        seen.add(key)
        for old_a,old_b,old_split in windows.get(identity, []):
            if split != old_split and max(a,old_a) < min(b,old_b):
                raise ValueError("Temporal window overlap across partitions")
        windows.setdefault(identity, []).append((a,b,split))
