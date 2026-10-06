"""Subject partitions and checks for mislabeled duplicate/derived samples."""
from __future__ import annotations

from typing import Sequence, Mapping
import numpy as np


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
