"""Pilot acquisition and cuff provenance, separate from signal-only features."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import math
from typing import Any


def _aware_time(value: str) -> datetime:
    t = datetime.fromisoformat(value)
    if t.utcoffset() is None:
        raise ValueError("Acquisition/reference times require an explicit timezone offset")
    return t


@dataclass
class Subject:
    subject_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    subject_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BPReference:
    sbp: float
    dbp: float
    measurement_index: int
    start_time: str
    end_time: str
    relation_to_video: str
    device: str
    heart_rate_if_available: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.sbp, self.dbp)) or not 0 < self.dbp < self.sbp:
            raise ValueError("Invalid cuff SBP/DBP (mmHg)")
        if _aware_time(self.end_time) < _aware_time(self.start_time):
            raise ValueError("Cuff end time precedes start time")
        if self.relation_to_video not in {"before", "during", "after"} or not self.device or self.measurement_index < 0:
            raise ValueError("Invalid cuff relation/device/index")


@dataclass
class ReferenceAssociation:
    reference_index: int
    interval_start_sec: float
    interval_end_sec: float
    method: str
    notes: str

    def __post_init__(self) -> None:
        if not 0 <= self.interval_start_sec < self.interval_end_sec or not math.isfinite(self.interval_end_sec):
            raise ValueError("Invalid associated video interval")
        if self.method != "nearby_cuff_pilot" or not self.notes:
            raise ValueError("Pilot association must state its nearby-cuff approximation in notes")


@dataclass
class Recording:
    subject_id: str
    session_id: str
    recording_id: str
    video_path: Path
    video_start_time: str
    references: list[BPReference] = field(default_factory=list)
    associations: list[ReferenceAssociation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all((self.subject_id, self.session_id, self.recording_id)):
            raise ValueError("Subject/session/recording IDs are required")
        _aware_time(self.video_start_time)
        indices = [r.measurement_index for r in self.references]
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate cuff measurement indices")
        for association in self.associations:
            if association.reference_index not in indices:
                raise ValueError("Association references an unknown cuff measurement")
        ordered = sorted(self.associations, key=lambda a: a.interval_start_sec)
        if any(a.interval_end_sec > b.interval_start_sec for a,b in zip(ordered, ordered[1:])):
            raise ValueError("Overlapping cuff association intervals are ambiguous")

    @classmethod
    def load(cls, manifest: str | Path) -> "Recording":
        path = Path(manifest).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["references"] = [BPReference(**r) for r in data.get("references", [])]
        data["associations"] = [ReferenceAssociation(**a) for a in data.get("associations", [])]
        video = Path(data["video_path"])
        data["video_path"] = video if video.is_absolute() else path.parent / video
        return cls(**data)
