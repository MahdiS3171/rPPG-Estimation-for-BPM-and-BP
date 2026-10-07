"""Small research data objects. All relative times are seconds; RGB order is RGB."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np


def validate_timestamps(timestamps: np.ndarray) -> np.ndarray:
    t = np.asarray(timestamps, dtype=np.float64)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be finite, one-dimensional and strictly increasing")
    return t


@dataclass
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps_nominal: float
    frame_count_reported: int
    codec: str = ""
    orientation_degrees: float = 0.0
    timestamps: np.ndarray = field(default_factory=lambda: np.empty(0))
    timestamp_sources: list[str] = field(default_factory=list)
    backend_timestamps: list[float] = field(default_factory=list)
    device: str | None = None
    truncated: bool = False
    decoded_frame_shortfall: int = 0

    @property
    def frame_count(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0]) if self.frame_count > 1 else 0.0


@dataclass
class RegionObservation:
    frame_index: int
    timestamp: float
    bbox: tuple[int, int, int, int] | None = None
    landmarks: np.ndarray | None = None
    confidence: float = float("nan")
    confidence_kind: str = "not_exposed"
    valid: bool = False
    identity: str | None = None
    reason: str | None = None


@dataclass
class RGBTrace:
    timestamps: np.ndarray
    values: np.ndarray
    roi_name: str
    validity_mask: np.ndarray
    quality: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamps = validate_timestamps(self.timestamps)
        self.values = np.asarray(self.values, dtype=np.float64)
        self.validity_mask = np.asarray(self.validity_mask, dtype=bool)
        if self.values.shape != (len(self.timestamps), 3) or self.validity_mask.shape != self.timestamps.shape:
            raise ValueError("RGBTrace requires (T,3) RGB and (T,) validity/timestamps")
        if np.any(self.validity_mask & ~np.isfinite(self.values).all(axis=1)):
            raise ValueError("Valid RGB samples must be finite")

    @property
    def r(self) -> np.ndarray:
        return self.values[:, 0]

    @property
    def g(self) -> np.ndarray:
        return self.values[:, 1]

    @property
    def b(self) -> np.ndarray:
        return self.values[:, 2]


@dataclass
class PhysiologicalSignal:
    values: np.ndarray
    timestamps: np.ndarray
    sample_rate: float
    source_region: str
    algorithm: str
    preprocessing: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamps = validate_timestamps(self.timestamps)
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.shape != self.timestamps.shape:
            raise ValueError("Signal length must equal timestamp length")
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive and finite")
        if len(self.timestamps) > 1 and not np.allclose(np.diff(self.timestamps), 1 / self.sample_rate, atol=1e-7, rtol=1e-5):
            raise ValueError("PhysiologicalSignal must use its declared uniform sample rate")


@dataclass
class RegionResult:
    rgb: RGBTrace
    rppg: PhysiologicalSignal
    quality: dict[str, Any]
    observations: list[RegionObservation] = field(default_factory=list)


@dataclass
class SharedTimebase:
    timestamps: np.ndarray
    original_timestamps: np.ndarray


@dataclass
class DualROIResult:
    face: RegionResult
    hand: RegionResult
    shared: SharedTimebase
    video: VideoMetadata
    hr: dict[str, float]
    delay: dict[str, Any]
    bp_features: list[dict[str, Any]]
    exclusions: list[dict[str, Any]]
    config: dict[str, Any]
    # Each RGBTrace uses original timestamps; each signal uses the shared grid.
    # face remains the configured legacy result, also present in this mapping.
    face_rois: dict[str, RegionResult] = field(default_factory=dict)
