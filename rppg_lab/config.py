"""Validated stdlib JSON configuration for new research runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math


@dataclass(frozen=True)
class PipelineConfig:
    regions: tuple[str, ...] = ("face", "hand")
    method: str = "GREEN"
    sample_rate: float = 30.0
    face_roi: str = "combined"
    hand_roi: str = "palm"
    skin_mask: bool = False
    skin_cr_range: tuple[int, int] = (133, 173)
    skin_cb_range: tuple[int, int] = (77, 127)
    min_pixels: int = 50
    detector_backend: str = "auto"
    face_model: str = "models/face_landmarker.task"
    hand_model: str = "models/hand_landmarker.task"
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    expected_hand: str = "any"
    input_mirrored: bool = False
    max_hand_jump_fraction: float = 0.15
    timestamp_mode: str = "auto"
    allow_nominal_timing: bool = False
    max_gap_sec: float = 0.15
    min_segment_sec: float = 4.0
    filter_low_hz: float = 0.7
    filter_high_hz: float = 4.0
    filter_order: int = 3
    edge_guard_sec: float = 1.0
    hr_min_hz: float = 0.7
    hr_max_hz: float = 3.5
    window_sec: float = 10.0
    stride_sec: float = 5.0
    max_delay_sec: float = 0.3
    min_valid_fraction: float = 0.9
    min_snr_db: float = 0.0
    min_delay_correlation: float = 0.5
    max_hr_difference_bpm: float = 5.0
    seed: int = 42
    debug_every: int = 60
    # Additive extraction; face_roi still selects the legacy HR/timing output.
    face_rois: tuple[str, ...] = ("forehead", "left_cheek", "right_cheek", "combined")

    def __post_init__(self) -> None:
        from .classical import METHOD_FUNCS
        if not self.regions or len(set(self.regions)) != len(self.regions) or not set(self.regions) <= {"face", "hand"}:
            raise ValueError("regions must contain face and/or hand once")
        if self.method not in METHOD_FUNCS:
            raise ValueError(f"Unknown method {self.method}")
        from .region_masks import FACE_MASK_STRATEGIES
        if self.face_roi not in FACE_MASK_STRATEGIES:
            raise ValueError("Unknown face_roi")
        if (not self.face_rois or isinstance(self.face_rois, str) or
                len(set(self.face_rois)) != len(self.face_rois) or
                not set(self.face_rois) <= FACE_MASK_STRATEGIES):
            raise ValueError("face_rois must contain known, unique facial ROIs")
        if self.hand_roi not in {"palm", "back_of_hand", "landmark_polygon", "bounding_box", "skin_masked_bbox"}:
            raise ValueError("Unknown hand_roi")
        if self.detector_backend not in {"auto", "legacy", "tasks"} or self.timestamp_mode not in {"auto", "nominal"}:
            raise ValueError("Unknown detector backend or timestamp mode")
        if self.expected_hand not in {"any", "Left", "Right"}:
            raise ValueError("expected_hand must be any, Left or Right (anatomical)")
        for name in ("sample_rate", "max_gap_sec", "min_segment_sec", "window_sec", "stride_sec", "max_delay_sec", "max_hand_jump_fraction"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.filter_low_hz < self.filter_high_hz < self.sample_rate / 2:
            raise ValueError("Filter cutoffs must lie below Nyquist")
        if not 0 < self.hr_min_hz < self.hr_max_hz <= self.filter_high_hz:
            raise ValueError("Invalid HR frequency range")
        for name in ("min_valid_fraction", "min_detection_confidence", "min_tracking_confidence", "min_delay_correlation"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0,1]")
        for name in ("min_pixels", "filter_order", "debug_every", "seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.min_pixels < 1 or self.filter_order < 1 or self.debug_every < 0 or not math.isfinite(self.edge_guard_sec) or self.edge_guard_sec < 0:
            raise ValueError("Invalid pixel/filter/debug/edge configuration")
        for name in ("skin_mask", "input_mirrored", "allow_nominal_timing"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false")
        if not math.isfinite(self.min_snr_db) or not math.isfinite(self.max_hr_difference_bpm) or self.max_hr_difference_bpm < 0:
            raise ValueError("Invalid quality thresholds")
        for interval in (self.skin_cr_range, self.skin_cb_range):
            if len(interval) != 2 or not 0 <= interval[0] <= interval[1] <= 255:
                raise ValueError("Skin chroma bounds must lie in [0,255]")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def requested_face_rois(self) -> tuple[str, ...]:
        """Always include the legacy selector, even for an explicit ROI subset."""
        return tuple(dict.fromkeys((*self.face_rois, self.face_roi)))

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for name in ("regions", "skin_cr_range", "skin_cb_range", "face_rois"):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)
