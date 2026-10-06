"""Detector adapters. Geometry is independent of ROI selection and extraction."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from .types import RegionObservation


class BaseRegionDetector(ABC):
    @abstractmethod
    def detect(self, frame_rgb: np.ndarray, frame_index: int, timestamp: float) -> RegionObservation:
        """Return a valid observation or an explicit missing observation."""

    def close(self) -> None:
        pass


def _legacy_solutions():
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        return mp.solutions
    try:
        from mediapipe.python import solutions
        return solutions
    except ImportError:
        return None


def _choose_backend(backend: str) -> str:
    if backend not in {"auto", "tasks", "legacy"}:
        raise ValueError("Unknown detector backend")
    if backend == "auto":
        return "legacy" if _legacy_solutions() is not None else "tasks"
    if backend == "legacy" and _legacy_solutions() is None:
        raise ImportError("MediaPipe legacy solutions unavailable; use Tasks and run scripts/setup_landmarkers.py")
    return backend


def _points(landmarks, frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return np.asarray([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float64)


def _observation(points: np.ndarray, frame: np.ndarray, index: int, timestamp: float, **kwargs) -> RegionObservation:
    if not np.isfinite(points).all():
        return RegionObservation(index, timestamp, reason="invalid_landmarks")
    h, w = frame.shape[:2]
    lo = np.floor(points.min(axis=0)).astype(int)
    hi = np.ceil(points.max(axis=0)).astype(int) + 1
    bbox = (int(np.clip(lo[0], 0, w)), int(np.clip(lo[1], 0, h)),
            int(np.clip(hi[0], 0, w)), int(np.clip(hi[1], 0, h)))
    return RegionObservation(index, timestamp, bbox=bbox, landmarks=points, valid=True, **kwargs)


class _MediaPipeDetector(BaseRegionDetector):
    def _task_result(self, frame_rgb: np.ndarray, timestamp: float):
        import mediapipe as mp
        # Only the backend's integer timestamp is rounded; scientific times stay
        # untouched. Prevent equal ms timestamps at high capture rates.
        ms = max(self._last_ms + 1, int(round(timestamp * 1000)))
        self._last_ms = ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        return self._model.detect_for_video(image, ms)

    def close(self) -> None:
        self._model.close()

    @staticmethod
    def _base_options(path: str | Path):
        import mediapipe as mp
        if not Path(path).is_file():
            raise FileNotFoundError(f"Landmarker model missing: {path}. Run python scripts/setup_landmarkers.py")
        return mp.tasks.BaseOptions(model_asset_path=str(Path(path).resolve()))


class FaceRegionDetector(_MediaPipeDetector):
    def __init__(self, backend: str = "auto", model_path: str | Path = "models/face_landmarker.task",
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.backend = _choose_backend(backend)
        self._last_ms = -1
        if self.backend == "legacy":
            self._model = _legacy_solutions().face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence)
        else:
            import mediapipe as mp
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=self._base_options(model_path), running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1, min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
            self._model = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_rgb: np.ndarray, frame_index: int, timestamp: float) -> RegionObservation:
        if self.backend == "legacy":
            result = self._model.process(frame_rgb)
            groups = [g.landmark for g in (result.multi_face_landmarks or [])]
        else:
            groups = self._task_result(frame_rgb, timestamp).face_landmarks
        if not groups:
            return RegionObservation(frame_index, timestamp, reason="face_not_detected")
        # FaceMesh/FaceLandmarker do not expose per-frame detection scores. Do
        # not substitute a threshold or 1.0 as measured confidence.
        return _observation(_points(groups[0], frame_rgb), frame_rgb, frame_index, timestamp, identity="face")


class HandRegionDetector(_MediaPipeDetector):
    def __init__(self, backend: str = "auto", model_path: str | Path = "models/hand_landmarker.task",
                 expected_hand: str = "any", input_mirrored: bool = False,
                 max_jump_fraction: float = 0.15, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.backend = _choose_backend(backend)
        self.expected_hand = expected_hand
        self.input_mirrored = input_mirrored
        self.max_jump_fraction = max_jump_fraction
        self._identity: str | None = None
        self._center: np.ndarray | None = None
        self._last_ms = -1
        if self.backend == "legacy":
            self._model = _legacy_solutions().hands.Hands(
                static_image_mode=False, max_num_hands=2,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence)
        else:
            import mediapipe as mp
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=self._base_options(model_path), running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2, min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
            self._model = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _select(self, candidates: list[tuple[np.ndarray, str, float]], frame_rgb: np.ndarray,
                frame_index: int, timestamp: float) -> RegionObservation:
        """Lock anatomical handedness and reject large jumps, including after loss."""
        target = self._identity or (None if self.expected_hand == "any" else self.expected_hand)
        candidates = [c for c in candidates if target is None or c[1] == target]
        if not candidates:
            return RegionObservation(frame_index, timestamp, identity=target, reason="hand_not_detected")
        candidates.sort(key=lambda c: -c[2] if self._center is None else np.linalg.norm(c[0].mean(axis=0) - self._center))
        points, label, score = candidates[0]
        center = points.mean(axis=0)
        h, w = frame_rgb.shape[:2]
        if self._center is not None and np.linalg.norm(center - self._center) > self.max_jump_fraction * np.hypot(h, w):
            return RegionObservation(frame_index, timestamp, identity=target, reason="hand_tracking_jump")
        self._identity, self._center = label, center
        return _observation(points, frame_rgb, frame_index, timestamp, identity=label,
                            confidence=score, confidence_kind="handedness_classification")

    def detect(self, frame_rgb: np.ndarray, frame_index: int, timestamp: float) -> RegionObservation:
        # MediaPipe handedness assumes mirrored input. Mirror only detector
        # input when necessary, then map geometry back to original coordinates.
        detector_frame = frame_rgb if self.input_mirrored else np.ascontiguousarray(frame_rgb[:, ::-1])
        candidates = []
        if self.backend == "legacy":
            result = self._model.process(detector_frame)
            groups = [g.landmark for g in (result.multi_hand_landmarks or [])]
            labels = [(g.classification[0].label, g.classification[0].score) for g in (result.multi_handedness or [])]
        else:
            result = self._task_result(detector_frame, timestamp)
            groups = result.hand_landmarks
            labels = [(g[0].category_name, g[0].score) for g in result.handedness]
        for group, (label, score) in zip(groups, labels):
            points = _points(group, frame_rgb)
            if not self.input_mirrored:
                points[:, 0] = frame_rgb.shape[1] - 1 - points[:, 0]
            candidates.append((points, label, float(score)))
        return self._select(candidates, frame_rgb, frame_index, timestamp)
