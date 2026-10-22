"""Single-visible-hand adapter; inherited geometry/identity rules are unchanged."""
from __future__ import annotations

from .detection import HandRegionDetector, _choose_backend, _legacy_solutions


class PilotHandDetector(HandRegionDetector):
    """One candidate for the protocol's one visible hand.

    Avoids searching for a nonexistent second hand on every frame. A new hand
    identity is still rejected by the inherited anatomical lock/jump checks.
    This does not modify the historical two-candidate detector default.
    """

    def __init__(self, config):
        self.backend = _choose_backend(config.detector_backend)
        self.expected_hand = config.expected_hand
        self.input_mirrored = config.input_mirrored
        self.max_jump_fraction = config.max_hand_jump_fraction
        self._identity = self._center = None
        self._last_ms = -1
        if self.backend == "legacy":
            self._model = _legacy_solutions().hands.Hands(
                static_image_mode=False,max_num_hands=1,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence)
        else:
            import mediapipe as mp
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=self._base_options(config.hand_model),running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=1,min_hand_detection_confidence=config.min_detection_confidence,
                min_hand_presence_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence)
            self._model = mp.tasks.vision.HandLandmarker.create_from_options(options)
