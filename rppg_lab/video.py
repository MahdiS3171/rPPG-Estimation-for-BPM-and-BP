"""One-pass decoded video clock, with explicit fallback provenance."""
from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
from .types import VideoMetadata, validate_timestamps

LOG = logging.getLogger(__name__)


class VideoReader:
    def __init__(self, path: str | Path, timestamp_mode: str = "auto", timestamps: np.ndarray | None = None):
        import cv2
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if timestamp_mode not in {"auto", "nominal"}:
            raise ValueError("timestamp_mode must be auto or nominal")
        self.external = None if timestamps is None else validate_timestamps(timestamps)
        if self.external is not None and (not len(self.external) or self.external[0] < 0):
            raise ValueError("External clock must contain nonnegative video-relative seconds")
        self.mode = timestamp_mode
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Could not open {self.path}")
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.metadata = VideoMetadata(
            self.path, int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps,
            int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            codec="".join(chr((fourcc >> 8 * i) & 255) for i in range(4)),
            orientation_degrees=float(self.cap.get(getattr(cv2, "CAP_PROP_ORIENTATION_META", 48))),
        )

    def frames(self, max_frames: int | None = None):
        import cv2
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive")
        times: list[float] = []
        try:
            index = 0
            while max_frames is None or index < max_frames:
                ok, bgr = self.cap.read()
                if not ok:
                    break
                raw = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                self.metadata.backend_timestamps.append(raw)
                if self.external is not None:
                    if index >= len(self.external):
                        raise ValueError("External timestamps shorter than decoded video")
                    t, source = float(self.external[index]), "external"
                elif self.mode == "auto" and np.isfinite(raw) and raw >= 0 and (not times or raw > times[-1]):
                    t, source = raw, "container"
                else:
                    fps = self.metadata.fps_nominal
                    if not np.isfinite(fps) or fps <= 0:
                        raise ValueError("No usable timestamps or nominal FPS; supply frame timestamps")
                    t = times[-1] + 1 / fps if times else index / fps
                    source = "nominal" if self.mode == "nominal" else "nominal_fallback"
                times.append(t)
                self.metadata.timestamp_sources.append(source)
                yield index, t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                index += 1
            self.metadata.truncated = max_frames is not None and index == max_frames
            if not self.metadata.truncated:
                self.metadata.decoded_frame_shortfall = max(0,self.metadata.frame_count_reported-len(times))
            if self.external is not None and not self.metadata.truncated and len(times) != len(self.external):
                raise ValueError("External timestamp count differs from decoded frames")
        finally:
            self.metadata.timestamps = np.asarray(times, dtype=np.float64)
            self.cap.release()
        if not times:
            raise ValueError("Video contains no decodable frames")
        if any(s.startswith("nominal") for s in self.metadata.timestamp_sources):
            LOG.warning("Nominal FPS used for some/all timestamps in %s", self.path)

    def close(self) -> None:
        self.cap.release()
