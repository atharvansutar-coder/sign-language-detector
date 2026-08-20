"""
FPS and recognition statistics for SIGNBRIDGE.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict


@dataclass
class RecognitionStats:
    total_frames: int = 0
    hands_detected: int = 0
    signs_recognized: int = 0
    low_confidence_count: int = 0
    emergency_triggers: int = 0
    sign_counts: Dict[str, int] = field(default_factory=dict)

    def record_sign(self, word: str) -> None:
        self.signs_recognized += 1
        self.sign_counts[word] = self.sign_counts.get(word, 0) + 1

    def record_low_confidence(self) -> None:
        self.low_confidence_count += 1


class StatsTracker:
    """Track FPS and recognition metrics over a rolling window."""

    def __init__(self, fps_window: int = 30):
        self._frame_times: Deque[float] = deque(maxlen=fps_window)
        self._last_frame_time: float = time.perf_counter()
        self.stats = RecognitionStats()

    def tick_frame(self, hand_present: bool) -> None:
        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._frame_times.append(dt)
        self.stats.total_frames += 1
        if hand_present:
            self.stats.hands_detected += 1

    def record_sign(self, word: str) -> None:
        self.stats.record_sign(word)

    def record_low_confidence(self) -> None:
        self.stats.record_low_confidence()

    def record_emergency_trigger(self) -> None:
        self.stats.emergency_triggers += 1

    @property
    def fps(self) -> float:
        if not self._frame_times:
            return 0.0
        avg_dt = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    @property
    def hand_detection_rate(self) -> float:
        if self.stats.total_frames == 0:
            return 0.0
        return self.stats.hands_detected / self.stats.total_frames
