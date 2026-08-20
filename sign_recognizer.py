"""
Continuous ISL sign recognition with temporal smoothing and confidence gating.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from gesture_classifier import GestureClassifier
from hand_gesture_utils import HandDetector
from modes import ModeState
from sentence_builder import SentenceBuilder
from stats_tracker import StatsTracker
from vocabulary import (
    CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_MESSAGE,
    SIGN_HOLD_MS,
    AppMode,
    get_display,
    get_word,
    signs_for_mode,
)


@dataclass
class RecognitionResult:
    """Output of one recognition frame."""
    gesture_id: int = 0
    display: str = "—"
    word: str = "Unknown"
    confidence: float = 0.0
    smoothed_confidence: float = 0.0
    is_low_confidence: bool = True
    message: str = ""
    hand_present: bool = False
    hand_label: str = ""


@dataclass
class ConfirmedSign:
    """A sign that passed hold + confidence checks."""
    gesture_id: int
    word: str
    display: str
    confidence: float
    emergency_alert: Optional[str] = None


class SignRecognizer:
    """
    Orchestrates hand detection → classification → smoothing → sentence building.

    Designed for future extension with face/pose landmarks and temporal ML models.
    """

    def __init__(
        self,
        detector: Optional[HandDetector] = None,
        classifier: Optional[GestureClassifier] = None,
        history_size: int = 8,
        hold_ms: int = SIGN_HOLD_MS,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.detector = detector or HandDetector(
            min_detection_confidence=0.65, min_tracking_confidence=0.65
        )
        self.classifier = classifier or GestureClassifier()
        self.mode_state = ModeState()
        self.sentence = SentenceBuilder()
        self.stats = StatsTracker()
        self.conversation: List[dict] = []

        self._history_size = history_size
        self._hold_ms = hold_ms
        self._confidence_threshold = confidence_threshold

        self._id_history: Deque[int] = deque(maxlen=history_size)
        self._conf_history: Deque[float] = deque(maxlen=history_size)

        self._hold_gesture_id: int = 0
        self._hold_start_ms: float = 0.0
        self._last_committed_key: str = ""

        self._current_sign_display: str = "—"
        self._current_confidence: float = 0.0
        self._status_message: str = "Show a sign to the camera"

        # Incremented every time a sign is confirmed. The UI/audio layer can
        # use this to react exactly once to each confirmed sign.
        self.commit_counter: int = 0
        self.last_committed_display: str = ""
        self.last_committed_sentence: Optional[str] = None
        self.last_audio_text: Optional[str] = None
        self.audio_events: List[Tuple[str, float]] = []

    def process_frame(self, img) -> Tuple[any, RecognitionResult]:
        """Process one camera frame. Returns (annotated_img, result)."""
        img, results = self.detector.find_hands(img, draw=True)
        hand_present = bool(results.multi_hand_landmarks)

        self.stats.tick_frame(hand_present)

        if not hand_present:
            self._reset_hold()
            return img, RecognitionResult(
                message=self._status_message,
                hand_present=False,
            )

        # Process primary hand (first detected)
        landmarks = self.detector.get_landmarks_array(img, hand_no=0)
        position_list = self.detector.find_position(img, hand_no=0, draw=False)
        hand_label = self.detector.get_hand_label(hand_no=0) or "Right"
        finger_count = self.detector.count_fingers(position_list, hand_label)

        gesture_id, display, confidence = self.classifier.classify_gesture(
            landmarks, finger_count, hand_label
        )

        # Mode filter: ignore signs outside vocabulary for current mode
        allowed = signs_for_mode(self.mode_state.mode)
        word = get_word(gesture_id)
        if allowed is not None and word not in allowed and gesture_id != 0:
            gesture_id, display, confidence = 0, get_display(0), confidence * 0.3
            word = "Unknown"

        self._id_history.append(gesture_id)
        self._conf_history.append(confidence)

        smoothed_id = self._smooth_gesture()
        smoothed_conf = self._smooth_confidence()
        is_low = smoothed_conf < self._confidence_threshold or smoothed_id == 0

        if is_low:
            self.stats.record_low_confidence()
            self._current_sign_display = get_display(smoothed_id) if smoothed_id else "—"
            self._current_confidence = smoothed_conf
            self._status_message = LOW_CONFIDENCE_MESSAGE if smoothed_id != 0 else "Show a sign to the camera"
            self._reset_hold()
        else:
            self._current_sign_display = get_display(smoothed_id)
            self._current_confidence = smoothed_conf
            self._status_message = ""
            self._update_hold(smoothed_id, smoothed_conf)

        return img, RecognitionResult(
            gesture_id=smoothed_id,
            display=self._current_sign_display,
            word=get_word(smoothed_id),
            confidence=confidence,
            smoothed_confidence=smoothed_conf,
            is_low_confidence=is_low,
            message=self._status_message,
            hand_present=True,
            hand_label=hand_label,
        )

    def _smooth_gesture(self) -> int:
        if not self._id_history:
            return 0
        # Weight recent frames slightly higher
        counts = {}
        for i, gid in enumerate(self._id_history):
            weight = 1.0 + (i / len(self._id_history)) * 0.5
            counts[gid] = counts.get(gid, 0) + weight
        return max(counts, key=counts.get)

    def _smooth_confidence(self) -> float:
        if not self._conf_history:
            return 0.0
        return float(sum(self._conf_history) / len(self._conf_history))

    def _update_hold(self, gesture_id: int, confidence: float) -> None:
        now_ms = time.time() * 1000
        if gesture_id != self._hold_gesture_id:
            self._hold_gesture_id = gesture_id
            self._hold_start_ms = now_ms
            return

        if (now_ms - self._hold_start_ms) >= self._hold_ms:
            key = f"{gesture_id}_{get_word(gesture_id)}"
            if key != self._last_committed_key:
                self._commit_sign(gesture_id, confidence)
                self._last_committed_key = key

    def _commit_sign(self, gesture_id: int, confidence: float) -> None:
        word = get_word(gesture_id)
        display = get_display(gesture_id)
        if word == "Unknown":
            return

        alert = self.mode_state.handle_sign(gesture_id, confidence)
        matched_sentence = self.sentence.append(word, display)
        self.stats.record_sign(word)

        self.commit_counter += 1
        self.last_committed_display = display
        self.last_committed_sentence = matched_sentence
        # Every confirmed sign produces audio. If the sign sequence also
        # completes a known sentence, the full sentence is queued afterwards.
        # The AudioAnnouncer worker inserts the delay without blocking OpenCV.
        self.last_audio_text = matched_sentence or display
        self.audio_events.append((display, 0.65))
        if matched_sentence:
            self.audio_events.append((matched_sentence, 0.90))
        if alert:
            self.stats.record_emergency_trigger()

        if matched_sentence:
            # Replace the individual sign entries that formed this sentence
            # with one clean sentence entry in the conversation log.
            remove_count = self.sentence.matched_length
            removed = 0
            while removed < remove_count and self.conversation:
                if self.conversation[-1].get("role") == "sign":
                    self.conversation.pop()
                    removed += 1
                else:
                    break
            self.conversation.append({
                "role": "sentence",
                "text": matched_sentence,
                "word": "SENTENCE",
                "confidence": round(confidence, 2),
            })
        else:
            self.conversation.append({
                "role": "sign",
                "text": display,
                "word": word,
                "confidence": round(confidence, 2),
            })

    def _reset_hold(self) -> None:
        self._hold_gesture_id = 0
        self._hold_start_ms = 0.0

    def set_mode(self, mode: AppMode) -> None:
        self.mode_state.set_mode(mode)
        self._last_committed_key = ""

    def cycle_mode(self) -> AppMode:
        mode = self.mode_state.cycle_mode()
        self._last_committed_key = ""
        return mode

    def add_speech_message(self, text: str) -> None:
        """Add a speech-to-text message to the conversation."""
        if text.strip():
            self.conversation.append({"role": "speech", "text": text.strip()})

    def clear_sentence(self) -> None:
        self.sentence.clear()
        self._last_committed_key = ""
        self.last_committed_sentence = None
        self.last_audio_text = None
        self.audio_events.clear()

    def drain_audio_events(self) -> List[Tuple[str, float]]:
        """Return and clear confirmed-sign/sentence audio events."""
        events = list(self.audio_events)
        self.audio_events.clear()
        return events

    def undo_last_sign(self) -> None:
        self.sentence.undo_last()
        self._last_committed_key = ""
