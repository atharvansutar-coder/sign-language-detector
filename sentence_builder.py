"""Build readable sentences from confirmed ISL signs."""
import time
from typing import List, Optional


class SentenceBuilder:
    """Collect confirmed signs and convert common sequences into sentences."""

    SENTENCE_PATTERNS = {
        ("HELLO", "HOW_ARE_YOU"): "Hello, how are you?",
        ("I", "AM_GOOD"): "I am good.",
        ("PLEASE", "GIVE", "WATER"): "Please give me water.",
        ("I", "NEED", "HELP"): "I need help.",
        ("I", "NEED", "DOCTOR"): "I need a doctor.",
        ("PLEASE", "CALL", "POLICE"): "Please call the police.",
        ("I", "HAVE", "PAIN"): "I have pain.",
        ("PLEASE", "GIVE", "MEDICINE"): "Please give me medicine.",
        ("I", "WANT", "FOOD"): "I want food.",
        ("I", "WANT", "GO", "HOME"): "I want to go home.",
        ("HELLO", "YES"): "Hello, yes.",
        ("HELLO", "HELP"): "Hello, I need help.",
        ("PLEASE", "WATER"): "Please give me water.",
        ("DOCTOR", "HELP"): "I need a doctor.",
        ("PAIN", "DOCTOR", "HELP"): "I am in pain, I need a doctor.",
        ("THANK_YOU", "HELLO"): "Hello, thank you.",
        ("PLEASE", "HELP"): "Please help me.",
        ("CALL", "POLICE"): "Please call the police.",
        ("EMERGENCY", "HELP"): "This is an emergency, please help.",
        ("PAIN", "DOCTOR"): "I have pain, I need a doctor.",
        ("BATHROOM", "PLEASE"): "Please show me the bathroom.",
        ("MEDICINE", "PLEASE"): "Please give me medicine.",
        ("WATER", "PLEASE"): "Please give me water.",
        ("FAMILY", "HELP"): "Please contact my family.",
        ("THANK_YOU", "HELP"): "Thank you for your help.",
        ("HELLO", "THANK_YOU"): "Hello, thank you.",
        ("YES", "HELP"): "Yes, please help me.",
        ("NO", "HELP"): "No, I need help.",
        ("STOP", "HELP"): "Please stop and help me.",
    }

    def __init__(self, pause_ms: int = 1800):
        self._words: List[str] = []
        self._displays: List[str] = []
        self._last_commit_time = 0.0
        self._pause_ms = pause_ms
        self._matched_sentence: Optional[str] = None
        self._matched_length = 0

    @property
    def text(self) -> str:
        return self._matched_sentence or ""

    @property
    def words(self) -> List[str]:
        return list(self._words)

    @property
    def matched_sentence(self) -> Optional[str]:
        return self._matched_sentence

    @property
    def matched_length(self) -> int:
        return self._matched_length

    def append(self, word: str, display: str) -> Optional[str]:
        now = time.time() * 1000
        self._matched_sentence = None
        self._matched_length = 0

        # Do not commit the same sign repeatedly while the user holds it.
        if self._words and self._words[-1] == word:
            if now - self._last_commit_time < self._pause_ms:
                return None

        self._words.append(word)
        self._displays.append(display)
        self._last_commit_time = now

        # Prefer the longest matching suffix.
        matches = []
        for pattern, sentence in self.SENTENCE_PATTERNS.items():
            if len(self._words) >= len(pattern) and tuple(self._words[-len(pattern):]) == pattern:
                matches.append((len(pattern), sentence))

        if matches:
            self._matched_length, self._matched_sentence = max(matches, key=lambda x: x[0])
            return self._matched_sentence
        return None

    def clear(self) -> None:
        self._words.clear()
        self._displays.clear()
        self._last_commit_time = 0.0
        self._matched_sentence = None
        self._matched_length = 0

    def undo_last(self) -> Optional[str]:
        if not self._words:
            return None
        self._words.pop()
        result = self._displays.pop()
        self._matched_sentence = None
        self._matched_length = 0
        return result
