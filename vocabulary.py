"""
ISL vocabulary definitions for SIGNBRIDGE.

Add new signs here — each entry maps a gesture_id (from gesture_classifier)
to display text and mode/category metadata.
"""
from enum import Enum
from typing import Dict, List, Optional, Set


class AppMode(str, Enum):
    GENERAL = "general"
    EMERGENCY = "emergency"
    HOSPITAL = "hospital"


# gesture_id → sign metadata
# gesture_ids are defined in gesture_classifier.GestureClassifier
VOCABULARY: Dict[int, dict] = {
    0:  {"word": "Unknown",      "display": "—",           "categories": set()},
    1:  {"word": "A",            "display": "A",           "categories": {"alphabet"}},
    2:  {"word": "B",            "display": "B",           "categories": {"alphabet", "stop"}},
    3:  {"word": "C",            "display": "C",           "categories": {"alphabet"}},
    4:  {"word": "HELLO",        "display": "Hello",       "categories": {"social", "hospital"}},
    5:  {"word": "GOODBYE",      "display": "Goodbye",     "categories": {"social"}},
    6:  {"word": "OK",           "display": "OK",          "categories": {"social", "hospital"}},
    7:  {"word": "YES",          "display": "Yes",         "categories": {"common", "emergency", "hospital"}},
    8:  {"word": "NO",           "display": "No",          "categories": {"common", "emergency", "hospital"}},
    9:  {"word": "POINT",        "display": "Point",       "categories": {"alphabet"}},
    10: {"word": "HELP",         "display": "Help",        "categories": {"emergency", "hospital"}},
    11: {"word": "EMERGENCY",    "display": "Emergency",   "categories": {"emergency", "hospital"}},
    12: {"word": "DOCTOR",       "display": "Doctor",      "categories": {"emergency", "hospital"}},
    13: {"word": "PAIN",         "display": "Pain",        "categories": {"emergency", "hospital"}},
    14: {"word": "STOP",         "display": "Stop",        "categories": {"emergency"}},
    15: {"word": "CALL",         "display": "Call",        "categories": {"emergency", "hospital"}},
    16: {"word": "WATER",        "display": "Water",       "categories": {"emergency", "hospital"}},
    17: {"word": "POLICE",       "display": "Police",      "categories": {"emergency"}},
    18: {"word": "MEDICINE",     "display": "Medicine",    "categories": {"hospital"}},
    19: {"word": "BATHROOM",     "display": "Bathroom",    "categories": {"hospital"}},
    20: {"word": "FAMILY",       "display": "Family",      "categories": {"hospital"}},
    21: {"word": "THANK_YOU",    "display": "Thank you",   "categories": {"social", "hospital"}},
    22: {"word": "PLEASE",       "display": "Please",      "categories": {"social", "hospital"}},
}

EMERGENCY_SIGNS: Set[str] = {
    "HELP", "EMERGENCY", "DOCTOR", "PAIN", "STOP", "CALL",
    "YES", "NO", "WATER", "POLICE",
}

HOSPITAL_SIGNS: Set[str] = {
    "PAIN", "DOCTOR", "MEDICINE", "WATER", "BATHROOM", "FAMILY",
    "EMERGENCY", "YES", "NO", "HELP", "CALL", "HELLO", "OK",
    "PLEASE", "THANK_YOU",
}

# Minimum confidence to accept a sign (0–1)
CONFIDENCE_THRESHOLD = 0.55
LOW_CONFIDENCE_MESSAGE = "Please repeat the sign."

# Hold duration before a sign is committed to the sentence (ms)
SIGN_HOLD_MS = 700

# Pause after last sign before starting a new word group (ms)
SENTENCE_PAUSE_MS = 1800


def get_word(gesture_id: int) -> str:
    entry = VOCABULARY.get(gesture_id, VOCABULARY[0])
    return entry["word"]


def get_display(gesture_id: int) -> str:
    entry = VOCABULARY.get(gesture_id, VOCABULARY[0])
    return entry["display"]


def is_emergency_sign(gesture_id: int) -> bool:
    return get_word(gesture_id) in EMERGENCY_SIGNS


def signs_for_mode(mode: AppMode) -> Optional[Set[str]]:
    """Return allowed sign words for a mode, or None for all signs."""
    if mode == AppMode.EMERGENCY:
        return EMERGENCY_SIGNS
    if mode == AppMode.HOSPITAL:
        return HOSPITAL_SIGNS
    return None


def list_vocabulary(mode: AppMode = AppMode.GENERAL) -> List[dict]:
    """Return vocabulary entries relevant to the current mode."""
    allowed = signs_for_mode(mode)
    result = []
    for gid, meta in VOCABULARY.items():
        if gid == 0:
            continue
        if allowed is None or meta["word"] in allowed:
            result.append({"id": gid, **meta})
    return result
