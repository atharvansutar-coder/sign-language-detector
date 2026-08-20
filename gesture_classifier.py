"""
Gesture classification module for ISL sign recognition.

Returns (gesture_id, gesture_name, confidence) for each frame.
Rule-based — designed for extension with ML models (LSTM/Transformer) later.
"""
import numpy as np
from typing import List, Optional, Tuple

from vocabulary import VOCABULARY, get_display


class GestureClassifier:
    """Classify hand gestures based on finger positions and hand landmarks."""

    def __init__(self):
        self.gesture_names = [VOCABULARY[i]["display"] for i in sorted(VOCABULARY.keys())]
        self._id_map = sorted(VOCABULARY.keys())

    def classify_gesture(
        self,
        landmarks: Optional[np.ndarray],
        finger_count: int = 0,
        hand_label: Optional[str] = None,
    ) -> Tuple[int, str, float]:
        """
        Classify hand gesture based on landmarks and finger count.

        Returns:
            (gesture_id, display_name, confidence 0–1)
        """
        if landmarks is None:
            return 0, get_display(0), 0.0

        normalized = landmarks - landmarks[0]
        gesture_id, confidence = self._analyze_gesture(normalized, finger_count, hand_label)
        return gesture_id, get_display(gesture_id), confidence

    def _get_distance(self, p1, p2) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _finger_states(self, lm: np.ndarray, hand_scale: float, hand_label: str) -> dict:
        """Compute finger up/down states and helper metrics."""
        wrist, thumb_tip, thumb_ip = lm[0], lm[4], lm[3]
        index_mcp, index_pip, index_tip = lm[5], lm[6], lm[8]
        middle_mcp, middle_pip, middle_tip = lm[9], lm[10], lm[12]
        ring_pip, ring_tip = lm[14], lm[16]
        pinky_mcp, pinky_pip, pinky_tip = lm[17], lm[18], lm[20]

        index_up = index_tip[1] < index_pip[1] - 0.02 * hand_scale
        middle_up = middle_tip[1] < middle_pip[1] - 0.02 * hand_scale
        ring_up = ring_tip[1] < ring_pip[1] - 0.02 * hand_scale
        pinky_up = pinky_tip[1] < pinky_pip[1] - 0.02 * hand_scale

        index_down = index_tip[1] > index_pip[1]
        middle_down = middle_tip[1] > middle_pip[1]
        ring_down = ring_tip[1] > ring_pip[1]
        pinky_down = pinky_tip[1] > pinky_pip[1]

        thumb_vertical = thumb_tip[1] < thumb_ip[1] - 0.02 * hand_scale
        dist_thumb_pinky = self._get_distance(thumb_tip, pinky_mcp)
        thumb_tucked = dist_thumb_pinky < 0.55 * hand_scale

        if hand_label == "Left":
            thumb_out = thumb_tip[0] < thumb_ip[0] - 0.08 * hand_scale
        else:
            thumb_out = thumb_tip[0] > thumb_ip[0] + 0.08 * hand_scale

        dist_thumb_index_tip = self._get_distance(thumb_tip, index_tip)
        dist_thumb_index_mcp = self._get_distance(thumb_tip, index_mcp)

        return {
            "hand_scale": hand_scale,
            "index_up": index_up, "middle_up": middle_up,
            "ring_up": ring_up, "pinky_up": pinky_up,
            "index_down": index_down, "middle_down": middle_down,
            "ring_down": ring_down, "pinky_down": pinky_down,
            "thumb_vertical": thumb_vertical, "thumb_tucked": thumb_tucked,
            "thumb_out": thumb_out,
            "dist_thumb_index_tip": dist_thumb_index_tip,
            "dist_thumb_index_mcp": dist_thumb_index_mcp,
            "thumb_tip": thumb_tip, "thumb_ip": thumb_ip,
            "index_mcp": index_mcp, "index_tip": index_tip,
            "middle_tip": middle_tip, "middle_pip": middle_pip,
            "ring_tip": ring_tip, "ring_pip": ring_pip,
            "pinky_tip": pinky_tip, "pinky_pip": pinky_pip,
            "wrist": wrist,
        }

    def _analyze_gesture(
        self, landmarks: np.ndarray, finger_count: int, hand_label: str
    ) -> Tuple[int, float]:
        """Return (gesture_id, confidence)."""
        hand_scale = self._get_distance(landmarks[0], landmarks[9])
        if hand_scale == 0:
            hand_scale = 1.0

        s = self._finger_states(landmarks, hand_scale, hand_label or "Right")
        hs = s["hand_scale"]

        candidates: List[Tuple[int, float]] = []

        def add(gid: int, conf: float) -> None:
            if conf > 0:
                candidates.append((gid, min(conf, 1.0)))

        # --- Fist family: YES, NO, A, PAIN ---
        if s["index_down"] and s["middle_down"] and s["ring_down"] and s["pinky_down"]:
            if s["thumb_tip"][1] < s["index_mcp"][1] - 0.18 * hs and s["thumb_vertical"]:
                conf = 0.7 + min(0.3, (s["index_mcp"][1] - s["thumb_tip"][1]) / hs * 0.3)
                add(7, conf)   # YES
            elif s["thumb_tip"][1] > s["index_mcp"][1] + 0.12 * hs:
                conf = 0.65 + min(0.35, (s["thumb_tip"][1] - s["index_mcp"][1]) / hs * 0.3)
                add(8, conf)   # NO
            elif s["thumb_vertical"] and s["dist_thumb_index_mcp"] < 0.45 * hs:
                add(1, 0.72)   # Letter A
            else:
                add(13, 0.68)  # PAIN (closed fist)

        # --- Open palm: B, HELLO, STOP, HELP ---
        if s["index_up"] and s["middle_up"] and s["ring_up"] and s["pinky_up"]:
            if s["thumb_tucked"]:
                add(2, 0.78)   # B / STOP
                add(14, 0.72)  # STOP
            elif s["thumb_out"] or s["thumb_vertical"]:
                add(4, 0.80)   # HELLO
                add(10, 0.70)  # HELP (open palm)
                add(21, 0.65)  # THANK YOU

        # --- OK sign: OK, MEDICINE, FAMILY ---
        if s["dist_thumb_index_tip"] < 0.10 * hs:
            if s["middle_up"] and s["ring_up"] and s["pinky_up"]:
                add(6, 0.82)   # OK
                add(18, 0.70)  # MEDICINE
                add(20, 0.68)  # FAMILY

        # --- Peace: GOODBYE ---
        if s["index_up"] and s["middle_up"] and s["ring_down"] and s["pinky_down"]:
            gap = abs(s["index_tip"][0] - s["middle_tip"][0])
            conf = 0.75 + min(0.2, gap / hs * 0.2)
            add(5, conf)

        # --- Point: POINT, POLICE ---
        if s["index_up"] and s["middle_down"] and s["ring_down"] and s["pinky_down"]:
            if s["index_tip"][1] < s["index_mcp"][1] - 0.12 * hs:
                add(9, 0.78)
                add(17, 0.72)

        # --- Three fingers: WATER ---
        if s["index_up"] and s["middle_up"] and s["ring_up"] and s["pinky_down"]:
            add(16, 0.75)

        # --- Shaka: DOCTOR, CALL ---
        if s["thumb_out"] and s["pinky_up"] and s["index_down"] and s["middle_down"] and s["ring_down"]:
            add(12, 0.76)
            add(15, 0.74)

        # --- Rock (index+pinky): EMERGENCY ---
        if s["index_up"] and s["pinky_up"] and s["middle_down"] and s["ring_down"]:
            add(11, 0.73)

        # --- C shape ---
        if s["index_up"] and s["middle_up"]:
            d = s["dist_thumb_index_tip"]
            if 0.18 * hs < d < 0.65 * hs and not s["thumb_tucked"]:
                if not (s["ring_up"] and s["pinky_up"]):
                    add(3, 0.70)

        # --- Thumb between index & middle: BATHROOM ---
        if s["middle_up"] and s["index_up"] and s["ring_down"] and s["pinky_down"]:
            if 0.08 * hs < s["dist_thumb_index_tip"] < 0.20 * hs:
                add(19, 0.68)

        # --- Flat hand tilted: PLEASE (3 fingers partial) ---
        if s["index_up"] and s["middle_up"] and s["ring_down"] and s["pinky_down"] and s["thumb_tucked"]:
            add(22, 0.62)

        if not candidates:
            return 0, 0.0

        # Pick highest confidence; boost if multiple rules agree
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_id, best_conf = candidates[0]
        agreeing = sum(1 for gid, _ in candidates if gid == best_id)
        if agreeing > 1:
            best_conf = min(1.0, best_conf + 0.05)
        return best_id, best_conf
