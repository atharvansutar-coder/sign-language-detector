"""
Application mode logic for SIGNBRIDGE (General, Emergency, Hospital).
"""
from dataclasses import dataclass
from typing import Optional

from vocabulary import AppMode, EMERGENCY_SIGNS, get_display, get_word, is_emergency_sign


@dataclass
class ModeState:
    mode: AppMode = AppMode.GENERAL
    active_emergency: bool = False
    emergency_message: str = ""

    def set_mode(self, mode: AppMode) -> None:
        self.mode = mode
        if mode != AppMode.EMERGENCY:
            self.active_emergency = False
            self.emergency_message = ""

    def cycle_mode(self) -> AppMode:
        order = [AppMode.GENERAL, AppMode.EMERGENCY, AppMode.HOSPITAL]
        idx = order.index(self.mode)
        self.set_mode(order[(idx + 1) % len(order)])
        return self.mode

    def handle_sign(self, gesture_id: int, confidence: float) -> Optional[str]:
        """
        Process a confirmed sign in the current mode.
        Returns an emergency alert message if triggered, else None.
        """
        word = get_word(gesture_id)
        if word == "Unknown":
            return None

        if self.mode == AppMode.EMERGENCY:
            if word not in EMERGENCY_SIGNS:
                return None
            if is_emergency_sign(gesture_id):
                self.active_emergency = True
                display = get_display(gesture_id)
                self.emergency_message = f"🚨 {display.upper()} — Assistance needed"
                return self.emergency_message

        if self.mode == AppMode.HOSPITAL and word in EMERGENCY_SIGNS:
            if word in ("EMERGENCY", "HELP", "PAIN"):
                self.active_emergency = True
                display = get_display(gesture_id)
                self.emergency_message = f"🏥 {display} — Patient needs attention"
                return self.emergency_message

        return None

    @property
    def mode_label(self) -> str:
        labels = {
            AppMode.GENERAL: "General",
            AppMode.EMERGENCY: "Emergency",
            AppMode.HOSPITAL: "Hospital",
        }
        return labels[self.mode]

    @property
    def mode_color_bgr(self) -> tuple:
        colors = {
            AppMode.GENERAL: (200, 200, 200),
            AppMode.EMERGENCY: (0, 0, 220),
            AppMode.HOSPITAL: (200, 140, 0),
        }
        return colors[self.mode]
