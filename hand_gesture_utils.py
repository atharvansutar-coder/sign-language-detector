"""
Utility functions for hand gesture detection and processing.

Uses the mediapipe Tasks API (mp.tasks.vision.HandLandmarker) which is the
only API available in mediapipe >= 0.10.x on Python 3.12+.
mp.solutions was removed from these builds.
"""
import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import landmark as mp_landmark
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Hand connection pairs (21 landmarks, same topology as the old solutions API)
# Used for drawing the skeleton overlay.
# ---------------------------------------------------------------------------
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (5,9),(9,10),(10,11),(11,12),  # middle
    (9,13),(13,14),(14,15),(15,16),# ring
    (13,17),(17,18),(18,19),(19,20),# pinky
    (0,17),                         # palm base
]

# Path to the .task model file (downloaded alongside this script)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


class _FakeResults:
    """Mimics the old mp.solutions results object so the rest of the code is unchanged."""
    def __init__(self):
        self.multi_hand_landmarks = None
        self.multi_handedness = None


class _FakeLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _FakeHandLandmarks:
    def __init__(self, landmark_list):
        self.landmark = landmark_list


class _FakeClassification:
    def __init__(self, label):
        self.label = label


class _FakeHandedness:
    def __init__(self, label):
        self.classification = [_FakeClassification(label)]


class HandDetector:
    """Hand detection and landmark extraction using MediaPipe Tasks API."""

    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialise the hand detector.

        Args:
            static_image_mode: If True, use IMAGE mode (no tracking).
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for hand tracking.
        """
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"hand_landmarker.task not found at {_MODEL_PATH}\n"
                "Download it with:\n"
                "  curl -L https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task "
                "-o hand_landmarker.task"
            )

        running_mode = (
            mp_vision.RunningMode.IMAGE
            if static_image_mode
            else mp_vision.RunningMode.VIDEO
        )

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._running_mode = running_mode
        self._frame_ts = 0          # monotonic timestamp counter for VIDEO mode
        self.results = _FakeResults()

    # ------------------------------------------------------------------
    # Public interface  (same signatures as the old mp.solutions version)
    # ------------------------------------------------------------------

    def find_hands(self, img, draw=True):
        """
        Detect hands in an image and optionally draw landmarks.

        Args:
            img:  Input image in BGR format (from cv2).
            draw: Whether to overlay the hand skeleton on the image.

        Returns:
            (annotated_img, results)  — results has .multi_hand_landmarks
            and .multi_handedness just like the old API.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        if self._running_mode == mp_vision.RunningMode.VIDEO:
            self._frame_ts += 33          # ~30 fps cadence in ms
            detection = self._landmarker.detect_for_video(mp_image, self._frame_ts)
        else:
            detection = self._landmarker.detect(mp_image)

        # Convert Tasks result → fake solutions-style result
        fake = _FakeResults()
        if detection.hand_landmarks:
            fake.multi_hand_landmarks = []
            fake.multi_handedness = []
            h, w = img.shape[:2]

            for i, hand_lms in enumerate(detection.hand_landmarks):
                # Build landmark objects
                lm_objs = [_FakeLandmark(lm.x, lm.y, lm.z) for lm in hand_lms]
                fake.multi_hand_landmarks.append(_FakeHandLandmarks(lm_objs))

                # Handedness label
                if detection.handedness and i < len(detection.handedness):
                    label = detection.handedness[i][0].category_name  # "Left" or "Right"
                else:
                    label = "Right"
                fake.multi_handedness.append(_FakeHandedness(label))

                # Draw skeleton
                if draw:
                    for start_idx, end_idx in _HAND_CONNECTIONS:
                        x1 = int(hand_lms[start_idx].x * w)
                        y1 = int(hand_lms[start_idx].y * h)
                        x2 = int(hand_lms[end_idx].x * w)
                        y2 = int(hand_lms[end_idx].y * h)
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for lm in hand_lms:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)

        self.results = fake
        return img, fake

    def find_position(self, img, hand_no=0, draw=False):
        """
        Return landmark pixel positions for one hand.

        Returns:
            List of [id, x, y] for each of the 21 landmarks.
        """
        lm_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            h, w = img.shape[:2]
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lm_list

    def get_landmarks_array(self, img, hand_no=0):
        """
        Return normalised (x, y) coordinates as a (21, 2) numpy array.
        Returns None if no hand is detected.
        """
        if not self.results.multi_hand_landmarks:
            return None
        if hand_no >= len(self.results.multi_hand_landmarks):
            return None

        hand = self.results.multi_hand_landmarks[hand_no]
        return np.array([[lm.x, lm.y] for lm in hand.landmark])

    def count_fingers(self, landmarks, hand_label=None):
        """
        Count extended fingers from pixel landmark list.

        Args:
            landmarks: List of [id, x, y] positions.
            hand_label: "Left" or "Right".

        Returns:
            Integer 0-5.
        """
        if not landmarks:
            return 0

        tip_ids = [4, 8, 12, 16, 20]
        fingers = []

        thumb_tip_id  = tip_ids[0]
        thumb_ip_id   = thumb_tip_id - 1

        if thumb_tip_id < len(landmarks) and thumb_ip_id < len(landmarks):
            wrist_x     = landmarks[0][1]
            thumb_tip_x = landmarks[thumb_tip_id][1]
            thumb_ip_x  = landmarks[thumb_ip_id][1]
            thumb_extended = abs(thumb_tip_x - wrist_x) > abs(thumb_ip_x - wrist_x) * 1.2
            fingers.append(1 if thumb_extended else 0)
        else:
            fingers.append(0)

        for id in range(1, 5):
            tip_id = tip_ids[id]
            pip_id = tip_id - 2
            if tip_id < len(landmarks) and pip_id < len(landmarks):
                fingers.append(1 if landmarks[tip_id][2] < landmarks[pip_id][2] else 0)
            else:
                fingers.append(0)

        return sum(fingers)

    def get_hand_label(self, hand_no=0):
        """
        Return "Left" or "Right" for the given hand index.
        """
        if not self.results.multi_handedness:
            return None
        if hand_no >= len(self.results.multi_handedness):
            return None
        return self.results.multi_handedness[hand_no].classification[0].label

