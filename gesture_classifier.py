"""
Gesture classification module for recognizing different hand gestures.
"""
import numpy as np
from typing import List, Optional, Tuple


class GestureClassifier:
    """Classify hand gestures based on finger positions and hand landmarks."""
    
    def __init__(self):
        """Initialize the gesture classifier."""
        self.gesture_names = [
            "Unknown",      # 0
            "Letter A",     # 1
            "Letter B",     # 2
            "Letter C",     # 3
            "Hi",           # 4
            "Bye",          # 5
            "Nice",         # 6
            "Yes",          # 7
            "No",           # 8
            "Good Morning", # 9
        ]
    
    def classify_gesture(self, landmarks: Optional[np.ndarray], 
                        finger_count: int = 0,
                        hand_label: Optional[str] = None) -> Tuple[int, str]:
        """
        Classify hand gesture based on landmarks and finger count.
        
        Args:
            landmarks: Array of hand landmarks (21, 2) or None.
            finger_count: Number of extended fingers.
            hand_label: "Left" or "Right" hand label.
            
        Returns:
            Tuple of (gesture_id, gesture_name).
        """
        if landmarks is None:
            return 0, self.gesture_names[0]
        
        # Normalize landmarks relative to wrist (landmark 0)
        normalized_landmarks = landmarks - landmarks[0]
        
        # Get gesture based on finger count and landmark analysis
        gesture_id = self._analyze_gesture(normalized_landmarks, finger_count, hand_label)
        
        return gesture_id, self.gesture_names[gesture_id]
    
    def _get_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _analyze_gesture(self, landmarks: np.ndarray, finger_count: int, hand_label: str) -> int:
        """
        Analyze landmarks to determine gesture type using strict ASL and social gesture rules.
        
        Args:
            landmarks: Normalized landmark array (21, 2).
            finger_count: Number of extended fingers.
            hand_label: "Left" or "Right" hand label.
            
        Returns:
            Gesture ID.
        """
        # Landmarks
        wrist = landmarks[0]
        thumb_cmc = landmarks[1]
        thumb_mcp = landmarks[2]
        thumb_ip = landmarks[3]
        thumb_tip = landmarks[4]
        
        index_mcp = landmarks[5]
        index_pip = landmarks[6]
        index_dip = landmarks[7]
        index_tip = landmarks[8]
        
        middle_mcp = landmarks[9]
        middle_pip = landmarks[10]
        middle_dip = landmarks[11]
        middle_tip = landmarks[12]
        
        ring_mcp = landmarks[13]
        ring_pip = landmarks[14]
        ring_dip = landmarks[15]
        ring_tip = landmarks[16]
        
        pinky_mcp = landmarks[17]
        pinky_pip = landmarks[18]
        pinky_dip = landmarks[19]
        pinky_tip = landmarks[20]
        
        # Calculate hand scale (Wrist to Middle MCP) to make thresholds scale-invariant
        hand_scale = self._get_distance(wrist, middle_mcp)
        if hand_scale == 0: hand_scale = 1.0 # Prevent division by zero
        
        # --- Finger States ---
        # A finger is "Up" if tip is significantly above PIP (y coordinate is smaller)
        
        index_up = index_tip[1] < index_pip[1]
        middle_up = middle_tip[1] < middle_pip[1]
        ring_up = ring_tip[1] < ring_pip[1]
        pinky_up = pinky_tip[1] < pinky_pip[1]
        
        # A finger is "Down" (Curled) if tip is below PIP
        index_down = index_tip[1] > index_pip[1]
        middle_down = middle_tip[1] > middle_pip[1]
        ring_down = ring_tip[1] > ring_pip[1]
        pinky_down = pinky_tip[1] > pinky_pip[1]
        
        # Thumb States
        # Vertical: Tip is above IP (y coordinate is smaller)
        thumb_vertical = thumb_tip[1] < thumb_ip[1]
        
        # Tucked: Tip is close to the palm center (approx Middle MCP) or Pinky MCP
        dist_thumb_pinky_mcp = self._get_distance(thumb_tip, pinky_mcp)
        thumb_tucked = dist_thumb_pinky_mcp < 0.6 * hand_scale
        
        # Extended Out: Tip is far from Index MCP in X direction
        # Direction depends on hand label
        if hand_label == "Left":
            thumb_out = thumb_tip[0] < thumb_ip[0] - (0.1 * hand_scale)
        else: # Right
            thumb_out = thumb_tip[0] > thumb_ip[0] + (0.1 * hand_scale)

        # Debug print (uncomment to see live states)
        print(f"I:{int(index_up)} M:{int(middle_up)} R:{int(ring_up)} P:{int(pinky_up)} | T_Vert:{int(thumb_vertical)} T_Out:{int(thumb_out)}")

        # --- Gesture Logic ---
        # ORDERED BY PRIORITY to prevent conflicts
        
        # PRIORITY 1: Fist-like gestures (A, Yes, No)
        # Check these FIRST because they require ALL 4 fingers down
        # This prevents confusion with other gestures that might have some fingers "slightly" up
        if index_down and middle_down and ring_down and pinky_down:
            
            # Vertical Position of Thumb Tip relative to Index MCP
            # Y increases downwards (screen coordinates)
            
            # YES (Thumbs Up): Thumb tip is HIGH above Index MCP
            if thumb_tip[1] < index_mcp[1] - (0.2 * hand_scale):
                if thumb_vertical:
                    return 7 # Yes (Thumbs Up)
            
            # NO (Thumbs Down): Thumb tip is BELOW Index MCP
            elif thumb_tip[1] > index_mcp[1] + (0.15 * hand_scale):
                # Must be pointing down (Tip below IP)
                if thumb_tip[1] > thumb_ip[1]:
                    return 8 # No (Thumbs Down)
            
            # LETTER A: Thumb tip is LEVEL with Index MCP (middle zone)
            else:
                if thumb_vertical:
                    dist_thumb_index = self._get_distance(thumb_tip, index_mcp)
                    if dist_thumb_index < 0.5 * hand_scale:
                        # Must be in the middle zone (not too high, not too low)
                        if (thumb_tip[1] > index_mcp[1] - (0.1 * hand_scale) and 
                            thumb_tip[1] < index_mcp[1] + (0.1 * hand_scale)):
                            return 1 # Letter A
            
            # If none of the fist gestures matched, fall through to check others
        
        # PRIORITY 2: Four fingers up (Hi vs B)
        # Check BEFORE Nice/C to avoid confusion
        if index_up and middle_up and ring_up and pinky_up:
            # Must check ring and pinky are CLEARLY up to avoid confusion with Nice
            if ring_tip[1] < ring_pip[1] and pinky_tip[1] < pinky_pip[1]:
                if thumb_tucked:
                    return 2 # Letter B (Thumb tucked)
                elif thumb_vertical or thumb_out:
                    return 4 # Hi (Thumb extended/up)
                else:
                    return 4 # Hi (default when 4 fingers up)
        
        # PRIORITY 3: Nice (OK Sign)
        # Thumb and Index tips VERY CLOSE (touching)
        # Check BEFORE C to prioritize touching over curved
        dist_thumb_index_tip = self._get_distance(thumb_tip, index_tip)
        
        if dist_thumb_index_tip < 0.12 * hand_scale:  # Very strict - must be touching
            # Other fingers should be up
            if middle_up and ring_up and pinky_up:
                return 6 # Nice (OK Sign)
        
        # PRIORITY 4: Two fingers up (Bye - Peace Sign)
        # EXACTLY index and middle up, others down
        if index_up and middle_up and ring_down and pinky_down:
            # Make sure index and middle are CLEARLY up
            if (index_tip[1] < index_pip[1] - (0.05 * hand_scale) and 
                middle_tip[1] < middle_pip[1] - (0.05 * hand_scale)):
                return 5 # Bye
        
        # PRIORITY 5: One finger up (Good Morning - Pointing)
        # EXACTLY index up, others down
        # Very strict to avoid false positives
        if index_up and middle_down and ring_down and pinky_down:
            # Index must be CLEARLY pointing up (well above knuckle)
            if index_tip[1] < index_mcp[1] - (0.15 * hand_scale):
                # Make sure other fingers are CLEARLY down
                if (middle_tip[1] > middle_pip[1] and 
                    ring_tip[1] > ring_pip[1] and 
                    pinky_tip[1] > pinky_pip[1]):
                    return 9 # Good Morning
        
        # PRIORITY 6: Letter C (Curved fingers)
        # Check LAST among multi-finger gestures
        # Requires index and middle up, with specific gap
        if index_up and middle_up:
            # Must have a "C" gap (bigger than Nice, smaller than open hand)
            if 0.20 * hand_scale < dist_thumb_index_tip < 0.7 * hand_scale:
                # Thumb should NOT be tucked (otherwise it's B or Hi)
                if not thumb_tucked:
                    # At least index and middle should be up, but not all 4
                    # (to avoid confusion with Hi/B)
                    if not (ring_up and pinky_up and thumb_out):
                        return 3 # Letter C

        return 0  # Unknown


