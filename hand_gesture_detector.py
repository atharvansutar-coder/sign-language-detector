"""
Main hand gesture detection application.
"""
import sys
import time
import cv2
import numpy as np

# Handle import errors with helpful messages
try:
    from hand_gesture_utils import HandDetector
    from gesture_classifier import GestureClassifier
except ImportError as e:
    error_msg = str(e)
    if "DLL load failed" in error_msg or "_framework_bindings" in error_msg:
        print("=" * 60)
        print("ERROR: MediaPipe DLL Load Failed")
        print("=" * 60)
        print("\nThis is a common issue on Windows. Solutions:")
        print("\n1. Install Visual C++ Redistributables:")
        print("   Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("   Install the x64 version and restart your computer")
        print("\n2. Try using Python 3.11 instead of 3.12:")
        print("   MediaPipe may have compatibility issues with Python 3.12")
        print("\n3. Reinstall MediaPipe:")
        print("   pip uninstall mediapipe")
        print("   pip install mediapipe==0.10.9")
        print("\n4. Check TROUBLESHOOTING.md for more solutions")
        print("\n" + "=" * 60)
    else:
        print(f"Import Error: {error_msg}")
        print("\nPlease ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Main function to run hand gesture detection."""
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        print("Please ensure your camera is connected and not being used by another application")
        return
    
    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height
    
    # Initialize detector and classifier
    detector = HandDetector(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    classifier = GestureClassifier()
    
    # Initialize audio announcer
    try:
        from audio_announcer import AudioAnnouncer
        # backend="pyttsx3"  → offline, Samantha voice (default)
        # backend="edge-tts" → online, Microsoft Jenny neural voice (much better quality)
        announcer = AudioAnnouncer(enabled=True, backend="pyttsx3")
        audio_enabled = True
    except ImportError:
        print("Warning: Audio module not available. Install pyttsx3 for audio support:")
        print("  pip install pyttsx3")
        print("Continuing without audio support.")
        announcer = None
        audio_enabled = False
    except Exception as e:
        print(f"Warning: Audio initialization failed: {e}")
        print("Continuing without audio support.")
        announcer = None
        audio_enabled = False
    
    # Gesture history for smoothing
    gesture_history = []
    history_size = 10  # Increased for stability

    # ---- Debounce / hold state ----
    # A gesture must be held steadily for HOLD_MS milliseconds before the
    # announcement fires.  This prevents audio blurting while the hand is
    # still moving between two signs.
    HOLD_MS = 600                  # hold duration before announcing
    hold_gesture: str = ""         # which gesture is currently being timed
    hold_start_ms: float = 0.0     # when that hold started (time.time() * 1000)
    last_announced_gesture: str = ""  # last gesture we actually spoke aloud
    
    print("Hand Gesture Detection Started!")
    if audio_enabled:
        print("Audio announcements: ENABLED")
    else:
        print("Audio announcements: DISABLED")
    print("Press 'q' to quit")
    print("-" * 50)
    
    while True:
        # Read frame from camera
        success, img = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Flip image horizontally for mirror effect
        img = cv2.flip(img, 1)
        
        # Find hands
        img, results = detector.find_hands(img, draw=True)
        
        # Reset gesture history if no hands detected (for fresh detection)
        if not results.multi_hand_landmarks:
            gesture_history.clear()
            # Reset all debounce state so the next hand appearance starts fresh
            hold_gesture = ""
            hold_start_ms = 0.0
            last_announced_gesture = ""
        
        # Process each detected hand
        if results.multi_hand_landmarks:
            for hand_idx in range(len(results.multi_hand_landmarks)):
                # Get landmarks
                landmarks = detector.get_landmarks_array(img, hand_no=hand_idx)
                position_list = detector.find_position(img, hand_no=hand_idx, draw=False)
                
                if landmarks is not None and len(position_list) > 0:
                    # Get hand label first
                    hand_label = detector.get_hand_label(hand_no=hand_idx)
                    
                    # Count fingers
                    finger_count = detector.count_fingers(position_list, hand_label)
                    
                    # Classify gesture
                    gesture_id, gesture_name = classifier.classify_gesture(
                        landmarks, finger_count, hand_label
                    )
                    
                    # Add to history for smoothing
                    gesture_history.append(gesture_id)
                    if len(gesture_history) > history_size:
                        gesture_history.pop(0)
                    
                    # Get most common gesture from history
                    if gesture_history:
                        smoothed_gesture_id = max(set(gesture_history), 
                                                 key=gesture_history.count)
                        smoothed_gesture_name = classifier.gesture_names[smoothed_gesture_id]
                    else:
                        smoothed_gesture_name = gesture_name
                    
                    # ---- Debounce + announce logic ----
                    if audio_enabled and announcer:
                        current_key = f"{hand_label}_{smoothed_gesture_name}"
                        now_ms = time.time() * 1000

                        if smoothed_gesture_name == "Unknown":
                            # Unknown resets the hold timer but does NOT update
                            # last_announced_gesture — that way the next real sign
                            # can still fire even if we're returning to it.
                            hold_gesture = ""
                            hold_start_ms = 0.0
                        else:
                            if current_key != hold_gesture:
                                # Hand moved to a new gesture: restart hold timer
                                hold_gesture = current_key
                                hold_start_ms = now_ms
                                print(f"[Gesture] Holding: {current_key} …")
                            elif (now_ms - hold_start_ms) >= HOLD_MS:
                                # Held long enough — announce if not already spoken
                                if current_key != last_announced_gesture:
                                    print(f"[Gesture] Announcing: {smoothed_gesture_name}")
                                    announcer.announce(smoothed_gesture_name)
                                    last_announced_gesture = current_key
                    
                    # Get hand bounding box for text placement
                    if position_list:
                        x_coords = [pos[1] for pos in position_list]
                        y_coords = [pos[2] for pos in position_list]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        # Draw background rectangle for text
                        cv2.rectangle(img, (x_min - 10, y_min - 60), 
                                    (x_max + 10, y_min - 10), 
                                    (0, 0, 0), cv2.FILLED)
                        
                        # Display gesture information
                        info_text = f"{smoothed_gesture_name} ({finger_count})"
                        
                        if hand_label:
                            info_text = f"{hand_label}: {info_text}"
                        
                        cv2.putText(img, info_text, (x_min, y_min - 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display FPS
        cv2.putText(img, "Press 'q' to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the image
        cv2.imshow("Hand Gesture Detection", img)
        
        # Break on 'q' key press
        # Increased delay to 50ms (approx 20fps) to slow down execution
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    
    # Cleanup
    if audio_enabled and announcer:
        announcer.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("\nHand Gesture Detection Stopped.")


if __name__ == "__main__":
    main()

