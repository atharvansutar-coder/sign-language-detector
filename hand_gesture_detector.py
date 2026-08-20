"""
SIGNBRIDGE - Real-Time Indian Sign Language Accessibility & Emergency Communication Assistant
"""
import sys
import time
import cv2
import numpy as np

try:
    from sign_recognizer import SignRecognizer
    from vocabulary import AppMode
    from speech_input import SpeechInput
    from audio_announcer import AudioAnnouncer
except ImportError as e:
    error_msg = str(e)
    if "DLL load failed" in error_msg or "_framework_bindings" in error_msg:
        print("=" * 60)
        print("ERROR: MediaPipe DLL Load Failed")
        print("=" * 60)
        print("Install Visual C++ Redistributables or check Python version.")
    else:
        print(f"Import Error: {error_msg}")
        print("\nPlease ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
    sys.exit(1)


def draw_modern_ui(img, result, recognizer, speech, w, h):
    """Draws a modern overlay UI on the camera feed."""
    # Top Bar - Status & Mode
    cv2.rectangle(img, (0, 0), (w, 50), (25, 25, 25), cv2.FILLED)
    
    # Mode indicator
    mode_color = recognizer.mode_state.mode_color_bgr
    mode_label = recognizer.mode_state.mode_label.upper() + " MODE"
    cv2.putText(img, mode_label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
    
    # Speech Status
    speech_status = "Listening..." if speech.is_listening else "Press 'S' to Speak"
    speech_color = (0, 200, 0) if speech.is_listening else (150, 150, 150)
    cv2.putText(img, speech_status, (w - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, speech_color, 2)
    
    # FPS
    fps = recognizer.stats.fps
    cv2.putText(img, f"FPS: {fps:.1f}", (w // 2 - 50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Bottom Area - Sentence & Confidence
    bottom_y = h - 120
    cv2.rectangle(img, (0, bottom_y), (w, h), (25, 25, 25), cv2.FILLED)
    
    # Built Sentence
    sentence = recognizer.sentence.text
    if not sentence:
        sentence = "..."
    cv2.putText(img, sentence, (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Current Sign & Confidence bar
    if result.hand_present:
        # Sign Name
        cv2.putText(img, f"Current: {result.display}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Confidence Bar
        conf_w = int(200 * result.smoothed_confidence)
        bar_color = (0, 255, 0) if not result.is_low_confidence else (0, 100, 255)
        cv2.rectangle(img, (250, h - 35), (250 + 200, h - 15), (50, 50, 50), cv2.FILLED)
        cv2.rectangle(img, (250, h - 35), (250 + conf_w, h - 15), bar_color, cv2.FILLED)
        cv2.putText(img, f"{int(result.smoothed_confidence * 100)}%", (460, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 1)

        # Low confidence message
        if result.message:
            cv2.putText(img, result.message, (550, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)
    
    # Hand Bounding Box text (Optional, if we want text near hand)
    if result.hand_present and not result.is_low_confidence and result.display != "—":
        # we can just rely on the bottom bar for cleaner UI
        pass

    # Conversation Log (Two-way communication display)
    conv_y = 90
    cv2.putText(img, "Conversation Log:", (20, conv_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Display last 5 messages
    log = recognizer.conversation[-5:]
    for i, msg in enumerate(log):
        role = msg["role"]
        text = msg["text"]
        if role == "sentence":
            color = (0, 255, 0)
            prefix = "Sentence: "
        elif role == "sign":
            color = (255, 200, 0)
            prefix = "Sign: "
        else:
            color = (0, 200, 255)
            prefix = "Voice: "
        cv2.putText(img, f"{prefix}{text}", (20, conv_y + 30 + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Emergency Overlay
    if recognizer.mode_state.active_emergency:
        # Flashing effect
        if int(time.time() * 4) % 2 == 0:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), cv2.FILLED)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            
            msg = recognizer.mode_state.emergency_message
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.putText(img, msg, ((w - text_size[0]) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Controls hint
    controls = "'M': Mode | 'S': Speak | 'U': Undo | 'C': Clear | 'Q': Quit"
    cv2.putText(img, controls, (w - 520, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return img


def main():
    """Main application loop for SIGNBRIDGE."""
    print("Starting SIGNBRIDGE...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Use standard 720p resolution
    w, h = 1280, 720
    cap.set(3, w)
    cap.set(4, h)
    
    # Initialize core modules
    recognizer = SignRecognizer(history_size=8, hold_ms=600, confidence_threshold=0.60)
    
    # Audio announcement for signs
    try:
        announcer = AudioAnnouncer(enabled=True, backend="macos")
        print("Audio output initialized (macOS native say).")
        announcer.announce("SIGNBRIDGE ready")
    except Exception as e:
        print(f"Warning: Audio output failed: {e}")
        announcer = None

    # Speech input (Two-way communication)
    def on_speech_result(text):
        if text:
            print(f"Speech Recognized: {text}")
            recognizer.add_speech_message(text)

    speech = SpeechInput(on_result=on_speech_result)
    print(f"Speech input initialized. Backend: {speech.backend_label}")
    
    last_audio_commit = 0

    print("\n" + "="*40)
    print("SIGNBRIDGE is ready!")
    print("Controls:")
    print("  M - Cycle Modes (General -> Emergency -> Hospital)")
    print("  S - Start Speech Recognition (Voice to Text)")
    print("  U - Undo last sign")
    print("  C - Clear current sentence")
    print("  Q - Quit")
    print("="*40 + "\n")

    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Mirror image
        img = cv2.flip(img, 1)
        
        # Process frame
        img, result = recognizer.process_frame(img)
        
        # Speak every newly confirmed sign. When the latest sign completes
        # a multi-gesture sentence, speak the full sentence instead of only
        # the final sign. This gives audio output for all added signs while
        # keeping sentence output natural.
        if announcer:
            for text, delay in recognizer.drain_audio_events():
                announcer.announce(text, delay_after=delay)
                
        # Draw UI
        img = draw_modern_ui(img, result, recognizer, speech, w, h)
        
        cv2.imshow("SIGNBRIDGE", img)
        
        # Handle keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            new_mode = recognizer.cycle_mode()
            print(f"Mode changed to: {new_mode.name}")
        elif key == ord('s'):
            # Never call input() from the OpenCV loop. On macOS this can make
            # the camera window appear frozen. If speech recognition is not
            # installed/available, simply keep the camera running.
            if speech.is_available:
                print("Listening for speech...")
                speech.listen_async()
            else:
                print("Speech recognition is not available; camera remains active.")
        elif key == ord('u'):
            recognizer.undo_last_sign()
        elif key == ord('c'):
            recognizer.clear_sentence()
            recognizer.mode_state.active_emergency = False

    # Cleanup
    if announcer:
        announcer.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("SIGNBRIDGE stopped.")

if __name__ == "__main__":
    main()
