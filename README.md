# Hand Gesture Detection Project

A real-time hand gesture detection system built with Python, OpenCV, and MediaPipe. This project can recognize various hand gestures including thumbs up, peace sign, pointing, and more.

## Features

- Real-time hand detection and tracking
- Multiple gesture recognition:
  - Fist (0 fingers)
  - Open Hand (5 fingers)
  - Thumbs Up/Down
  - Peace Sign (V sign)
  - OK Sign
  - Pointing
  - Three, Four, Five fingers
  - Shaka / Call Me (thumb + pinky)
  - Rock Sign (index + pinky)
  - ILY Sign (thumb + index + pinky)
- Supports both left and right hands
- Smooth gesture recognition with history-based filtering
- Webcam integration with live preview

## Requirements

- Python 3.7 or higher
- Webcam/Camera
- Windows, Linux, or macOS

## Installation

### Windows Users

**Important:** MediaPipe on Windows requires Microsoft Visual C++ Redistributables.

1. **Install Visual C++ Redistributables** (Required):
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install the x64 version
   - Restart your computer after installation

2. **Quick Install (Windows):**
   ```bash
   # Run the installer script
   install_windows.bat
   # Or use PowerShell
   .\install_windows.ps1
   ```

3. **Manual Install:**
   ```bash
   pip install -r requirements.txt
   ```

### Linux/macOS Users

```bash
pip install -r requirements.txt
```

### Troubleshooting

If you encounter DLL errors or import issues, see **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** for detailed solutions.

**Common Issues:**
- **DLL Load Error**: Install Visual C++ Redistributables (see above)
- **Python 3.12 Issues**: Consider using Python 3.11 instead
- **Import Errors**: Check TROUBLESHOOTING.md for step-by-step solutions

## Usage

Run the main detection script:
```bash
python hand_gesture_detector.py
```

### Controls

- Press `q` to quit the application
- Make sure your hand is visible in the camera frame
- Hold gestures steady for better recognition

## Project Structure

```
sign-language-detector/
├── hand_gesture_detector.py  # Main application script
├── hand_gesture_utils.py     # Hand detection utilities
├── gesture_classifier.py     # Gesture classification logic
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## How It Works

1. **Hand Detection**: Uses MediaPipe Hands to detect and track hands in real-time
1.  **Hand Detection**: Uses MediaPipe Hands to detect and track hands in real-time
2.  **Landmark Extraction**: Extracts 21 hand landmark points (wrist, finger joints, tips)
3.  **Finger Counting**: Analyzes landmark positions to count extended fingers
4.  **Gesture Classification**: Classifies gestures based on finger positions and patterns
5.  **Display**: Shows the detected gesture name and finger count on screen

## Supported Gestures
1.  **Letter A**: Fist with thumb vertical against the side of the index finger.
2.  **Letter B**: Open palm with 4 fingers up and thumb tucked across the palm.
3.  **Letter C**: Hand forming a C shape (curved fingers).
4.  **Hi**: Open palm with all 5 fingers extended (Thumb out).
5.  **Bye**: Peace Sign (Index and Middle fingers up).
6.  **Nice**: OK Sign (Thumb and Index tips touching).
7.  **Yes**: Thumbs Up (Thumb high above knuckle).
8.  **No**: Thumbs Down.
9.  **Good Morning**: Pointing Up (Index finger up).

## Customization

You can customize the detection by modifying parameters in `hand_gesture_detector.py`:

- `min_detection_confidence`: Minimum confidence for hand detection (default: 0.7)
- `min_tracking_confidence`: Minimum confidence for hand tracking (default: 0.7)
- `history_size`: Number of frames to use for gesture smoothing (default: 5)

## Troubleshooting

- **Camera not working**: Make sure your camera is connected and not being used by another application
- **Poor detection**: Ensure good lighting and keep your hand clearly visible in the frame
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

## Future Enhancements

- Add machine learning-based gesture recognition
- Support for sign language recognition
- Gesture recording and training
- Multiple hand gesture combinations
- Custom gesture definitions

## License

This project is open source and available for educational purposes.

## Credits

- Built with [MediaPipe](https://mediapipe.dev/) by Google
- Uses [OpenCV](https://opencv.org/) for computer vision
- Developed with Python 3


