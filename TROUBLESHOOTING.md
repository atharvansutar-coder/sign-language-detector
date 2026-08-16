# Troubleshooting Guide

## MediaPipe DLL Load Error on Windows

If you encounter the error:
```
ImportError: DLL load failed while importing _framework_bindings: A dynamic link library (DLL) initialization routine failed
```

Follow these steps in order:

### Solution 1: Install Visual C++ Redistributables (Recommended)

MediaPipe requires Microsoft Visual C++ Redistributables on Windows.

1. Download and install the **Microsoft Visual C++ Redistributable**:
   - Visit: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Or search for "Visual C++ Redistributable 2015-2022" on Microsoft's website
   - Install the **x64** version (for 64-bit Windows)

2. After installation, restart your computer

3. Try running the application again:
   ```bash
   python hand_gesture_detector.py
   ```

### Solution 2: Reinstall MediaPipe

If Solution 1 doesn't work, try reinstalling MediaPipe:

```bash
pip uninstall mediapipe
pip install mediapipe==0.10.9
```

### Solution 3: Use Python 3.11 Instead of 3.12

MediaPipe may have compatibility issues with Python 3.12. If possible, use Python 3.11:

1. Install Python 3.11 from https://www.python.org/downloads/
2. Create a new virtual environment with Python 3.11:
   ```bash
   python3.11 -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Solution 4: Clean Installation

Perform a clean reinstall of all dependencies:

```bash
# Uninstall problematic packages
pip uninstall mediapipe opencv-python numpy -y

# Clear pip cache
pip cache purge

# Reinstall
pip install -r requirements.txt
```

### Solution 5: Use Conda Instead of Pip

Conda sometimes handles Windows dependencies better:

```bash
# Install Miniconda or Anaconda
# Then create environment
conda create -n gesture-detector python=3.11
conda activate gesture-detector
conda install -c conda-forge opencv
pip install mediapipe numpy scikit-learn
```

### Solution 6: Check Python Architecture

Ensure you're using 64-bit Python (MediaPipe requires 64-bit):

```bash
python -c "import struct; print('64-bit' if struct.calcsize('P') * 8 == 64 else '32-bit')"
```

If it shows 32-bit, reinstall Python and select the 64-bit version.

### Solution 7: Alternative - Use Older MediaPipe Version

Try an older, more stable version of MediaPipe:

```bash
pip install mediapipe==0.10.8
```

## Common Issues

### Camera Not Found
- Ensure your webcam is connected and not used by another application
- On Windows, check Device Manager for camera issues
- Try changing the camera index in `hand_gesture_detector.py` (change `cap = cv2.VideoCapture(0)` to `cap = cv2.VideoCapture(1)`)

### Poor Detection
- Ensure good lighting
- Keep hand clearly visible in frame
- Hold gestures steady for 1-2 seconds

### Import Errors
- Verify all packages are installed: `pip list`
- Check if you're in the correct virtual environment
- Ensure you're using the correct Python interpreter

## Getting Help

If none of these solutions work:
1. Check MediaPipe GitHub issues: https://github.com/google/mediapipe/issues
2. Verify your Python version: `python --version`
3. Check installed packages: `pip list`
4. Share error messages and system information when seeking help


