# Quick Fix for DLL Error

## If you see: "DLL load failed while importing _framework_bindings"

### Fastest Solution (5 minutes):

1. **Download and Install Visual C++ Redistributables:**
   - Click this link: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Run the installer
   - Restart your computer

2. **Try running again:**
   ```bash
   python hand_gesture_detector.py
   ```

### If that doesn't work:

**Option A: Use Python 3.11**
```bash
# Install Python 3.11 from python.org
# Then create new virtual environment
python3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Option B: Reinstall MediaPipe**
```bash
pip uninstall mediapipe
pip install mediapipe==0.10.9
```

**Option C: Clean Reinstall**
```bash
pip uninstall mediapipe opencv-python numpy -y
pip cache purge
pip install -r requirements.txt
```

### Still having issues?

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.


