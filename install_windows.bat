@echo off
echo ========================================
echo Hand Gesture Detector - Windows Installer
echo ========================================
echo.

echo Step 1: Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo.

echo Step 2: Checking if Visual C++ Redistributables are installed...
echo IMPORTANT: MediaPipe requires Visual C++ Redistributables
echo If installation fails, download and install from:
echo https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.
pause
echo.

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip
echo.

echo Step 4: Installing requirements...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Installation failed!
    echo.
    echo Try the following:
    echo 1. Install Visual C++ Redistributables (see TROUBLESHOOTING.md)
    echo 2. Use Python 3.11 instead of 3.12
    echo 3. Check TROUBLESHOOTING.md for more solutions
    echo.
    pause
    exit /b 1
)
echo.

echo Step 5: Verifying installation...
python -c "import cv2; import mediapipe; import numpy; print('All packages installed successfully!')"
if errorlevel 1 (
    echo.
    echo WARNING: Some packages may not be working correctly
    echo Check TROUBLESHOOTING.md for solutions
    echo.
) else (
    echo.
    echo ========================================
    echo Installation completed successfully!
    echo ========================================
    echo.
    echo You can now run: python hand_gesture_detector.py
    echo.
)
pause


