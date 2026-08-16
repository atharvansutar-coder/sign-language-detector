# Hand Gesture Detector - Windows PowerShell Installer

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Hand Gesture Detector - Windows Installer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python
Write-Host "Step 1: Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: Warning about Visual C++ Redistributables
Write-Host "Step 2: Important Notice" -ForegroundColor Yellow
Write-Host "MediaPipe requires Microsoft Visual C++ Redistributables" -ForegroundColor Yellow
Write-Host "If installation fails, download and install from:" -ForegroundColor Yellow
Write-Host "https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Cyan
Write-Host ""
$continue = Read-Host "Press Enter to continue or Ctrl+C to cancel"

# Step 3: Upgrade pip
Write-Host ""
Write-Host "Step 3: Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host ""

# Step 4: Install requirements
Write-Host "Step 4: Installing requirements..." -ForegroundColor Yellow
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Installation failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Try the following:" -ForegroundColor Yellow
    Write-Host "1. Install Visual C++ Redistributables (see TROUBLESHOOTING.md)" -ForegroundColor Yellow
    Write-Host "2. Use Python 3.11 instead of 3.12" -ForegroundColor Yellow
    Write-Host "3. Check TROUBLESHOOTING.md for more solutions" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
Write-Host ""

# Step 5: Verify installation
Write-Host "Step 5: Verifying installation..." -ForegroundColor Yellow
python -c "import cv2; import mediapipe; import numpy; print('All packages installed successfully!')"
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Installation completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run: python hand_gesture_detector.py" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "WARNING: Some packages may not be working correctly" -ForegroundColor Red
    Write-Host "Check TROUBLESHOOTING.md for solutions" -ForegroundColor Yellow
    Write-Host ""
}


