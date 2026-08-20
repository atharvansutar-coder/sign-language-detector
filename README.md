# SIGNBRIDGE

**A Real-Time Indian Sign Language Accessibility & Emergency Communication Assistant**

SIGNBRIDGE is an advanced, offline-first application that bridges the communication gap using Computer Vision, Natural Language Processing, and Speech Recognition. It tracks hand gestures in real-time to recognize Indian Sign Language (ISL), forms continuous sentences, and translates them to speech. It also features Speech-to-Text for two-way communication.

## Priority Features

- **Continuous ISL Recognition**: Recognizes a practical vocabulary and combines consecutive signs into words/sentences automatically.
- **Emergency Mode 🔴**: Specialized vocabulary for critical situations (HELP, EMERGENCY, DOCTOR, PAIN). Triggers a highly visible visual alert.
- **Hospital Mode 🏥**: Tailored vocabulary for patient-caregiver communication (medicine, water, bathroom, family).
- **Two-Way Communication**: 
  - ISL → Text → Speech (using offline Pyttsx3).
  - Speech → Text (using SpeechRecognition) to display the hearing person's speech on screen.
- **Offline-First**: Core recognition runs entirely locally.
- **Confidence System**: Filters out low-confidence signs and prompts the user to repeat.
- **Modern UI**: Clean overlay displaying current mode, FPS, recognized sentence, conversation log, and confidence metrics.

## Requirements

- Python 3.7+
- Webcam/Camera
- Windows, Linux, or macOS

## Installation

### Windows Users
**Important:** MediaPipe on Windows requires Microsoft Visual C++ Redistributables.

1. **Install Visual C++ Redistributables** (Required):
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install the x64 version

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For offline speech recognition, you may also want to install `pocketsphinx`.*

## Usage

Run the main SIGNBRIDGE application:

```bash
python hand_gesture_detector.py
```

### Keyboard Controls

- `M` : Cycle Modes (General -> Emergency -> Hospital)
- `S` : Start Speech Recognition (Voice to Text)
- `U` : Undo the last recognized sign in the sentence
- `C` : Clear the current sentence and any active emergency alerts
- `Q` : Quit the application

## How It Works

1. **Hand Detection**: Uses MediaPipe to detect and track hand landmarks in real-time.
2. **Gesture Classification**: Analyzes finger positions and relative landmark distances to classify the gesture against a predefined vocabulary (`vocabulary.py`).
3. **Temporal Smoothing**: `SignRecognizer` buffers history to smooth out misclassifications and waits for a steady hold before committing a sign to the sentence.
4. **Mode Filtering**: Vocabulary is restricted based on the active mode to improve accuracy in specific contexts.
5. **Two-Way Comm**: `SpeechInput` listens to the microphone and appends text to the conversation log, while `AudioAnnouncer` reads out the built ISL sentence.

## Future Enhancements
- Expand vocabulary with more ISL signs.
- Integrate facial expression and body pose landmarks.
- Upgrade the rule-based classifier to an LSTM or Transformer model for temporal sequence recognition.

## License
This project is open source and available for educational purposes.
