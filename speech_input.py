"""
Offline-first speech-to-text input for two-way communication.

Uses SpeechRecognition with Sphinx (offline) when available;
falls back to a simple typed input prompt.
"""
import threading
from typing import Callable, Optional


class SpeechInput:
    """Capture speech and convert to text (offline when PocketSphinx is installed)."""

    def __init__(self, on_result: Optional[Callable[[str], None]] = None):
        self._on_result = on_result
        self._listening = False
        self._available = False
        self._backend = "none"
        self._recognizer = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._sr = sr
            self._available = True
            # Check for offline Sphinx
            try:
                import pocketsphinx  # noqa: F401
                self._backend = "sphinx"
            except ImportError:
                self._backend = "google"
        except ImportError:
            self._available = False
            self._backend = "none"

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def backend_label(self) -> str:
        labels = {
            "sphinx": "Offline (PocketSphinx)",
            "google": "Online (Google — requires internet)",
            "none": "Unavailable — pip install SpeechRecognition",
        }
        return labels.get(self._backend, self._backend)

    @property
    def is_listening(self) -> bool:
        return self._listening

    def listen_async(self, timeout: float = 5.0, phrase_limit: float = 6.0) -> None:
        """Start listening in a background thread."""
        if self._listening or not self._available:
            return
        thread = threading.Thread(
            target=self._listen_worker,
            args=(timeout, phrase_limit),
            daemon=True,
        )
        thread.start()

    def _listen_worker(self, timeout: float, phrase_limit: float) -> None:
        self._listening = True
        text = ""
        try:
            with self._sr.Microphone() as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.4)
                audio = self._recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_limit
                )
            if self._backend == "sphinx":
                text = self._recognizer.recognize_sphinx(audio)
            else:
                text = self._recognizer.recognize_google(audio)
        except Exception as exc:
            print(f"[SpeechInput] {exc}")
            text = ""
        finally:
            self._listening = False
            if text and self._on_result:
                self._on_result(text.strip())

    @staticmethod
    def prompt_typed_input() -> str:
        """Fallback: read text from terminal (non-blocking alternative: key 't')."""
        try:
            return input("\n[SIGNBRIDGE] Type message: ").strip()
        except EOFError:
            return ""
