"""Thread-safe audio output for SIGNBRIDGE.

On macOS the native `say` command is used so audio does not depend on
pyttsx3/PortAudio and never blocks the OpenCV camera loop.
"""
import platform
import queue
import subprocess
import threading
import time
from typing import Optional, Tuple

DEFAULT_RATE = 150
DEFAULT_DELAY = 1.0

class AudioAnnouncer:
    def __init__(self, enabled: bool = True, backend: str = "macos", rate: int = DEFAULT_RATE, volume: float = 1.0):
        self.enabled = enabled
        self.backend = backend
        self.rate = rate
        self.volume = volume
        self.speech_queue: queue.Queue[Optional[Tuple[str, float]]] = queue.Queue(maxsize=50)
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False
        if enabled:
            self._start_worker()

    def _start_worker(self):
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker, name="signbridge-audio", daemon=True)
        self.worker_thread.start()
        print(f"[Audio] Backend: {self.backend} - worker thread started.")

    def _speak(self, text: str):
        if platform.system() == "Darwin" or self.backend == "macos":
            subprocess.run(["say", "-r", str(self.rate), text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as exc:
                print(f"[Audio] Speech error: {exc}")

    def _worker(self):
        while True:
            item = self.speech_queue.get()
            if item is None:
                self.speech_queue.task_done()
                break
            text, delay = item
            try:
                self._speak(text)
                if delay > 0:
                    time.sleep(delay)
                print(f"[Audio] Spoken: '{text}'")
            finally:
                self.speech_queue.task_done()

    def announce(self, text: str, delay_after: float = DEFAULT_DELAY):
        if not self.enabled or not text or not self.is_running:
            return
        try:
            self.speech_queue.put_nowait((str(text), max(0.0, float(delay_after))))
            print(f"[Audio] Queued: '{text}'")
        except queue.Full:
            print(f"[Audio] Queue full - skipped: '{text}'")

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        try:
            self.speech_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        print("[Audio] Announcer stopped.")
