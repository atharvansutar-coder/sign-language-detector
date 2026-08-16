"""
Audio announcement module for gesture detection.

TWO BACKENDS — pick one via the `backend` constructor argument:

  "pyttsx3"  (default) — offline, uses the best female system voice available.
              Samantha (macOS), Microsoft Zira (Windows), espeak female (Linux).
              Sounds decent; zero latency; no internet needed.

  "edge-tts" — online, uses Microsoft's free neural TTS service.
               Voice: en-US-JennyNeural (warm, clear, natural female).
               Sounds significantly better than any offline SAPI/NSSpeech voice.
               Requires internet. First announcement has ~300-500 ms network latency;
               subsequent ones cache the audio in memory for near-instant playback.

Queue / worker-thread / debounce logic is unchanged.
"""
import io
import os
import platform
import queue
import subprocess
import threading
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Voice-quality constants — tweak these to taste
# ---------------------------------------------------------------------------
PYTTSX3_RATE   = 145    # words per minute  (130-160 is comfortable)
PYTTSX3_VOLUME = 1.0    # 0.0 – 1.0

EDGE_VOICE     = "en-US-JennyNeural"   # or "en-US-AriaNeural", "en-GB-SoniaNeural"
EDGE_RATE      = "+0%"                 # e.g. "-10%" to slow down, "+10%" to speed up
EDGE_VOLUME    = "+0%"                 # e.g. "+20%" to boost

# ---------------------------------------------------------------------------
# Preferred English female voices per platform (pyttsx3 backend)
# Listed best-first; first match wins.
# ---------------------------------------------------------------------------
_FEMALE_PREFS = {
    "Darwin": [          # macOS
        "com.apple.voice.Tara",               # Tara  — best quality (premium)
        "com.apple.voice.compact.en-US.Samantha",  # Samantha — crisp standard
        "com.apple.voice.compact.en-AU.Karen",     # Karen — Australian
        "com.apple.voice.compact.en-IE.Moira",     # Moira — Irish
        "com.apple.voice.compact.en-ZA.Tessa",     # Tessa — South African
    ],
    "Windows": [
        "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0",
        "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-GB_HAZEL_11.0",
    ],
}


def _pick_female_voice(engine) -> Optional[str]:
    """
    Return the voice ID of the best available English female voice,
    or None if nothing matches (caller falls back to voices[0]).
    """
    voices = engine.getProperty("voices") or []

    # 1. Try exact-ID matches from our preference list
    sys_name = platform.system()
    prefs = _FEMALE_PREFS.get(sys_name, [])
    voice_by_id = {v.id: v for v in voices}
    for preferred_id in prefs:
        if preferred_id in voice_by_id:
            return preferred_id

    # 2. Fall back: any voice whose gender attribute says Female
    for v in voices:
        gender = getattr(v, "gender", "") or ""
        if "female" in gender.lower():
            # prefer English voices
            name_lower = v.name.lower()
            if any(lang in name_lower for lang in ("english", "en-us", "en-gb", "en-au")):
                return v.id

    # 3. Broader: any Female voice regardless of language
    for v in voices:
        gender = getattr(v, "gender", "") or ""
        if "female" in gender.lower():
            return v.id

    # 4. Name heuristic for pyttsx3 builds that don't expose gender
    female_names = ("samantha", "zira", "hazel", "karen", "moira", "tessa",
                    "tara", "victoria", "kate", "emma", "jenny", "aria",
                    "susan", "linda", "allison")
    for v in voices:
        if any(n in v.name.lower() for n in female_names):
            return v.id

    return None   # give up — caller will use voices[0]


# ===========================================================================
# pyttsx3 backend worker
# ===========================================================================

def _pyttsx3_worker(speech_queue: queue.Queue, rate: int, volume: float) -> None:
    """Persistent worker: one engine, reused for every utterance."""
    engine = None
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate",   rate)
        engine.setProperty("volume", volume)

        chosen_id = _pick_female_voice(engine)
        if chosen_id:
            engine.setProperty("voice", chosen_id)
            voices = engine.getProperty("voices") or []
            name = next((v.name for v in voices if v.id == chosen_id), chosen_id)
            print(f"[Audio/pyttsx3] Voice: {name}")
        else:
            voices = engine.getProperty("voices") or []
            if voices:
                engine.setProperty("voice", voices[0].id)
                print(f"[Audio/pyttsx3] No female voice found — using: {voices[0].name}")
            else:
                print("[Audio/pyttsx3] No voices available.")

        print(f"[Audio/pyttsx3] Ready  rate={rate} wpm  volume={volume}")

        while True:
            try:
                text = speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:          # sentinel → shut down
                speech_queue.task_done()
                break

            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                print(f"[Audio/pyttsx3] Speech error: {exc}")
            finally:
                speech_queue.task_done()

    except ImportError:
        print("[Audio/pyttsx3] pyttsx3 not installed — pip install pyttsx3")
    except Exception as exc:
        print(f"[Audio/pyttsx3] Engine init failed: {exc}")
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        # drain the queue so callers don't block
        while True:
            try:
                speech_queue.get_nowait()
                speech_queue.task_done()
            except queue.Empty:
                break


# ===========================================================================
# edge-tts backend worker
# ===========================================================================

def _edge_worker(speech_queue: queue.Queue, voice: str,
                 rate: str, volume: str) -> None:
    """
    Persistent worker using edge-tts (Microsoft neural voices).
    Audio is rendered to PCM via ffmpeg/afplay/aplay depending on platform.
    """
    try:
        import edge_tts          # noqa: F401 – verify import
        import asyncio
    except ImportError:
        print("[Audio/edge-tts] edge-tts not installed — pip install edge-tts")
        return

    _cache: dict[str, bytes] = {}   # text → MP3 bytes (in-memory cache)

    async def _speak(text: str) -> None:
        if text not in _cache:
            communicate = edge_tts.Communicate(text, voice=voice,
                                               rate=rate, volume=volume)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            _cache[text] = buf.getvalue()

        mp3_bytes = _cache[text]

        # Play via system command: afplay (mac), ffplay (linux/win with ffmpeg)
        sys_name = platform.system()
        if sys_name == "Darwin":
            proc = subprocess.Popen(
                ["afplay", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.communicate(input=mp3_bytes)
        else:
            # Requires ffmpeg on PATH
            proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.communicate(input=mp3_bytes)

    print(f"[Audio/edge-tts] Ready  voice={voice}  rate={rate}  volume={volume}")

    loop = asyncio.new_event_loop()
    while True:
        try:
            text = speech_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if text is None:
            speech_queue.task_done()
            break

        try:
            loop.run_until_complete(_speak(text))
        except Exception as exc:
            print(f"[Audio/edge-tts] Error: {exc}")
        finally:
            speech_queue.task_done()

    loop.close()


# ===========================================================================
# Public class
# ===========================================================================

class AudioAnnouncer:
    """
    Thread-safe TTS announcer with selectable backend.

    Args:
        enabled: Master on/off switch.
        backend: "pyttsx3" (offline) or "edge-tts" (online neural).
        rate:    Speaking rate in WPM (pyttsx3 only).
        volume:  Volume 0.0–1.0 (pyttsx3 only).
    """

    def __init__(
        self,
        enabled: bool = True,
        backend: str = "pyttsx3",    # ← change to "edge-tts" for neural voice
        rate: int    = PYTTSX3_RATE,
        volume: float = PYTTSX3_VOLUME,
    ):
        self.enabled = enabled
        self.backend = backend
        self.speech_queue: queue.Queue = queue.Queue(maxsize=10)
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False

        if self.enabled:
            self._start_worker(rate, volume)

    # ------------------------------------------------------------------

    def _start_worker(self, rate: int, volume: float) -> None:
        if self.backend == "edge-tts":
            target = lambda: _edge_worker(          # noqa: E731
                self.speech_queue, EDGE_VOICE, EDGE_RATE, EDGE_VOLUME
            )
        else:
            target = lambda: _pyttsx3_worker(       # noqa: E731
                self.speech_queue, rate, volume
            )

        self.is_running = True
        self.worker_thread = threading.Thread(
            target=target, name=f"tts-{self.backend}", daemon=True
        )
        self.worker_thread.start()
        print(f"[Audio] Backend: {self.backend}  — worker thread started.")

    # ------------------------------------------------------------------

    def announce(self, text: str) -> None:
        """
        Queue text for spoken announcement.
        Drops stale items if the queue is backing up (keeps latest gesture).
        """
        if not self.enabled or not text:
            return

        if self.speech_queue.qsize() > 2:
            try:
                while not self.speech_queue.empty():
                    self.speech_queue.get_nowait()
                    self.speech_queue.task_done()
            except Exception:
                pass

        try:
            self.speech_queue.put_nowait(text)
            print(f"[Audio] Queued: '{text}'")
        except queue.Full:
            print(f"[Audio] Queue full — skipped: '{text}'")

    def stop(self) -> None:
        """Graceful shutdown — waits up to 3 s for the worker to finish."""
        self.is_running = False
        try:
            self.speech_queue.put_nowait(None)   # sentinel
        except queue.Full:
            pass
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)
        print("[Audio] Announcer stopped.")



