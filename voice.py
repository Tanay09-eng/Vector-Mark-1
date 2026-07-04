"""
Vector AI — Voice Layer
STT: OpenAI Whisper (runs locally, no API key needed)
TTS: Edge TTS (Microsoft, free, natural voices)

Install:
    pip install openai-whisper edge-tts sounddevice scipy
"""

import asyncio
import tempfile
import os
import io
from pathlib import Path

# ── STT: Whisper ──────────────────────────────────────────────────────────────
try:
    import whisper
    import sounddevice as sd
    import scipy.io.wavfile as wav
    import numpy as np
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[Voice] Whisper or sounddevice not found.")
    print("        Run: pip install openai-whisper sounddevice scipy")

# ── TTS: Edge TTS ─────────────────────────────────────────────────────────────
try:
    import edge_tts
    import pygame
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("[Voice] edge-tts or pygame not found.")
    print("        Run: pip install edge-tts pygame")


# ── Whisper STT ───────────────────────────────────────────────────────────────

class WhisperSTT:
    """
    Records audio from mic using sounddevice,
    transcribes with local Whisper model.
    Model sizes: tiny, base, small, medium, large
    'base' is recommended — good balance of speed and accuracy.
    """

    def __init__(self, model_size: str = "base"):
        self.model = None
        self.model_size = model_size
        self._load_model()

    def _load_model(self):
        if not WHISPER_AVAILABLE:
            return
        try:
            print(f"[STT] Loading Whisper '{self.model_size}' model...")
            self.model = whisper.load_model(self.model_size)
            print("[STT] Whisper ready.")
        except Exception as e:
            print(f"[STT] Failed to load Whisper: {e}")

    def is_ready(self) -> bool:
        return self.model is not None

    def listen(self, duration: int = 6, sample_rate: int = 16000) -> str:
        """
        Record `duration` seconds from the microphone,
        then transcribe with Whisper.
        Returns the transcribed text.
        """
        if not self.is_ready():
            return ""

        print(f"\n[Vector] Listening... ({duration}s)")
        try:
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            print("[Vector] Processing speech...")

            # Save to temp file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                # Convert float32 to int16 for wav
                audio_int16 = (recording * 32767).astype("int16")
                wav.write(temp_path, sample_rate, audio_int16)

            result = self.model.transcribe(temp_path, language="en")
            os.unlink(temp_path)

            text = result.get("text", "").strip()
            if text:
                print(f"[You] {text}")
            return text

        except Exception as e:
            print(f"[STT] Error: {e}")
            return ""

    def listen_until_silence(self, max_duration: int = 15,
                              silence_threshold: float = 0.01,
                              silence_duration: float = 1.5,
                              sample_rate: int = 16000) -> str:
        """
        Records until the user stops speaking (silence detected),
        up to max_duration seconds.
        """
        if not self.is_ready():
            return ""

        print("\n[Vector] Listening... (speak, then pause to send)")
        chunk_size = int(sample_rate * 0.1)  # 100ms chunks
        all_audio = []
        silent_chunks = 0
        silence_chunks_needed = int(silence_duration / 0.1)
        started_speaking = False
        max_chunks = int(max_duration / 0.1)
        chunks_recorded = 0

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1,
                                 dtype="float32", blocksize=chunk_size) as stream:
                while chunks_recorded < max_chunks:
                    chunk, _ = stream.read(chunk_size)
                    all_audio.append(chunk.copy())
                    volume = float(np.abs(chunk).mean())
                    chunks_recorded += 1

                    if volume > silence_threshold:
                        started_speaking = True
                        silent_chunks = 0
                    elif started_speaking:
                        silent_chunks += 1
                        if silent_chunks >= silence_chunks_needed:
                            break

            if not all_audio or not started_speaking:
                return ""

            audio = np.concatenate(all_audio, axis=0)
            print("[Vector] Processing...")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                audio_int16 = (audio * 32767).astype("int16")
                wav.write(temp_path, sample_rate, audio_int16)

            result = self.model.transcribe(temp_path, language="en")
            os.unlink(temp_path)
            text = result.get("text", "").strip()
            if text:
                print(f"[You] {text}")
            return text

        except Exception as e:
            print(f"[STT] Error: {e}")
            return ""


# ── Edge TTS ──────────────────────────────────────────────────────────────────

class EdgeTTS:
    """
    Microsoft Edge TTS — free, natural, no API key.
    Voice: en-GB-RyanNeural (British male — most J.A.R.V.I.S-like)
    Other options:
      en-US-GuyNeural       — American male
      en-US-ChristopherNeural — American male, deeper
      en-IN-PrabhatNeural   — Indian male
    """

    VOICE = "en-GB-RyanNeural"

    def __init__(self, voice: str = None):
        self.voice = voice or self.VOICE
        self._init_pygame()

    def _init_pygame(self):
        if not EDGE_TTS_AVAILABLE:
            return
        try:
            import pygame
            pygame.mixer.init()
        except Exception as e:
            print(f"[TTS] pygame init error: {e}")

    def is_ready(self) -> bool:
        return EDGE_TTS_AVAILABLE

    def speak(self, text: str):
        """Convert text to speech and play it."""
        if not EDGE_TTS_AVAILABLE:
            print(f"[TTS unavailable] {text}")
            return
        # Strip markdown
        import re
        clean = re.sub(r"[*_`#]", "", text)
        clean = re.sub(r"\n+", ". ", clean).strip()
        # Cap at 600 chars for TTS
        if len(clean) > 600:
            clean = clean[:597] + "..."
        try:
            asyncio.run(self._speak_async(clean))
        except RuntimeError:
            # If already in an event loop (shouldn't happen in CLI but just in case)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._speak_async(clean))

    async def _speak_async(self, text: str):
        import pygame
        communicate = edge_tts.Communicate(text, self.voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
        await communicate.save(temp_path)
        try:
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
        finally:
            pygame.mixer.music.stop()
            os.unlink(temp_path)

    def speak_background(self, text: str):
        """Speak without blocking (runs in background thread)."""
        import threading
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()


# ── Voice Session ─────────────────────────────────────────────────────────────

class VoiceSession:
    """
    Combines STT + TTS into a single interface.
    Used by the FastAPI backend for voice endpoints.
    """

    def __init__(self, whisper_model: str = "base", tts_voice: str = None):
        self.stt = WhisperSTT(model_size=whisper_model)
        self.tts = EdgeTTS(voice=tts_voice)

    def listen(self, smart: bool = True, duration: int = 6) -> str:
        """Listen and return transcribed text."""
        if smart:
            return self.stt.listen_until_silence()
        return self.stt.listen(duration=duration)

    def speak(self, text: str, background: bool = False):
        """Speak the response."""
        if background:
            self.tts.speak_background(text)
        else:
            self.tts.speak(text)

    def is_ready(self) -> bool:
        return self.stt.is_ready() and self.tts.is_ready()


# ── Singleton ─────────────────────────────────────────────────────────────────
voice_session = VoiceSession()
