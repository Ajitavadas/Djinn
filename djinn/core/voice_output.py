"""Voice output: Kokoro ONNX (primary) + Edge TTS (fallback).

Streams audio sentence-by-sentence for minimum perceived latency.
Kokoro runs locally on CPU (~90MB), Edge TTS uses Microsoft's cloud.
"""
import asyncio
import io
import logging
import os
import re
import tempfile
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

log = logging.getLogger("djinn.voice_output")


# ---------------------------------------------------------------------------
# Kokoro ONNX TTS
# ---------------------------------------------------------------------------

class KokoroTTS:
    """Local TTS using Kokoro-82M ONNX model (no PyTorch needed)."""

    def __init__(self, voice: str = "af_heart"):
        self.voice = voice
        self._kokoro = None
        self._available = False
        self._model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "models",
        )

    @property
    def available(self) -> bool:
        return self._available

    def load(self) -> None:
        """Load the Kokoro ONNX model and voices."""
        try:
            from kokoro_onnx import Kokoro

            model_path = os.path.join(self._model_dir, "kokoro-v1.0.int8.onnx")
            voices_path = os.path.join(self._model_dir, "voices-v1.0.bin")

            if not os.path.exists(model_path) or not os.path.exists(voices_path):
                log.warning(
                    "Kokoro model files not found at %s. "
                    "Download kokoro-v1.0.int8.onnx and voices-v1.0.bin from "
                    "https://github.com/thewh1teagle/kokoro-onnx/releases",
                    self._model_dir,
                )
                self._available = False
                return

            t0 = time.perf_counter()
            self._kokoro = Kokoro(model_path, voices_path)
            dt = time.perf_counter() - t0
            self._available = True

            # Kokoro pulls in phonemizer, which resets its own log level on
            # import and then warns "words count mismatch" on almost every
            # sentence. Silencing it in setup_logging() is too early — it gets
            # overwritten — so it has to happen here, after the import.
            logging.getLogger("phonemizer").setLevel(logging.ERROR)

            log.info("Kokoro TTS loaded in %.1fs (voice=%s)", dt, self.voice)

        except ImportError:
            log.warning("kokoro-onnx not installed, Kokoro TTS unavailable")
            self._available = False
        except Exception as e:
            log.error("Failed to load Kokoro: %s", e)
            self._available = False

    def synthesize(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """Synthesize text to audio.

        Args:
            text: Text to speak.

        Returns:
            Tuple of (audio_samples, sample_rate) or None on failure.
        """
        if not self._available or not self._kokoro:
            return None

        try:
            t0 = time.perf_counter()
            samples, sample_rate = self._kokoro.create(
                text,
                voice=self.voice,
                speed=1.0,
                lang="en-us",
            )
            dt = time.perf_counter() - t0
            log.debug("Kokoro synthesized %.1fs audio in %.2fs", len(samples) / sample_rate, dt)
            return samples, sample_rate
        except Exception as e:
            log.error("Kokoro synthesis failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# Edge TTS (cloud fallback)
# ---------------------------------------------------------------------------

class EdgeTTS:
    """Microsoft Edge TTS — cloud-based, high quality, requires internet."""

    def __init__(self, voice: str = "en-IN-NeerjaNeural"):
        self.voice = voice

    async def synthesize(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """Synthesize text to audio using Edge TTS.

        Args:
            text: Text to speak.

        Returns:
            Tuple of (audio_samples, sample_rate) or None on failure.
        """
        try:
            import edge_tts

            t0 = time.perf_counter()
            communicate = edge_tts.Communicate(text, voice=self.voice)

            # Collect audio bytes
            audio_bytes = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes.write(chunk["data"])

            audio_bytes.seek(0)

            if audio_bytes.getbuffer().nbytes == 0:
                log.warning("Edge TTS returned empty audio")
                return None

            # Edge TTS returns MP3, convert to numpy via soundfile
            samples, sample_rate = sf.read(audio_bytes, dtype="float32")

            dt = time.perf_counter() - t0
            log.debug("Edge TTS synthesized %.1fs audio in %.2fs", len(samples) / sample_rate, dt)
            return samples, sample_rate

        except Exception as e:
            log.error("Edge TTS failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# Unified Voice Output
# ---------------------------------------------------------------------------

# Regex to split text into sentences
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


class VoiceOutput:
    """Unified TTS with Kokoro primary and Edge TTS fallback.

    Supports sentence-level streaming: as the LLM generates text,
    each sentence is synthesized and played independently, so the
    user hears audio while the rest of the response is still generating.
    """

    def __init__(
        self,
        primary: str = "kokoro",
        kokoro_voice: str = "af_heart",
        edge_voice: str = "en-IN-NeerjaNeural",
        edge_fallback: bool = True,
    ):
        self.primary = primary
        self.edge_fallback = edge_fallback
        self.kokoro = KokoroTTS(voice=kokoro_voice)
        self.edge = EdgeTTS(voice=edge_voice)

        # Playback state
        self._playing = False
        self._interrupt = threading.Event()

    def load(self) -> None:
        """Load TTS models."""
        if self.primary == "kokoro":
            self.kokoro.load()
            if not self.kokoro.available:
                log.warning("Kokoro unavailable, falling back to Edge TTS")
                self.primary = "edge"
        log.info("Voice output ready (primary=%s, edge_fallback=%s)", self.primary, self.edge_fallback)

    def interrupt(self) -> None:
        """Stop current playback immediately (barge-in)."""
        self._interrupt.set()

    async def speak(self, text: str) -> None:
        """Speak text. Uses primary TTS, falls back to Edge if needed.

        Args:
            text: Full text to speak.
        """
        self._interrupt.clear()
        self._playing = True

        try:
            # Split into sentences for streaming feel
            sentences = SENTENCE_SPLIT.split(text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                return

            for sentence in sentences:
                if self._interrupt.is_set():
                    log.debug("Speech interrupted")
                    break

                result = None

                # Try primary TTS
                if self.primary == "kokoro" and self.kokoro.available:
                    result = self.kokoro.synthesize(sentence)

                # Fallback to Edge TTS
                if result is None and self.edge_fallback:
                    result = await self.edge.synthesize(sentence)

                if result is None:
                    log.warning("Both TTS engines failed for: %s", sentence[:50])
                    continue

                samples, sample_rate = result
                await self._play_audio(samples, sample_rate)

        finally:
            self._playing = False

    async def speak_streaming(self, text_generator) -> None:
        """Speak text as it streams from LLM, sentence by sentence.

        Args:
            text_generator: Async generator yielding text chunks.
        """
        self._interrupt.clear()
        self._playing = True

        try:
            buffer = ""
            async for chunk in text_generator:
                if self._interrupt.is_set():
                    break

                buffer += chunk

                # Check for complete sentences
                while True:
                    match = SENTENCE_SPLIT.search(buffer)
                    if not match:
                        break

                    # Extract complete sentence
                    sentence = buffer[:match.start()].strip()
                    buffer = buffer[match.end():]

                    if sentence:
                        await self._speak_sentence(sentence)

            # Speak remaining buffer
            if buffer.strip() and not self._interrupt.is_set():
                await self._speak_sentence(buffer.strip())

        finally:
            self._playing = False

    async def _speak_sentence(self, sentence: str) -> None:
        """Synthesize and play a single sentence."""
        if self._interrupt.is_set():
            return

        result = None
        if self.primary == "kokoro" and self.kokoro.available:
            result = self.kokoro.synthesize(sentence)
        if result is None and self.edge_fallback:
            result = await self.edge.synthesize(sentence)
        if result is None:
            return

        samples, sample_rate = result
        await self._play_audio(samples, sample_rate)

    async def _play_audio(self, samples: np.ndarray, sample_rate: int) -> None:
        """Play audio samples through speakers, interruptible.

        Uses sounddevice for low-latency playback.
        """
        if self._interrupt.is_set():
            return

        # Capture the loop HERE, on the async side. Calling
        # asyncio.get_event_loop() from inside the playback thread raises
        # RuntimeError ("no current event loop in thread ..."), and if that is
        # swallowed the done flag never gets set — speak() then spins forever
        # and the whole assistant wedges after its first reply.
        loop = asyncio.get_running_loop()
        done_event = asyncio.Event()

        def _play() -> None:
            try:
                sd.play(samples, samplerate=sample_rate)
                sd.wait()
            except Exception as e:
                log.error("Audio playback error: %s", e)
            finally:
                loop.call_soon_threadsafe(done_event.set)

        threading.Thread(target=_play, daemon=True).start()

        # Never wait longer than the clip could possibly take. Even if the
        # audio backend wedges, the assistant keeps running.
        deadline = time.monotonic() + (len(samples) / sample_rate) + 5.0

        while not done_event.is_set():
            if self._interrupt.is_set():
                sd.stop()
                break
            if time.monotonic() > deadline:
                log.warning("Playback did not finish in time; moving on")
                sd.stop()
                break
            await asyncio.sleep(0.02)
