"""Voice input pipeline: Microphone → Silero VAD → faster-whisper STT.

Runs on a background thread. Silero VAD detects speech boundaries,
then faster-whisper transcribes the buffered audio on GPU (INT8).
Adaptive silence detection shortens wait for quick queries.
"""
import logging
import threading
import queue
import time
from collections import deque
from typing import Optional, Callable

import numpy as np
import sounddevice as sd

log = logging.getLogger("djinn.voice_input")

# ---------------------------------------------------------------------------
# Silero VAD wrapper
# ---------------------------------------------------------------------------

class SileroVAD:
    """Silero Voice Activity Detection using ONNX runtime.

    Silero v5 accepts only one chunk size per sample rate: 512 samples at
    16kHz, 256 at 8kHz. Anything else raises "Input audio chunk is too short".
    """

    # Sample rate -> the only chunk size the model will accept.
    CHUNK_SAMPLES = {16000: 512, 8000: 256}

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        if sample_rate not in self.CHUNK_SAMPLES:
            raise ValueError(
                f"Silero VAD supports 8000 or 16000 Hz, got {sample_rate}"
            )
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_samples = self.CHUNK_SAMPLES[sample_rate]
        self._model = None

    def load(self) -> None:
        """Load the Silero VAD ONNX model."""
        import silero_vad
        self._model = silero_vad.load_silero_vad(onnx=True)
        self.reset()
        log.info(
            "Silero VAD loaded (ONNX, threshold=%.2f, chunk=%d samples)",
            self.threshold, self.chunk_samples,
        )

    def reset(self) -> None:
        """Reset VAD recurrent state between utterances."""
        if self._model is not None:
            self._model.reset_states()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check whether an audio chunk contains speech.

        Args:
            audio_chunk: float32 array of exactly self.chunk_samples samples.

        Returns:
            True if speech probability exceeds threshold.
        """
        if len(audio_chunk) != self.chunk_samples:
            raise ValueError(
                f"VAD needs exactly {self.chunk_samples} samples at "
                f"{self.sample_rate}Hz, got {len(audio_chunk)}"
            )

        import torch
        tensor = torch.from_numpy(audio_chunk).float()
        with torch.no_grad():
            speech_prob = self._model(tensor, self.sample_rate)
        return float(speech_prob) > self.threshold


# ---------------------------------------------------------------------------
# faster-whisper STT wrapper
# ---------------------------------------------------------------------------

def _register_cuda_dlls() -> None:
    """Make the pip-installed CUDA DLLs findable on Windows.

    ctranslate2 loads cuBLAS/cuDNN dynamically at FIRST INFERENCE, not at
    model load — so a missing DLL passes every startup check and then kills
    the first real transcription. The DLLs come from the nvidia-cublas-cu12 /
    nvidia-cudnn-cu12 wheels, whose bin dirs Windows doesn't search by
    default.
    """
    import os
    import site
    from pathlib import Path

    for sp in site.getsitepackages():
        nvidia = Path(sp) / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            # ctranslate2 resolves some DLLs through PATH rather than the
            # add_dll_directory search path, so set both.
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


class WhisperSTT:
    """faster-whisper speech-to-text with CTranslate2 backend."""

    def __init__(
        self,
        model_name: str = "distil-large-v3",
        device: str = "cuda",
        compute_type: str = "int8",
        beam_size: int = 1,
        language: str = "en",
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self._model = None

    def load(self) -> None:
        """Load the Whisper model. Called once at startup."""
        from faster_whisper import WhisperModel

        if self.device == "cuda":
            _register_cuda_dlls()

        log.info(
            "Loading Whisper '%s' on %s (%s)...",
            self.model_name, self.device, self.compute_type,
        )
        t0 = time.perf_counter()
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        dt = time.perf_counter() - t0
        log.info("Whisper loaded in %.1fs", dt)

        # The first CUDA inference pays ~4s of lazy DLL loading and kernel
        # autotuning. Pay it here, during startup, not on the user's first
        # question. vad_filter must be off or silence skips the encoder.
        if self.device == "cuda":
            t0 = time.perf_counter()
            segments, _ = self._model.transcribe(
                np.zeros(16000, dtype=np.float32),
                beam_size=1, language=self.language, vad_filter=False,
            )
            list(segments)
            log.info("CUDA warmed up in %.1fs", time.perf_counter() - t0)

    def fall_back_to_cpu(self) -> None:
        """Reload the model on CPU after a GPU failure. Sticky for the run."""
        log.warning("Reloading Whisper on CPU — expect slower transcription.")
        self.device = "cpu"
        self._model = None
        self.load()

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio buffer to text.

        Args:
            audio: float32 numpy array, 16kHz mono.

        Returns:
            Transcribed text string (stripped).
        """
        if self._model is None:
            raise RuntimeError("Whisper model not loaded. Call load() first.")

        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=True,
            vad_parameters=dict(
                min_speech_duration_ms=250,
                min_silence_duration_ms=200,
            ),
        )
        text = " ".join(seg.text for seg in segments).strip()
        dt = time.perf_counter() - t0

        if text:
            log.info("STT (%.1fs): \"%s\"", dt, text)
        else:
            log.debug("STT (%.1fs): [empty]", dt)

        return text


# ---------------------------------------------------------------------------
# Audio capture + VAD + STT pipeline
# ---------------------------------------------------------------------------

class VoiceInput:
    """Continuously captures audio, detects speech via VAD, transcribes via Whisper.

    Usage:
        vi = VoiceInput(config)
        vi.load()     # load models
        vi.start()    # start background capture
        text = vi.get_transcription()  # blocks until speech detected + transcribed
        vi.stop()
    """

    def __init__(
        self,
        vad_threshold: float = 0.5,
        min_silence_ms: int = 600,
        min_speech_ms: int = 250,
        max_speech_s: float = 30.0,
        sample_rate: int = 16000,
        input_device: Optional[str | int] = None,
        whisper_model: str = "distil-large-v3",
        whisper_device: str = "cuda",
        whisper_compute: str = "int8",
        whisper_beam: int = 1,
        whisper_language: str = "en",
    ):
        self.sample_rate = sample_rate
        self.input_device = input_device
        self._device_index: Optional[int] = None
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_s = max_speech_s

        # Sub-components
        self.vad = SileroVAD(threshold=vad_threshold, sample_rate=sample_rate)

        # Chunk size is dictated by the VAD, not chosen freely: Silero v5 only
        # accepts 512 samples at 16kHz (= 32ms), and rejects anything else.
        self.chunk_samples = self.vad.chunk_samples
        self.chunk_ms = 1000.0 * self.chunk_samples / sample_rate
        self.stt = WhisperSTT(
            model_name=whisper_model,
            device=whisper_device,
            compute_type=whisper_compute,
            beam_size=whisper_beam,
            language=whisper_language,
        )

        # State
        self._result_queue: queue.Queue[str] = queue.Queue()
        self._running = False
        self._listening = False  # True when actively waiting for speech
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

    def load(self) -> None:
        """Load VAD and Whisper models."""
        self.vad.load()
        self.stt.load()

    def _resolve_device(self) -> None:
        """Pick the microphone to capture from.

        The system default is not trustworthy: on desktops it is often a
        phantom Realtek "Microphone Array" with nothing plugged into it, which
        delivers perfect silence and makes Djinn look deaf. So the device can
        be pinned in config by name substring (survives index reshuffling when
        Bluetooth devices come and go) or by index.
        """
        chosen = None

        if isinstance(self.input_device, int):
            chosen = self.input_device
        elif isinstance(self.input_device, str) and self.input_device.strip():
            want = self.input_device.lower()
            for idx, dev in enumerate(sd.query_devices()):
                if dev["max_input_channels"] < 1 or want not in dev["name"].lower():
                    continue
                try:
                    sd.check_input_settings(device=idx, samplerate=self.sample_rate)
                except Exception:
                    continue  # this host API can't do 16kHz; another entry will
                chosen = idx
                break
            if chosen is None:
                log.warning(
                    "No input device matching %r found (is it connected?). "
                    "Falling back to the system default.", self.input_device,
                )

        self._device_index = chosen
        name = sd.query_devices(chosen, kind="input")["name"]
        log.info("Microphone: %s%s", name, "" if chosen is not None else " (system default)")

    def start(self) -> None:
        """Start background audio capture thread."""
        if self._running:
            return
        self._resolve_device()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info("Voice input started (listening for hotkey activation)")

    def stop(self) -> None:
        """Stop background capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info("Voice input stopped")

    def activate(self) -> None:
        """Activate listening — called when hotkey is pressed."""
        self._listening = True
        log.debug("Listening activated")

    def get_transcription(self, timeout: float = 60.0) -> Optional[str]:
        """Block until a transcription is available.

        Args:
            timeout: Max seconds to wait.

        Returns:
            Transcribed text, or None on timeout.
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        """Main capture loop running on background thread.

        Flow:
        1. Wait for self._listening to be True (set by hotkey)
        2. Open mic stream
        3. Feed 30ms chunks to VAD
        4. Buffer speech audio
        5. When silence detected after speech → transcribe
        6. Put result in queue
        """
        while self._running:
            if not self._listening:
                time.sleep(0.05)  # Idle poll
                continue

            # --- Active listening phase ---
            log.info("🎤 Listening...")
            audio_buffer = []
            is_speaking = False
            speech_start = 0.0
            silence_start = 0.0
            total_speech_chunks = 0
            peak_level = 0.0

            self.vad.reset()

            try:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=self.chunk_samples,
                    device=self._device_index,
                ) as stream:
                    while self._running and self._listening:
                        data, overflowed = stream.read(self.chunk_samples)
                        if overflowed:
                            log.warning("Audio buffer overflowed")

                        chunk = data[:, 0]  # mono
                        peak_level = max(peak_level, float(np.abs(chunk).max()))
                        speech_detected = self.vad.is_speech(chunk)

                        if speech_detected:
                            if not is_speaking:
                                is_speaking = True
                                speech_start = time.perf_counter()
                                log.debug("Speech started")
                            silence_start = 0.0
                            total_speech_chunks += 1
                            audio_buffer.append(chunk.copy())

                        elif is_speaking:
                            # Speech was happening, now silence
                            audio_buffer.append(chunk.copy())  # Keep some trailing audio

                            if silence_start == 0.0:
                                silence_start = time.perf_counter()

                            # Adaptive silence threshold
                            # Short utterances get shorter silence wait
                            speech_duration = time.perf_counter() - speech_start
                            if speech_duration < 1.0:
                                silence_threshold_ms = 400  # Quick query
                            elif speech_duration < 3.0:
                                silence_threshold_ms = self.min_silence_ms
                            else:
                                silence_threshold_ms = 900  # Long speech, wait more

                            silence_duration_ms = (time.perf_counter() - silence_start) * 1000
                            if silence_duration_ms >= silence_threshold_ms:
                                log.debug(
                                    "Speech ended (%.1fs speech, %dms silence)",
                                    speech_duration, silence_duration_ms,
                                )
                                break

                        # Safety: max recording length
                        if is_speaking and (time.perf_counter() - speech_start) > self.max_speech_s:
                            log.warning("Max speech duration reached (%.0fs)", self.max_speech_s)
                            break

            except sd.PortAudioError as e:
                log.error("Audio device error: %s", e)
                self._listening = False
                continue

            self._listening = False

            # --- Transcription phase ---
            if not audio_buffer or total_speech_chunks < int(self.min_speech_ms / self.chunk_ms):
                # Say WHY out loud in the log — this used to fail silently and
                # made a dead microphone indistinguishable from a working one.
                if peak_level < 0.001:
                    log.warning(
                        "No audio at all (peak %.5f) — the microphone is "
                        "delivering silence. Wrong input device? Set "
                        "vad.input_device in config.yaml.", peak_level,
                    )
                else:
                    log.info(
                        "Didn't catch that (peak level %.3f, %d speech chunks) "
                        "— too short or too quiet.", peak_level, total_speech_chunks,
                    )
                self._result_queue.put("")
                continue

            audio = np.concatenate(audio_buffer)

            # An uncaught exception here kills this thread and Djinn goes
            # permanently deaf while looking alive. GPU trouble (missing
            # CUDA DLLs, out of VRAM) degrades to CPU instead.
            try:
                text = self.stt.transcribe(audio)
            except Exception as e:
                log.error("Transcription failed on %s: %s", self.stt.device, e)
                if self.stt.device == "cuda":
                    try:
                        self.stt.fall_back_to_cpu()
                        text = self.stt.transcribe(audio)
                    except Exception as e2:
                        log.error("CPU fallback also failed: %s", e2)
                        text = ""
                else:
                    text = ""

            self._result_queue.put(text)
