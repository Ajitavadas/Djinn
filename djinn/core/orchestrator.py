"""Orchestrator: main event loop connecting all Djinn components.

Handles:
- Hotkey activation (Ctrl+Alt+D)
- Mode detection (online/offline/gaming)
- Voice pipeline: listen → route → think → speak
- Text-only mode for debugging
"""
import asyncio
import logging
import os
import time
from typing import Optional

import yaml
import keyboard

from djinn.core.voice_input import VoiceInput
from djinn.core.voice_output import VoiceOutput
from djinn.core import router
from djinn.brain.gemini import GeminiBrain
from djinn.memory.working import WorkingMemory
from djinn.tools import registry
from djinn.ui.chat_window import ChatWindow

log = logging.getLogger("djinn.orchestrator")


class Orchestrator:
    """Main Djinn orchestrator — glues all components together."""

    # Manual override of the router.
    #   auto -> router picks local / flash / pro per query
    #   fast -> always the fast tier (never pro)
    #   pro  -> always the deep tier (never flash)
    MODES = ("auto", "fast", "pro")

    def __init__(
        self,
        config_path: str = "config.yaml",
        force_cpu: bool = False,
        text_only: bool = False,
        mode: Optional[str] = None,
    ):
        self.text_only = text_only
        self.force_cpu = force_cpu
        self._running = False

        # Load config
        self.config = self._load_config(config_path)
        djinn_cfg = self.config.get("djinn", {})

        # CLI flag wins over config file.
        self.mode = mode or djinn_cfg.get("mode", "auto")
        if self.mode not in self.MODES:
            log.warning("Unknown mode %r, falling back to auto", self.mode)
            self.mode = "auto"
        self.mode_hotkey = djinn_cfg.get("mode_hotkey", "ctrl+alt+m")
        whisper_cfg = self.config.get("whisper", {})
        vad_cfg = self.config.get("vad", {})
        tts_cfg = self.config.get("tts", {})
        gemini_cfg = self.config.get("gemini", {})
        memory_cfg = self.config.get("memory", {})
        self.tools_cfg = self.config.get("tools", {})

        # Hotkey
        self.hotkey = djinn_cfg.get("wake_hotkey", "ctrl+alt+d")

        # Voice Input
        whisper_device = "cpu" if force_cpu else whisper_cfg.get("device", "cuda")
        self.voice_input = VoiceInput(
            vad_threshold=vad_cfg.get("threshold", 0.5),
            min_silence_ms=vad_cfg.get("min_silence_ms", 600),
            min_speech_ms=vad_cfg.get("min_speech_ms", 250),
            max_speech_s=vad_cfg.get("max_speech_s", 30.0),
            sample_rate=vad_cfg.get("sample_rate", 16000),
            input_device=vad_cfg.get("input_device"),
            whisper_model=whisper_cfg.get("model", "distil-large-v3"),
            whisper_device=whisper_device,
            whisper_compute=whisper_cfg.get("compute_type", "int8"),
            whisper_beam=whisper_cfg.get("beam_size", 1),
            whisper_language=whisper_cfg.get("language", "en"),
        )

        # Speak sentences as they are generated, rather than after the full reply.
        self.streaming = tts_cfg.get("streaming", True)

        # Voice Output
        self.voice_output = VoiceOutput(
            primary=tts_cfg.get("primary", "edge"),
            kokoro_voice=tts_cfg.get("kokoro_voice", "af_heart"),
            edge_voice=tts_cfg.get("edge_voice", "en-IN-NeerjaNeural"),
            edge_fallback=tts_cfg.get("edge_fallback", True),
        )

        # Brain
        api_key = os.environ.get("GEMINI_API_KEY") or gemini_cfg.get("api_key", "")
        project = (
            gemini_cfg.get("project")
            or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        )
        self.brain = GeminiBrain(
            backend=gemini_cfg.get("backend", "vertex"),
            api_key=api_key,
            project=project,
            location=gemini_cfg.get("location", "us-central1"),
            flash_model=gemini_cfg.get("flash_model", "gemini-2.5-flash"),
            pro_model=gemini_cfg.get("pro_model", "gemini-2.5-pro"),
            system_prompt_template=self.config.get("system_prompt", ""),
            user_name=djinn_cfg.get("name", "User"),
            max_output_tokens=gemini_cfg.get("max_output_tokens", 800),
            pro_max_output_tokens=gemini_cfg.get("pro_max_output_tokens", 2048),
            pro_thinking_budget=gemini_cfg.get("pro_thinking_budget", 1024),
            use_tools=self.tools_cfg.get("enabled", True),
        )

        # Memory
        self.memory = WorkingMemory(
            window_size=memory_cfg.get("working_window", 12),
        )

    def _load_config(self, path: str) -> dict:
        """Load YAML configuration file."""
        if not os.path.exists(path):
            log.warning("Config not found at %s, using defaults", path)
            return {}

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        log.info("Config loaded from %s", path)
        return config

    async def run(self) -> None:
        """Main async run loop."""
        self._running = True

        # --- Initialize all components ---
        log.info("Initializing components...")

        # Load voice output (TTS)
        self.voice_output.load()

        # Load voice input (VAD + STT) — skip in text-only mode
        if not self.text_only:
            log.info("Loading voice pipeline (this may take a moment)...")
            self.voice_input.load()
            self.voice_input.start()

        # Initialize brain
        self.brain.initialize()

        # Tools reuse the brain's client for grounded web lookups.
        if self.brain.ready and self.tools_cfg.get("enabled", True):
            registry.configure(
                client=self.brain.client,
                search_model=self.brain.flash_model,
                workspace=self.tools_cfg.get("workspace", "~/Djinn"),
                allow_code_execution=self.tools_cfg.get(
                    "allow_code_execution", False
                ),
            )
        else:
            self.brain.use_tools = False
            log.info("Tools disabled")

        log.info("━" * 50)
        log.info("  Djinn is ready!")
        if self.text_only:
            log.info("  Mode: TEXT INPUT (type your queries)")
        else:
            log.info("  Press %s to speak", self.hotkey)
        log.info("━" * 50)

        # --- Main loop ---
        if self.text_only:
            # Launch GUI chat window instead of terminal input
            self._chat_window = ChatWindow(
                on_close=self.shutdown,
                on_mode_change=self.set_mode,
                mode=self.mode,
            )
            self._chat_window.start()
            await self._text_loop()
        else:
            # Speak greeting (voice mode only)
            await self.voice_output.speak("Djinn is ready.")
            await self._voice_loop()

    async def _voice_loop(self) -> None:
        """Voice-activated main loop."""
        # Register hotkey
        hotkey_pressed = asyncio.Event()
        loop = asyncio.get_event_loop()

        def on_hotkey():
            loop.call_soon_threadsafe(hotkey_pressed.set)

        keyboard.add_hotkey(self.hotkey, on_hotkey, suppress=False)
        log.info("Hotkey registered: %s (speak)", self.hotkey)

        # Cycling the mode speaks the new mode aloud, so it needs the loop.
        mode_changed = asyncio.Event()

        def on_mode_hotkey():
            loop.call_soon_threadsafe(mode_changed.set)

        keyboard.add_hotkey(self.mode_hotkey, on_mode_hotkey, suppress=False)
        log.info("Hotkey registered: %s (cycle auto/fast/pro)", self.mode_hotkey)

        async def watch_mode_hotkey() -> None:
            while self._running:
                await mode_changed.wait()
                mode_changed.clear()
                if not self._running:
                    break
                self.voice_output.interrupt()
                await self.voice_output.speak(self.cycle_mode())

        mode_task = asyncio.create_task(watch_mode_hotkey())

        while self._running:
            # Wait for hotkey
            await hotkey_pressed.wait()
            hotkey_pressed.clear()

            if not self._running:
                break

            # Interrupt any current speech
            self.voice_output.interrupt()

            # Activate listening
            self.voice_input.activate()

            # Wait for transcription (runs on background thread)
            text = await asyncio.get_event_loop().run_in_executor(
                None, self.voice_input.get_transcription, 30.0
            )

            if not text:
                log.debug("No speech detected")
                continue

            # Process the query
            await self._process_query(text)

        mode_task.cancel()
        keyboard.remove_hotkey(self.hotkey)
        keyboard.remove_hotkey(self.mode_hotkey)

    async def _text_loop(self) -> None:
        """Text input loop using the GUI chat window."""
        while self._running:
            text = await self._chat_window.get_input()

            # None sentinel means window was closed
            if text is None:
                break

            if not text:
                continue

            if text.lower() in ("exit", "quit", "bye"):
                self._chat_window.append_message("Djinn", "Goodbye.")
                await self.voice_output.speak("Goodbye.")
                break

            # Slash commands never reach the LLM.
            if text.startswith("/"):
                cmd = text[1:].strip().lower()
                if cmd in self.MODES:
                    # set_mode on the window keeps the badge in sync, and its
                    # on_mode_change callback updates the orchestrator.
                    self._chat_window.set_mode(cmd, announce=True)
                elif cmd in ("clear", "reset"):
                    self.memory.clear()
                    self._chat_window.append_message(
                        "system", "Conversation memory cleared."
                    )
                else:
                    self._chat_window.append_message(
                        "system",
                        f"Unknown command /{cmd}. "
                        "Try /auto, /fast, /pro, /clear.",
                    )
                continue

            await self._process_query(text)

    async def _process_query(self, query: str) -> None:
        """Process a single user query through the full pipeline.

        Flow: route → get context → send to brain → speak response → update memory
        """
        t0 = time.perf_counter()

        # 1. Route the query — unless the user has pinned a mode.
        target = self._resolve_target(query)
        log.info(
            "Query: \"%s\" → %s [%s]",
            query[:80], router.get_route_label(target), self.mode,
        )

        # 2. Handle local queries (no LLM needed)
        if target == "local":
            response = self._handle_local(query)
            if response:
                await self._respond(query, response, t0)
                return

            # If local handler returned nothing, fall through to the fast tier
            target = "flash"

        # 3. Send to brain. History goes as real message turns, not spliced
        #    into the prompt, so it is sent exactly once.
        deep = target == "pro"

        if self.streaming:
            await self._respond_streaming(query, deep, t0)
            return

        try:
            response = await self.brain.chat(
                query,
                history=self.memory.get_history(),
                deep=deep,
            )
        except Exception as e:
            log.error("Brain error: %s", e)
            response = "Sorry, I had trouble processing that. Try again?"

        if response:
            await self._respond(query, response, t0)

    async def _respond_streaming(self, query: str, deep: bool, t0: float) -> None:
        """Stream the answer straight into the voice, speaking as it arrives.

        The text is tee'd: each chunk goes to the speaker and, in text mode, to
        the chat window, while being accumulated for memory. Speech therefore
        starts on the first sentence instead of waiting for the whole reply.
        """
        chunks: list[str] = []
        window = self._chat_window if self.text_only and hasattr(self, "_chat_window") else None
        first_chunk_at: Optional[float] = None

        async def tee():
            nonlocal first_chunk_at
            if window:
                window.begin_stream()
            try:
                async for chunk in self.brain.stream(
                    query, history=self.memory.get_history(), deep=deep
                ):
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter()
                    chunks.append(chunk)
                    if window:
                        window.stream_chunk(chunk)
                    yield chunk
            finally:
                if window:
                    window.end_stream()

        try:
            await self.voice_output.speak_streaming(tee())
        except Exception as e:
            log.error("Streaming failed: %s", e)

        response = "".join(chunks).strip()
        if not response:
            return

        ttft = (first_chunk_at - t0) if first_chunk_at else 0.0
        log.info(
            "Response (first text %.1fs, done %.1fs): \"%s\"",
            ttft, time.perf_counter() - t0, response[:100],
        )

        self.memory.add("user", query)
        self.memory.add("assistant", response)

    def _resolve_target(self, query: str) -> str:
        """Decide which tier handles this query.

        In auto mode the router decides. In fast/pro mode the user has pinned
        a tier, so we skip the router — except for the local shortcuts (time,
        date), which are instant and free and worth keeping in every mode.
        """
        target = router.route(query)

        if self.mode == "auto" or target == "local":
            return target

        return "flash" if self.mode == "fast" else "pro"

    def set_mode(self, mode: str) -> str:
        """Switch mode. Returns a line worth showing/speaking."""
        if mode not in self.MODES:
            return f"Unknown mode {mode}."
        self.mode = mode
        log.info("Mode set to %s", mode.upper())
        return {
            "auto": "Auto mode. I'll pick the right model per question.",
            "fast": "Fast mode. Quick answers only.",
            "pro": "Pro mode. Deeper reasoning, slower replies.",
        }[mode]

    def cycle_mode(self) -> str:
        """Advance to the next mode. Returns a line worth showing/speaking."""
        nxt = self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)]
        return self.set_mode(nxt)

    async def _respond(self, query: str, response: str, t0: float) -> None:
        """Display, speak, and remember a response."""
        dt = time.perf_counter() - t0
        log.info("Response (%.1fs): \"%s\"", dt, response[:100])

        if self.text_only and hasattr(self, "_chat_window"):
            self._chat_window.append_message("Djinn", response)

        await self.voice_output.speak(response)

        # Remember every exchange, including locally-answered ones, so
        # follow-ups like "and tomorrow?" have something to refer back to.
        self.memory.add("user", query)
        self.memory.add("assistant", response)

    def _handle_local(self, query: str) -> Optional[str]:
        """Handle queries that don't need an LLM.

        Returns response text, or None to fall through to LLM.
        """
        q = query.lower().strip()

        # Time
        if any(w in q for w in ["what time", "current time", "whats the time", "what's the time"]):
            from datetime import datetime
            now = datetime.now()
            return f"It's {now.strftime('%I:%M %p')}."

        # Date
        if any(w in q for w in ["what date", "today's date", "what day", "what's the date"]):
            from datetime import datetime
            now = datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}."

        return None

    def shutdown(self) -> None:
        """Graceful shutdown of all components."""
        log.info("Shutting down Djinn...")
        self._running = False

        if not self.text_only:
            self.voice_input.stop()

        self.voice_output.interrupt()
        log.info("Djinn stopped.")
