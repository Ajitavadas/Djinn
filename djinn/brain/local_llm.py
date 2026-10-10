"""Local brain: an Ollama-served model, for offline use and coding.

Same interface as GeminiBrain — initialize(), ready, chat(), stream() — so the
orchestrator can swap one for the other. It is used two ways:

  * pinned, via `--mode local` / `/local`: everything goes to the local model;
  * as a fallback, when Gemini is unconfigured or unreachable.

Talks to Ollama's native /api/chat over HTTP rather than through a client
library: it is one endpoint, httpx is already installed (google-genai depends
on it), and the native API is the only one that exposes `think` to switch off
Qwen3.5's reasoning — which, as with Gemini's fast tier, is dead air in a
voice assistant.

Tools work too: Ollama accepts OpenAI-style function schemas, and the calls it
returns are run through the same registry as Gemini's. web_search still needs
Gemini behind it, so it only works while online.
"""
import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Optional

import httpx

log = logging.getLogger("djinn.brain.local_llm")

# Same cap as the Gemini brain — a small model is, if anything, more prone to
# calling the same tool over and over.
MAX_TOOL_ROUNDS = 5


class LocalBrain:
    """Ollama-backed local model with Gemini-compatible chat() and stream()."""

    def __init__(
        self,
        model: str = "qwen3.5-4b-q3km-text",
        vision_model: str = "",
        base_url: str = "http://localhost:11434",
        num_ctx: int = 4096,
        num_gpu: Optional[int] = None,
        think: bool = False,
        keep_alive: str = "30m",
        options: Optional[dict] = None,
        system_prompt_template: str = "",
        user_name: str = "User",
        workspace: str = "",
        use_tools: bool = True,
    ):
        self.model = model
        # Same weights plus the vision projector. Loading the projector costs
        # ~0.7 GB of VRAM on every query, which on a 4 GB card pushes layers
        # onto the CPU — so it is only used for messages that carry images.
        self.vision_model = vision_model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.num_gpu = num_gpu
        self.think = think
        self.keep_alive = keep_alive
        self.options = options or {}
        self.system_prompt_template = system_prompt_template
        self.user_name = user_name
        self.workspace = workspace
        self.use_tools = use_tools

        self._ready = False
        self._loaded = False
        # A small model cold-loading from disk onto a 4 GB card can take a
        # while, so reads get a long timeout; connects stay short so a missing
        # server is noticed at once.
        self._timeout = httpx.Timeout(connect=3.0, read=180.0, write=30.0, pool=5.0)

    @property
    def ready(self) -> bool:
        return self._ready

    # Kept for the name the stub used.
    @property
    def available(self) -> bool:
        return self._ready

    def initialize(self) -> None:
        """Check Ollama is running and the model is pulled. Never raises."""
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            r.raise_for_status()
        except httpx.HTTPError as e:
            log.warning(
                "Ollama not reachable at %s (%s). Local model disabled — "
                "install Ollama and run: ollama serve",
                self.base_url, e,
            )
            self._ready = False
            return

        names = {m.get("name", "") for m in r.json().get("models", [])}

        def have(model: str) -> bool:
            # "qwen3.5:4b" is listed as-is; a bare name is listed as ":latest".
            return (model if ":" in model else f"{model}:latest") in names

        if not have(self.model):
            log.warning(
                "Ollama is running but %s is not installed. "
                "Run scripts/setup_local_models.ps1, or: ollama pull %s",
                self.model, self.model,
            )
            self._ready = False
            return

        if self.vision_model and not have(self.vision_model):
            log.warning(
                "Vision model %s not installed — images will not work.",
                self.vision_model,
            )
            self.vision_model = ""

        log.info(
            "Local model ready: %s via Ollama (ctx=%d, gpu layers=%s, vision=%s)",
            self.model, self.num_ctx,
            "auto" if self.num_gpu is None else self.num_gpu,
            self.vision_model or "none",
        )
        self._ready = True

    async def warm_up(self) -> None:
        """Load the model into memory now, so the first real query is not slow.

        An empty prompt makes Ollama load the weights and return immediately.
        If num_gpu forces more layers onto the GPU than fit, the load fails;
        we then hand the split back to Ollama and load again.
        """
        if not self._ready:
            return
        self._loaded = True
        t0 = time.perf_counter()
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    r = await client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "keep_alive": self.keep_alive,
                            "options": self._options(self.model),
                        },
                    )
                    r.raise_for_status()
                log.info("Local model loaded in %.1fs", time.perf_counter() - t0)
                return
            except httpx.HTTPError as e:
                if attempt == 0 and self.num_gpu is not None:
                    log.warning(
                        "Loading %s with num_gpu=%s failed (%s) — "
                        "letting Ollama choose the GPU split instead",
                        self.model, self.num_gpu, e,
                    )
                    self.num_gpu = None
                    continue
                log.warning("Local model warm-up failed: %s", e)
                return

    # ------------------------------------------------------------------
    # Request construction
    # ------------------------------------------------------------------

    def _system_prompt(self, memories: str = "") -> str:
        if not self.system_prompt_template:
            return (
                f"You are Djinn, a personal AI assistant for {self.user_name}. "
                "Be concise, technical, and direct. No filler phrases."
            )
        return self.system_prompt_template.format(
            name=self.user_name,
            memories=memories or "No memories yet.",
            workspace=self.workspace or "the workspace folder",
        )

    def _messages(
        self,
        query: str,
        history: list[tuple[str, str]],
        memories: str,
        images: Optional[list[str]] = None,
    ) -> list[dict]:
        messages = [{"role": "system", "content": self._system_prompt(memories)}]
        messages += [{"role": role, "content": text} for role, text in history]
        user = {"role": "user", "content": query}
        if images:
            user["images"] = images  # base64-encoded, as Ollama expects
        messages.append(user)
        return messages

    def _model_for(self, images: Optional[list[str]]) -> str:
        return self.vision_model if images and self.vision_model else self.model

    def _options(self, model: str) -> dict:
        opts = {"num_ctx": self.num_ctx, **self.options}
        # Unset means Ollama decides how many layers fit on the GPU. Its
        # estimate is cautious — on a 4 GB card it leaves a quarter of a model
        # on the CPU that would in fact fit — so num_gpu can force it. Never
        # for the vision model: the projector really does not leave room.
        if self.num_gpu is not None and model == self.model:
            opts["num_gpu"] = self.num_gpu
        return opts

    def _payload(
        self, messages: list[dict], stream: bool, model: str, think: bool
    ) -> dict:
        from djinn.tools import registry

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": think,
            "keep_alive": self.keep_alive,
            "options": self._options(model),
        }
        if self.use_tools:
            schemas = registry.json_schemas()
            if schemas:
                payload["tools"] = schemas
        return payload

    async def _run_tools(self, messages: list[dict], calls: list[dict]) -> None:
        """Run the model's tool calls and append the results to the transcript."""
        from djinn.tools import registry

        parsed = [
            (c["function"]["name"], _args(c["function"].get("arguments")))
            for c in calls
        ]
        results = await asyncio.gather(
            *(registry.dispatch(name, args) for name, args in parsed)
        )
        for (name, _a), result in zip(parsed, results):
            messages.append({"role": "tool", "tool_name": name, "content": result})

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def _ensure_loaded(self) -> None:
        """Load the text model on first use, so a bad num_gpu is caught early."""
        if not self._loaded:
            await self.warm_up()

    async def chat(
        self,
        query: str,
        history: Optional[list[tuple[str, str]]] = None,
        memories: str = "",
        deep: bool = False,
        think: Optional[bool] = None,
        images: Optional[list[str]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Return the full answer, running any tools.

        `deep` is accepted for interface parity with GeminiBrain and ignored;
        `think` overrides the configured default for this one query.
        """
        if not self._ready:
            return "The local model is not available. Check that Ollama is running."

        think = self.think if think is None else think
        # Thinking on this hardware runs 1-3 minutes; give it room.
        timeout = timeout or (900.0 if think else 180.0)
        model = self._model_for(images)
        if model == self.model:
            await self._ensure_loaded()

        messages = self._messages(query, history or [], memories, images)
        try:
            async with httpx.AsyncClient(timeout=self._client_timeout(think)) as client:
                return await asyncio.wait_for(
                    self._chat_loop(client, messages, model, think), timeout=timeout
                )
        except asyncio.TimeoutError:
            log.error("Local model timed out after %.0fs", timeout)
            return "Sorry, the local model took too long."
        except httpx.HTTPError as e:
            log.error("Local model error: %s", e)
            return self._explain_error(e)

    def _client_timeout(self, think: bool) -> httpx.Timeout:
        # A non-streamed thinking answer arrives as one response after minutes.
        return httpx.Timeout(connect=3.0, read=900.0 if think else 180.0, write=30.0, pool=5.0)

    async def _chat_loop(
        self, client: httpx.AsyncClient, messages: list[dict], model: str, think: bool
    ) -> str:
        for _round in range(MAX_TOOL_ROUNDS):
            r = await client.post(
                f"{self.base_url}/api/chat",
                json=self._payload(messages, stream=False, model=model, think=think),
            )
            r.raise_for_status()
            msg = r.json().get("message", {})
            calls = msg.get("tool_calls") or []
            if not calls:
                text = (msg.get("content") or "").strip()
                return text or "I didn't get a response for that."
            messages.append(msg)
            await self._run_tools(messages, calls)

        log.warning("Hit the %d-round tool limit", MAX_TOOL_ROUNDS)
        return "I got stuck on that. Could you rephrase?"

    async def stream(
        self,
        query: str,
        history: Optional[list[tuple[str, str]]] = None,
        memories: str = "",
        deep: bool = False,
        think: Optional[bool] = None,
        images: Optional[list[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream the answer in chunks, running tools as needed.

        `deep` is ignored; `think` overrides the configured default for this
        query. The model's reasoning arrives in a separate field and is never
        yielded — only the answer is spoken.

        Text is yielded as it arrives until the model asks for a tool; after
        that, the rest of the round is narration and is not spoken. Ollama
        usually sends a tool call before any text, so in practice only the
        final answer is heard — and holding everything back to be sure would
        cost the time-to-first-word that streaming exists to save.
        """
        if not self._ready:
            yield "The local model is not available. Check that Ollama is running."
            return

        think = self.think if think is None else think
        model = self._model_for(images)
        if model == self.model:
            await self._ensure_loaded()

        messages = self._messages(query, history or [], memories, images)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                for _round in range(MAX_TOOL_ROUNDS):
                    calls: list[dict] = []
                    text: list[str] = []
                    emitted = False

                    async with client.stream(
                        "POST",
                        f"{self.base_url}/api/chat",
                        json=self._payload(messages, stream=True, model=model, think=think),
                    ) as r:
                        if r.status_code >= 400:
                            body = (await r.aread()).decode(errors="replace")
                            raise httpx.HTTPStatusError(
                                f"{r.status_code}: {body[:200]}",
                                request=r.request, response=r,
                            )
                        async for line in r.aiter_lines():
                            if not line:
                                continue
                            chunk = json.loads(line)
                            if chunk.get("error"):
                                raise RuntimeError(chunk["error"])
                            msg = chunk.get("message") or {}
                            if msg.get("tool_calls"):
                                calls += msg["tool_calls"]
                            piece = msg.get("content") or ""
                            if piece:
                                text.append(piece)
                                if not calls:
                                    emitted = True
                                    yield piece

                    if not calls:
                        if not emitted:
                            yield "I didn't get a response for that."
                        return

                    messages.append(
                        {"role": "assistant", "content": "".join(text), "tool_calls": calls}
                    )
                    await self._run_tools(messages, calls)

                yield "I got stuck on that. Could you rephrase?"

        except (httpx.HTTPError, RuntimeError, json.JSONDecodeError) as e:
            log.error("Local model streaming error: %s", e)
            yield self._explain_error(e)

    def _explain_error(self, e: Exception) -> str:
        """Turn an Ollama failure into something worth saying out loud."""
        msg = str(e).lower()
        if isinstance(e, httpx.ConnectError):
            return "I can't reach Ollama. Is it running?"
        if "not found" in msg or "404" in msg:
            return f"Ollama doesn't have {self.model}. Pull it first."
        if "memory" in msg or "cuda" in msg:
            return "The local model ran out of memory. Try a smaller context or model."
        return "Sorry, the local model returned an error."


def _args(raw) -> dict:
    """Tool arguments arrive as a dict from Ollama, but tolerate a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
