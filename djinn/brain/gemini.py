"""Gemini Brain: fast + deep tiers over the google-genai SDK.

Supports two backends behind one interface:

  * vertex   — Google Cloud Vertex AI. Billed to a GCP project (uses your
               Cloud credits). Auth is ADC: `gcloud auth application-default
               login`. No API key.
  * aistudio — Google AI Studio. Auth is a GEMINI_API_KEY. Requires prepaid
               credits on the key itself; Cloud credits do NOT apply here.

The brain is stateless: conversation history is passed in from WorkingMemory
on every call rather than held in a chat object. That keeps a single source of
truth for history and avoids sending it twice.
"""
import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

log = logging.getLogger("djinn.brain.gemini")

# Thinking adds seconds of latency before the first token. For a voice
# assistant that is dead air, so the fast tier disables it outright and the
# deep tier keeps it.
THINKING_OFF = 0

# How many times the model may call tools before we make it answer. Stops a
# confused model from looping on search forever.
MAX_TOOL_ROUNDS = 5


class GeminiBrain:
    """Gemini client with a fast tier and a deep-reasoning tier."""

    def __init__(
        self,
        backend: str = "vertex",
        api_key: str = "",
        project: str = "",
        location: str = "us-central1",
        flash_model: str = "gemini-2.5-flash",
        pro_model: str = "gemini-2.5-pro",
        system_prompt_template: str = "",
        user_name: str = "User",
        workspace: str = "",
        max_output_tokens: int = 800,
        pro_max_output_tokens: int = 2048,
        pro_thinking_budget: int = 1024,
        use_tools: bool = True,
    ):
        self.use_tools = use_tools
        self.backend = backend
        self.api_key = api_key
        self.project = project
        self.location = location
        self.flash_model = flash_model
        self.pro_model = pro_model
        self.system_prompt_template = system_prompt_template
        self.user_name = user_name
        self.workspace = workspace
        self.max_output_tokens = max_output_tokens
        self.pro_max_output_tokens = pro_max_output_tokens
        self.pro_thinking_budget = pro_thinking_budget

        self._client = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def client(self):
        """The underlying genai client — tools reuse it for grounded lookups."""
        return self._client

    def initialize(self) -> None:
        """Create the genai client for the configured backend."""
        try:
            from google import genai
        except ImportError:
            log.error("google-genai not installed. Run: uv add google-genai")
            return

        try:
            if self.backend == "vertex":
                if not self.project:
                    log.error(
                        "Vertex backend needs a project id. Set gemini.project "
                        "in config.yaml, or run: gcloud config set project <id>"
                    )
                    return
                self._client = genai.Client(
                    vertexai=True,
                    project=self.project,
                    location=self.location,
                )
                log.info(
                    "Gemini ready via Vertex AI (project=%s, location=%s)",
                    self.project, self.location,
                )
            else:
                if not self.api_key:
                    log.error("AI Studio backend needs GEMINI_API_KEY.")
                    return
                self._client = genai.Client(api_key=self.api_key)
                log.info("Gemini ready via AI Studio")

            log.info("  fast tier: %s | deep tier: %s", self.flash_model, self.pro_model)
            self._ready = True

        except Exception as e:
            log.error("Failed to initialize Gemini (%s): %s", self.backend, e)
            self._ready = False

    # ------------------------------------------------------------------
    # Prompt / history construction
    # ------------------------------------------------------------------

    def _system_prompt(self, memories: str = "") -> str:
        """Render the system prompt template."""
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

    def _build_contents(self, query: str, history: list[tuple[str, str]]):
        """Turn (role, text) history plus the new query into genai contents.

        History arrives from WorkingMemory. Gemini names the assistant role
        "model", not "assistant".
        """
        from google.genai import types

        contents = []
        for role, text in history:
            contents.append(
                types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=[types.Part.from_text(text=text)],
                )
            )
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=query)])
        )
        return contents

    def _config(self, deep: bool, memories: str, with_tools: bool = False):
        """Build the generation config for a tier.

        Gemini charges thinking tokens against max_output_tokens. Leave
        thinking uncapped on a tight budget and it will spend the entire
        allowance reasoning and return an EMPTY string — Djinn goes silent.
        So each tier budgets thinking explicitly:

          fast: thinking off entirely, whole budget goes to words.
          deep: thinking capped well below max_output, leaving room to speak.
        """
        from google.genai import types

        from djinn.tools import registry

        kwargs: dict = {"system_instruction": self._system_prompt(memories)}

        if with_tools and self.use_tools:
            tools = registry.tools()
            if tools:
                kwargs["tools"] = tools

        if deep:
            budget = min(self.pro_thinking_budget, self.pro_max_output_tokens // 2)
            kwargs["max_output_tokens"] = self.pro_max_output_tokens
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
        else:
            kwargs["max_output_tokens"] = self.max_output_tokens
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=THINKING_OFF
            )

        return types.GenerateContentConfig(**kwargs)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def chat(
        self,
        query: str,
        history: Optional[list[tuple[str, str]]] = None,
        memories: str = "",
        deep: bool = False,
        timeout: float = 90.0,
    ) -> str:
        """Send a query to Gemini and return the answer, running any tools.

        Gemini may ask to call tools (search the web, open an app, run code).
        Each round we execute what it asked for, hand the results back, and let
        it continue — until it produces a final answer or hits MAX_TOOL_ROUNDS.

        Args:
            query: The user's query.
            history: Prior (role, text) turns from WorkingMemory.
            memories: Long-term memory string for the system prompt.
            deep: Use the deep-reasoning tier instead of the fast one.
            timeout: Total seconds for the whole exchange, tools included.
        """
        if not self._ready:
            return "Gemini is not initialized. Check the logs."

        import asyncio

        try:
            return await asyncio.wait_for(
                self._chat_with_tools(query, history or [], memories, deep),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            log.error("Exchange timed out after %.0fs", timeout)
            return "Sorry, that took too long. Try again?"
        except Exception as e:
            log.error("Brain error: %s", e)
            return self._explain_error(e)

    async def _chat_with_tools(
        self,
        query: str,
        history: list[tuple[str, str]],
        memories: str,
        deep: bool,
    ) -> str:
        """The tool-calling loop."""
        from google.genai import types

        from djinn.tools import registry

        model = self.pro_model if deep else self.flash_model
        contents = self._build_contents(query, history)
        config = self._config(deep, memories, with_tools=True)
        t0 = time.perf_counter()

        for round_num in range(MAX_TOOL_ROUNDS):
            response = await self._client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )

            candidate = response.candidates[0] if response.candidates else None
            parts = (candidate.content.parts if candidate and candidate.content else None) or []
            calls = [p.function_call for p in parts if p.function_call]

            if not calls:
                text = (response.text or "").strip()
                if text:
                    log.debug(
                        "%s answered in %.2fs after %d tool round(s)",
                        model, time.perf_counter() - t0, round_num,
                    )
                    return text
                return self._no_text(response, model)

            # Keep the model's tool-call turn in the transcript, then append
            # one function_response per call. Both are required: drop either
            # and the next request 400s on a malformed history.
            contents.append(candidate.content)

            results = await asyncio.gather(
                *(registry.dispatch(c.name, dict(c.args or {})) for c in calls)
            )

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=call.name, response={"result": result}
                        )
                        for call, result in zip(calls, results)
                    ],
                )
            )

        log.warning("Hit the %d-round tool limit", MAX_TOOL_ROUNDS)
        return "I got stuck looking that up. Could you rephrase?"

    def _no_text(self, response, model: str) -> str:
        """Explain an empty response instead of going silent."""
        usage = getattr(response, "usage_metadata", None)
        thoughts = getattr(usage, "thoughts_token_count", 0) or 0
        log.warning(
            "%s returned no text (thinking used %d tokens). "
            "Raise max_output_tokens or lower the thinking budget.",
            model, thoughts,
        )
        return "I ran out of room thinking about that. Try asking it more simply?"

    async def stream(
        self,
        query: str,
        history: Optional[list[tuple[str, str]]] = None,
        memories: str = "",
        deep: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Stream the answer in chunks, running tools as needed.

        Pairs with VoiceOutput.speak_streaming(): Djinn starts speaking the
        first sentence while the rest is still being written.

        Tool calls are handled as in chat() — when the model asks for tools we
        run them and keep going. A turn that ends in tool calls yields nothing,
        so the user never hears "let me look that up"; only the final answer,
        the round with no tool calls, is spoken.
        """
        if not self._ready:
            yield "Gemini is not initialized."
            return

        from google.genai import types

        from djinn.tools import registry

        model = self.pro_model if deep else self.flash_model
        contents = self._build_contents(query, history or [])
        config = self._config(deep, memories, with_tools=True)

        try:
            for _round in range(MAX_TOOL_ROUNDS):
                calls: list = []
                parts_seen: list = []
                pending_text: list[str] = []
                emitted = False

                stream = await self._client.aio.models.generate_content_stream(
                    model=model, contents=contents, config=config
                )

                async for chunk in stream:
                    candidate = chunk.candidates[0] if chunk.candidates else None
                    if not candidate or not candidate.content:
                        continue

                    for part in candidate.content.parts or []:
                        parts_seen.append(part)

                        if part.function_call:
                            calls.append(part.function_call)
                            pending_text.clear()  # it was preamble, not an answer
                        elif part.text:
                            if calls:
                                continue  # trailing narration around a tool call
                            pending_text.append(part.text)

                    # Nothing has asked for a tool yet, so this text is the real
                    # answer — release it and start the voice going.
                    if not calls and pending_text:
                        for piece in pending_text:
                            emitted = True
                            yield piece
                        pending_text.clear()

                if not calls:
                    if not emitted:
                        yield "I didn't get a response for that."
                    return

                # Hand the tool results back and go round again.
                contents.append(types.Content(role="model", parts=parts_seen))
                results = await asyncio.gather(
                    *(registry.dispatch(c.name, dict(c.args or {})) for c in calls)
                )
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=call.name, response={"result": result}
                            )
                            for call, result in zip(calls, results)
                        ],
                    )
                )

            yield "I got stuck looking that up. Could you rephrase?"

        except Exception as e:
            log.error("%s streaming error: %s", model, e)
            yield self._explain_error(e)

    def _explain_error(self, e: Exception) -> str:
        """Turn an SDK exception into something worth saying out loud."""
        msg = str(e).lower()
        if "credit" in msg or "quota" in msg or "429" in msg or "resource_exhausted" in msg:
            return "My Gemini quota is exhausted. Check the billing on your key."
        if "permission" in msg or "403" in msg or "denied" in msg:
            return "Gemini denied the request. Check credentials and that the API is enabled."
        if "not found" in msg or "404" in msg:
            return "That Gemini model name doesn't exist. Check config.yaml."
        return "Sorry, Gemini returned an error."
