"""Local LLM fallback via Ollama.

Phase 1: Stub — raises NotImplementedError.
Phase 2: Full Ollama wrapper for Qwen3-4B offline inference.
"""
import logging

log = logging.getLogger("djinn.brain.local_llm")


class LocalBrain:
    """Ollama-based local LLM for offline fallback.

    Uses Qwen3-4B Q4_K_M on CPU (GPU reserved for Whisper).
    Only loaded when Gemini is unreachable.
    """

    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        num_ctx: int = 4096,
    ):
        self.model = model
        self.base_url = base_url
        self.num_ctx = num_ctx
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def initialize(self) -> None:
        """Check if Ollama is running and model is available."""
        # TODO: Phase 2 — check Ollama health endpoint
        log.info("Local LLM stub initialized (not yet implemented)")
        self._available = False

    async def chat(self, query: str, context: str = "") -> str:
        """Send a query to the local LLM.

        Args:
            query: User's query text.
            context: Conversation context.

        Returns:
            Response text.

        Raises:
            NotImplementedError: Phase 1 stub.
        """
        raise NotImplementedError(
            "Local LLM not yet implemented. "
            "Install Ollama and pull qwen3:4b, then implement Phase 2."
        )
