"""Working memory: sliding window of recent conversation turns.

Maintains a fixed-size deque of (role, content) pairs.
Formatted and injected into the system prompt for every LLM call.
"""
import logging
from collections import deque
from typing import Literal

log = logging.getLogger("djinn.memory.working")

Role = Literal["user", "assistant", "system"]


class WorkingMemory:
    """Sliding window conversation memory.

    Keeps the last N turns (user + assistant pairs) and formats
    them for injection into the LLM's context.
    """

    def __init__(self, window_size: int = 12):
        """Initialize working memory.

        Args:
            window_size: Maximum number of turns to keep.
                         A turn is one (role, content) entry.
                         12 turns = ~6 exchanges.
        """
        self.window_size = window_size
        self._history: deque[tuple[Role, str]] = deque(maxlen=window_size)

    def add(self, role: Role, content: str) -> None:
        """Add a turn to memory.

        Args:
            role: "user", "assistant", or "system".
            content: The message content.
        """
        self._history.append((role, content))
        log.debug("Memory: added %s turn (%d/%d)", role, len(self._history), self.window_size)

    def get_history(self) -> list[tuple[Role, str]]:
        """Return history as (role, content) turns for the LLM.

        Preferred over get_context(): the brain sends these as real message
        turns, so the model sees a proper conversation rather than a blob of
        text pasted into the prompt.
        """
        return list(self._history)

    def get_context(self) -> str:
        """Format memory as a context string for the LLM.

        Legacy: only for display or prompt-splicing. Prefer get_history().

        Returns:
            Formatted conversation history, e.g.:
            User: What's Python?
            Djinn: Python is a programming language...
        """
        if not self._history:
            return ""

        lines = []
        for role, content in self._history:
            prefix = "User" if role == "user" else "Djinn"
            # Truncate very long messages for context efficiency
            truncated = content[:500] + "..." if len(content) > 500 else content
            lines.append(f"{prefix}: {truncated}")

        return "\n".join(lines)

    def get_last_exchange(self) -> tuple[str, str]:
        """Get the most recent user query and assistant response.

        Returns:
            Tuple of (last_user_query, last_assistant_response).
            Empty strings if not available.
        """
        last_user = ""
        last_assistant = ""

        for role, content in reversed(self._history):
            if role == "assistant" and not last_assistant:
                last_assistant = content
            elif role == "user" and not last_user:
                last_user = content
            if last_user and last_assistant:
                break

        return last_user, last_assistant

    def clear(self) -> None:
        """Clear all memory."""
        self._history.clear()
        log.info("Working memory cleared")

    @property
    def turn_count(self) -> int:
        """Number of turns currently in memory."""
        return len(self._history)

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"WorkingMemory(turns={len(self._history)}/{self.window_size})"
