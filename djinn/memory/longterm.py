"""Long-term memory: SQLite + sqlite-vec for semantic search.

Phase 1: Stub.
Phase 3: Full implementation with:
  - SQLite for structured data (conversation summaries, user preferences)
  - sqlite-vec for vector search (semantic memory retrieval)
  - all-MiniLM-L6-v2 ONNX embeddings (~80MB, CPU)
"""
import logging

log = logging.getLogger("djinn.memory.longterm")


class LongTermMemory:
    """Long-term memory with semantic search.

    Phase 3 implementation will provide:
    - store_fact(fact, category) — store a learned fact
    - store_summary(conversation_summary) — store conversation summary
    - retrieve(query, top_k=5) — semantic search for relevant memories
    - get_user_preferences() — retrieve stored user preferences
    """

    def __init__(self, db_path: str = "./data/djinn.db"):
        self.db_path = db_path
        self._initialized = False

    def initialize(self) -> None:
        """Initialize database and embedding model."""
        log.info("Long-term memory stub initialized (Phase 3)")
        self._initialized = False

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant memories for a query.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            Formatted memory string, or empty string.
        """
        return ""

    def store(self, content: str, category: str = "general") -> None:
        """Store a memory.

        Args:
            content: Memory content.
            category: Category tag.
        """
        pass
