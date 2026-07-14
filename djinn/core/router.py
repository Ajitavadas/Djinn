"""Task router: classifies queries into local | flash | pro.

Phase 1: Fast keyword-based classification (zero latency).
Phase 2 upgrade: Gemini Flash micro-call for ambiguous queries.
"""
import logging
import re
from typing import Literal

log = logging.getLogger("djinn.router")

RouteTarget = Literal["local", "flash", "pro"]

# ---------------------------------------------------------------------------
# Pattern definitions for local routing
# ---------------------------------------------------------------------------

# Queries that can be handled locally (no LLM needed)
LOCAL_PATTERNS = [
    r"\b(what|whats|what's)\s+(time|date|day)\b",
    r"\b(current|today'?s?)\s+(time|date|day)\b",
    r"\bset\s+(a\s+)?(timer|alarm|reminder)\b",
    r"\bremind\s+me\b",
    r"\bopen\s+\w+",
    r"\blaunch\s+\w+",
    r"\b(copy|paste|clipboard)\b",
    r"\b(volume|brightness)\s*(up|down|mute)\b",
    r"\bshutdown\b",
    r"\brestart\b",
    r"\block\s*(screen)?\b",
]

# Queries that warrant Gemini Pro (complex reasoning)
PRO_PATTERNS = [
    r"\b(analyze|analysis)\s+(this|the)\s+(document|paper|code|file)\b",
    r"\b(compare|contrast|synthesize)\b.*\b(multiple|several|different)\b",
    r"\bresearch\s+(paper|topic|subject|area)\b",
    r"\b(step.by.step|multi.step|chain.of.thought)\b",
    r"\bwrite\s+(a\s+)?(detailed|comprehensive|thorough)\b",
    r"\b(debug|refactor)\s+(this|the)\s+(entire|whole|full)\b",
    r"\bexplain\s+(in\s+)?detail\b",
]

_local_compiled = [re.compile(p, re.IGNORECASE) for p in LOCAL_PATTERNS]
_pro_compiled = [re.compile(p, re.IGNORECASE) for p in PRO_PATTERNS]


def route(query: str) -> RouteTarget:
    """Classify a query into a routing target.

    Priority:
    1. Check for local patterns (instant, no LLM)
    2. Check for pro patterns (complex, use Gemini Pro)
    3. Default to flash (most queries)

    Args:
        query: User's transcribed query text.

    Returns:
        One of "local", "flash", or "pro".
    """
    query_clean = query.strip()

    if not query_clean:
        return "local"

    # Check local patterns
    for pattern in _local_compiled:
        if pattern.search(query_clean):
            log.debug("Routed to LOCAL: matched pattern for '%s'", query_clean[:50])
            return "local"

    # Check pro patterns
    for pattern in _pro_compiled:
        if pattern.search(query_clean):
            log.debug("Routed to PRO: matched pattern for '%s'", query_clean[:50])
            return "pro"

    # Default: Gemini Flash handles everything else
    log.debug("Routed to FLASH (default) for '%s'", query_clean[:50])
    return "flash"


def get_route_label(target: RouteTarget) -> str:
    """Human-readable label for a route target."""
    labels = {
        "local": "⚡ Local",
        "flash": "🔵 Gemini Flash",
        "pro": "🟣 Gemini Pro",
    }
    return labels.get(target, target)
