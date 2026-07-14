"""Web search tool — Gemini with native Google Search grounding.

Vertex rejects any request that mixes a search tool with custom functions:
"Multiple tools are supported only when they are all search tools."

So search cannot simply sit alongside open_app/read_file in one tool list.
Instead it is exposed as an ordinary function. When the model calls it, we
make a SEPARATE, grounded Gemini call here and hand the answer back as the
function result. The main conversation only ever sees custom functions, so
the constraint never applies.
"""
import logging

log = logging.getLogger("djinn.tools.web")

_client = None
_model = "gemini-2.5-flash"


def configure(client, model: str = "gemini-2.5-flash") -> None:
    """Give the tool a genai client to run grounded lookups with."""
    global _client, _model
    _client = client
    _model = model


async def web_search(query: str) -> str:
    """Search the web and return a grounded summary with sources.

    Args:
        query: What to look up.

    Returns:
        A short factual summary, plus the sites it came from.
    """
    if _client is None:
        return "Web search is not configured."

    from google.genai import types

    try:
        response = await _client.aio.models.generate_content(
            model=_model,
            contents=(
                "Answer this using web search. Be factual and concise — "
                f"three sentences at most.\n\n{query}"
            ),
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )

        answer = (response.text or "").strip()
        if not answer:
            return f"No results found for '{query}'."

        sources = _sources(response)
        if sources:
            answer += "\n\nSources: " + ", ".join(sources)

        log.info(
            "web_search(%r) -> %d chars, %d sources",
            query[:60], len(answer), len(sources),
        )
        return answer

    except Exception as e:
        log.error("web_search failed: %s", e)
        return f"Web search failed: {e}"


def _sources(response) -> list[str]:
    """Pull the distinct source titles out of the grounding metadata."""
    try:
        meta = response.candidates[0].grounding_metadata
        if not meta or not meta.grounding_chunks:
            return []
        seen: list[str] = []
        for chunk in meta.grounding_chunks:
            if chunk.web and chunk.web.title and chunk.web.title not in seen:
                seen.append(chunk.web.title)
        return seen[:4]
    except (AttributeError, IndexError):
        return []
