"""LLM helpers that improve a query before it reaches the retriever.

Two independent techniques live here:

* **Rewriting** turns informal or Hinglish phrasing into the medical vocabulary
  the source PDFs actually use. "sugar ki problem" retrieves almost nothing;
  "diabetes mellitus symptoms" retrieves the right chunks.
* **Multi-query** produces a few differently-angled versions of the same
  information need, so a single unlucky phrasing does not decide what gets
  retrieved.

Both cost one LLM call each. They are applied inside the retriever (see
``rag.QueryExpandingRetriever``) rather than in the service layer so that
``evaluate_chatbot.py`` -- which builds the chain directly -- measures the same
retrieval path that production uses.
"""

from src.config import LOGGER, AppConfig
from src.prompts import MULTI_QUERY_PROMPT, QUERY_REWRITE_PROMPT

MAX_QUERY_CHARS = 300


def _clean_query(text: str) -> str:
    """Strip the decorations LLMs add even when told not to."""
    cleaned = text.strip().strip('"').strip("'").strip()
    # Remove a leading "1." / "-" / "*" bullet if one slipped through.
    for prefix in ("- ", "* ", "• "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    if len(cleaned) > 1 and cleaned[0].isdigit() and cleaned[1] in ".)":
        cleaned = cleaned[2:].strip()
    return cleaned[:MAX_QUERY_CHARS].strip()


def rewrite_query(query: str, config: AppConfig) -> str:
    """Rewrite an informal/Hinglish query into medical search terminology.

    Falls back to the original query on any failure -- a degraded query is far
    better than a failed request.
    """
    from src.rag import complete_text  # local import avoids a circular import

    try:
        rewritten = _clean_query(
            complete_text(
                config,
                QUERY_REWRITE_PROMPT.format(question=query),
                temperature=0.0,
                max_tokens=120,
            )
        )
    except Exception:
        LOGGER.warning("Query rewriting failed; using the original query", exc_info=True)
        return query

    if not rewritten:
        return query
    if rewritten.lower() != query.strip().lower():
        LOGGER.info("Query rewritten: %r -> %r", query, rewritten)
    return rewritten


def generate_query_variations(query: str, config: AppConfig, n: int = 2) -> list[str]:
    """Return up to `n` alternative phrasings of the same information need.

    The original query is never included here; the caller is responsible for
    always retrieving with it too, so a bad variation can only ever add noise
    that fusion and reranking then filter out -- it can never replace the
    user's actual question.
    """
    from src.rag import complete_text

    if n <= 0:
        return []

    try:
        raw = complete_text(
            config,
            MULTI_QUERY_PROMPT.format(question=query, n=n),
            temperature=0.3,  # a little variety is the point here
            max_tokens=200,
        )
    except Exception:
        LOGGER.warning("Multi-query generation failed; continuing without variations", exc_info=True)
        return []

    variations: list[str] = []
    seen = {query.strip().lower()}
    for line in raw.splitlines():
        candidate = _clean_query(line)
        if not candidate or candidate.lower() in seen:
            continue
        variations.append(candidate)
        seen.add(candidate.lower())
        if len(variations) >= n:
            break

    if variations:
        LOGGER.info("Generated %d query variations", len(variations))
    return variations
