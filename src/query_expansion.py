"""Query rewriting + multi-query fan-out before retrieval.

Rewriting turns Hinglish/informal input into medical terminology ("sugar ki
problem" -> "diabetes mellitus symptoms"). Multi-query generates a couple of
alternate phrasings so retrieval doesn't depend on one lucky wording.

Both live in the retriever (rag.QueryExpandingRetriever) instead of the
service layer so evaluate_chatbot.py exercises the same retrieval path as
production.
"""

from src.config import LOGGER, AppConfig
from src.prompts import MULTI_QUERY_PROMPT, QUERY_REWRITE_PROMPT

MAX_QUERY_CHARS = 300


def _clean_query(text: str) -> str:
    """Strip quotes/bullets/numbering the LLM adds despite being told not to."""
    cleaned = text.strip().strip('"').strip("'").strip()
    for prefix in ("- ", "* ", "• "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    if len(cleaned) > 1 and cleaned[0].isdigit() and cleaned[1] in ".)":
        cleaned = cleaned[2:].strip()
    return cleaned[:MAX_QUERY_CHARS].strip()


def rewrite_query(query: str, config: AppConfig) -> str:
    """Falls back to the original query on any failure."""
    from src.rag import complete_text  # avoids a circular import

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
    """Alternate phrasings of the same question. Doesn't include the original
    -- caller always retrieves with that separately, so a bad variation can
    only add noise, never replace the real query."""
    from src.rag import complete_text

    if n <= 0:
        return []

    try:
        raw = complete_text(
            config,
            MULTI_QUERY_PROMPT.format(question=query, n=n),
            temperature=0.3,
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
