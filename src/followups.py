"""Suggested follow-up questions shown under an answer.

Built from the question + the actual answer given, so suggestions stay
inside what the docs can support. Costs one extra LLM call, so it's gated by
AppConfig.enable_follow_ups and skipped for general chat / out-of-scope.
"""

from src.config import LOGGER, AppConfig
from src.prompts import FOLLOW_UP_PROMPT

MAX_FOLLOW_UPS = 3
MAX_FOLLOW_UP_CHARS = 120
ANSWER_EXCERPT_CHARS = 1200


def _clean_suggestion(line: str) -> str:
    cleaned = line.strip().strip('"').strip("'").strip()
    for prefix in ("- ", "* ", "• "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    if len(cleaned) > 1 and cleaned[0].isdigit() and cleaned[1] in ".)":
        cleaned = cleaned[2:].strip()
    return cleaned[:MAX_FOLLOW_UP_CHARS].strip()


def generate_follow_ups(question: str, answer: str, config: AppConfig) -> list[str]:
    """Never raises -- these are a nice-to-have, not worth failing the answer over."""
    from src.rag import complete_text

    if not config.enable_follow_ups or not answer.strip():
        return []

    try:
        raw = complete_text(
            config,
            FOLLOW_UP_PROMPT.format(question=question, answer=answer[:ANSWER_EXCERPT_CHARS]),
            temperature=0.4,
            max_tokens=180,
        )
    except Exception:
        LOGGER.warning("Follow-up generation failed; returning none", exc_info=True)
        return []

    suggestions: list[str] = []
    seen: set[str] = {question.strip().lower()}
    for line in raw.splitlines():
        candidate = _clean_suggestion(line)
        # skip stray headings like "Follow-up questions:" that have no "?"
        if not candidate or "?" not in candidate or candidate.lower() in seen:
            continue
        suggestions.append(candidate)
        seen.add(candidate.lower())
        if len(suggestions) >= MAX_FOLLOW_UPS:
            break

    return suggestions
