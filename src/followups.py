"""Suggested follow-up questions shown under an answer.

Generated from the question and the answer that was actually given, so the
suggestions stay inside what the document set can support -- suggesting a
question the corpus cannot answer would send the user straight into a
"the documents do not cover that" reply.

Costs one extra LLM call per medical answer, so it is behind
``AppConfig.enable_follow_ups`` and is skipped entirely for general chat and
out-of-scope replies, where follow-ups make no sense.
"""

from src.config import LOGGER, AppConfig
from src.prompts import FOLLOW_UP_PROMPT

MAX_FOLLOW_UPS = 3
MAX_FOLLOW_UP_CHARS = 120
# Enough context for relevant suggestions without paying to send a long answer
# back to the model.
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
    """Return up to three follow-up questions; never raises.

    Follow-ups are a nice-to-have garnish on the answer, so any failure returns
    an empty list rather than breaking a response the user is waiting for.
    """
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
        # Require a question mark: the model occasionally emits a stray heading
        # like "Follow-up questions:" which would otherwise become a button.
        if not candidate or "?" not in candidate or candidate.lower() in seen:
            continue
        suggestions.append(candidate)
        seen.add(candidate.lower())
        if len(suggestions) >= MAX_FOLLOW_UPS:
            break

    return suggestions
