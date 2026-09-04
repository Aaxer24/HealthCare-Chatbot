import src.followups as followups
import src.rag as rag
from tests.conftest import make_config


def test_parses_three_questions(monkeypatch):
    monkeypatch.setattr(
        rag, "complete_text",
        lambda *a, **k: "What causes anemia?\nHow is anemia treated?\nCan anemia be prevented?",
    )
    result = followups.generate_follow_ups("What is anemia?", "Anemia is...", make_config())
    assert result == [
        "What causes anemia?",
        "How is anemia treated?",
        "Can anemia be prevented?",
    ]


def test_strips_numbering_and_bullets(monkeypatch):
    monkeypatch.setattr(
        rag, "complete_text",
        lambda *a, **k: "1. What causes anemia?\n- How is it treated?\n* Is it preventable?",
    )
    result = followups.generate_follow_ups("q", "a", make_config())
    assert result == ["What causes anemia?", "How is it treated?", "Is it preventable?"]


def test_drops_lines_without_a_question_mark(monkeypatch):
    """Models sometimes emit a heading line, which must not become a button."""
    monkeypatch.setattr(
        rag, "complete_text",
        lambda *a, **k: "Follow-up questions:\nWhat causes anemia?\nHere are some ideas",
    )
    assert followups.generate_follow_ups("q", "a", make_config()) == ["What causes anemia?"]


def test_caps_at_three(monkeypatch):
    monkeypatch.setattr(rag, "complete_text", lambda *a, **k: "\n".join(f"Q{i}?" for i in range(10)))
    assert len(followups.generate_follow_ups("q", "a", make_config())) == 3


def test_skips_suggestion_identical_to_the_question(monkeypatch):
    monkeypatch.setattr(rag, "complete_text", lambda *a, **k: "What is anemia?\nWhat causes anemia?")
    result = followups.generate_follow_ups("What is anemia?", "a", make_config())
    assert result == ["What causes anemia?"]


def test_returns_empty_when_disabled(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not call the LLM when follow-ups are disabled")

    monkeypatch.setattr(rag, "complete_text", boom)
    assert followups.generate_follow_ups("q", "a", make_config(enable_follow_ups=False)) == []


def test_returns_empty_for_blank_answer(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not call the LLM with nothing to summarise")

    monkeypatch.setattr(rag, "complete_text", boom)
    assert followups.generate_follow_ups("q", "   ", make_config()) == []


def test_failure_returns_empty_rather_than_raising(monkeypatch):
    """Follow-ups are a garnish; they must never break a real answer."""
    def boom(*a, **k):
        raise RuntimeError("groq down")

    monkeypatch.setattr(rag, "complete_text", boom)
    assert followups.generate_follow_ups("q", "a", make_config()) == []
