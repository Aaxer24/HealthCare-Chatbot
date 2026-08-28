import src.router as router
from tests.conftest import make_config


def test_classify_message_out_of_scope(monkeypatch):
    monkeypatch.setattr(router, "invoke_llm_text", lambda llm, system, user: "OUT_OF_SCOPE")
    assert router.classify_message("what's the weather", [], make_config()) == "OUT_OF_SCOPE"


def test_classify_message_general_chat(monkeypatch):
    monkeypatch.setattr(router, "invoke_llm_text", lambda llm, system, user: "GENERAL_CHAT")
    assert router.classify_message("hi there", [], make_config()) == "GENERAL_CHAT"


def test_classify_message_defaults_to_medical_question(monkeypatch):
    monkeypatch.setattr(router, "invoke_llm_text", lambda llm, system, user: "MEDICAL_QUESTION")
    assert router.classify_message("what are diabetes symptoms", [], make_config()) == "MEDICAL_QUESTION"


def test_classify_message_ambiguous_label_prefers_medical(monkeypatch):
    # If the label mentions both, treat it as needing document grounding
    # rather than risk skipping retrieval for an actual medical question.
    monkeypatch.setattr(router, "invoke_llm_text", lambda llm, system, user: "GENERAL_CHAT MEDICAL_QUESTION")
    assert router.classify_message("ambiguous", [], make_config()) == "MEDICAL_QUESTION"


def test_classify_message_uses_recent_history_for_bare_followup(monkeypatch):
    # A bare follow-up like "explain it more" has no medical keywords on its
    # own -- the recent conversation must reach the classifier prompt so it
    # can recognize this continues a medical topic instead of defaulting to
    # OUT_OF_SCOPE.
    captured = {}

    def fake_invoke(llm, system, user):
        captured["user"] = user
        return "MEDICAL_QUESTION"

    monkeypatch.setattr(router, "invoke_llm_text", fake_invoke)
    history = [("what is anemia", "Anemia is a condition where...")]
    result = router.classify_message("explain it more", history, make_config())

    assert result == "MEDICAL_QUESTION"
    assert "what is anemia" in captured["user"]
    assert "explain it more" in captured["user"]
