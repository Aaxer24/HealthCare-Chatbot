import src.service as service
from tests.conftest import make_config


def test_general_chat_skips_retrieval(monkeypatch):
    monkeypatch.setattr(service, "classify_message", lambda prompt, chat_history, config: "GENERAL_CHAT")
    monkeypatch.setattr(service, "answer_general_chat", lambda prompt, config: "Hi! I can answer health questions.")
    monkeypatch.setattr(service, "answer_question", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not retrieve")))

    result = service.generate_chat_response("hi", [], make_config())

    assert result == {
        "message_type": "GENERAL_CHAT",
        "answer": "Hi! I can answer health questions.",
        "sources": [],
    }


def test_out_of_scope_skips_retrieval(monkeypatch):
    monkeypatch.setattr(service, "classify_message", lambda prompt, chat_history, config: "OUT_OF_SCOPE")
    monkeypatch.setattr(service, "answer_out_of_scope", lambda prompt, config: "I only handle health questions.")
    monkeypatch.setattr(service, "answer_question", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not retrieve")))

    result = service.generate_chat_response("what's the weather", [], make_config())

    assert result["message_type"] == "OUT_OF_SCOPE"
    assert result["sources"] == []


def test_medical_question_retrieves_and_returns_sources(monkeypatch):
    monkeypatch.setattr(service, "classify_message", lambda prompt, chat_history, config: "MEDICAL_QUESTION")
    monkeypatch.setattr(service, "answer_style_instruction", lambda prompt, chat_history, config: "Be concise.")

    captured = {}

    def fake_answer_question(question, chat_history, config):
        captured["question"] = question
        captured["chat_history"] = chat_history
        return "Diabetes symptoms include...", [{"source": "health-3.pdf, page 1", "snippet": "...", "score": "0.90"}]

    monkeypatch.setattr(service, "answer_question", fake_answer_question)

    result = service.generate_chat_response("what are diabetes symptoms", [("prior", "answer")], make_config())

    assert result["message_type"] == "MEDICAL_QUESTION"
    assert result["answer"] == "Diabetes symptoms include..."
    assert result["sources"] == [{"source": "health-3.pdf, page 1", "snippet": "...", "score": "0.90"}]
    # The style instruction should be appended to the question sent downstream.
    assert "Be concise." in captured["question"]
    assert captured["chat_history"] == [("prior", "answer")]
