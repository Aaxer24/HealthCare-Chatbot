import pytest
from fastapi.testclient import TestClient

import src.api as api


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api, "validate_runtime", lambda: None)
    monkeypatch.setattr(api, "get_cli_config", lambda: object())
    # The startup lifespan pre-warms the embedding model, reranker, and FAISS
    # index (see src/api.py) -- stub these out so tests stay fast and don't
    # need a real vector store or network access to Hugging Face Hub.
    monkeypatch.setattr(api, "get_embedding_model", lambda: None)
    monkeypatch.setattr(api, "get_reranker", lambda: None)
    monkeypatch.setattr(api, "get_vectorstore", lambda: None)
    with TestClient(api.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_answer_and_sources(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "generate_chat_response",
        lambda prompt, chat_history, config, document_text="": {
            "message_type": "MEDICAL_QUESTION",
            "answer": "Diabetes symptoms include...",
            "sources": [{"source": "health-3.pdf, page 1", "snippet": "...", "score": "0.90"}],
        },
    )

    response = client.post("/chat", json={"prompt": "what are diabetes symptoms", "chat_history": []})

    assert response.status_code == 200
    body = response.json()
    assert body["message_type"] == "MEDICAL_QUESTION"
    assert body["sources"][0]["source"] == "health-3.pdf, page 1"


def test_chat_pairs_alternating_history_correctly(client, monkeypatch):
    captured = {}

    def fake_generate(prompt, chat_history, config, document_text=""):
        captured["chat_history"] = chat_history
        return {"message_type": "MEDICAL_QUESTION", "answer": "ok", "sources": []}

    monkeypatch.setattr(api, "generate_chat_response", fake_generate)

    response = client.post(
        "/chat",
        json={
            "prompt": "follow-up question",
            "chat_history": [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
            ],
        },
    )

    assert response.status_code == 200
    assert captured["chat_history"] == [("first question", "first answer")]


def test_chat_error_does_not_leak_internal_exception_details(client, monkeypatch):
    def boom(prompt, chat_history, config):
        raise RuntimeError("Groq API key sk-secret-value-12345 rejected by upstream")

    monkeypatch.setattr(api, "generate_chat_response", boom)

    response = client.post("/chat", json={"prompt": "hi", "chat_history": []})

    assert response.status_code == 500
    assert "sk-secret-value-12345" not in response.text
