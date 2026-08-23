import pytest

from src.config import get_cli_config


def test_get_cli_config_requires_groq_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        get_cli_config()


def test_get_cli_config_defaults(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("RETRIEVAL_K", raising=False)
    monkeypatch.delenv("RERANK_K", raising=False)
    monkeypatch.delenv("ENABLE_RERANKING", raising=False)

    config = get_cli_config()

    assert config.groq_api_key == "test-key"
    assert config.retrieval_k == 8
    assert config.rerank_k == 5
    assert config.enable_reranking is True


def test_get_cli_config_clamps_rerank_k_to_retrieval_k(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("RETRIEVAL_K", "3")
    monkeypatch.setenv("RERANK_K", "10")

    config = get_cli_config()

    assert config.retrieval_k == 3
    assert config.rerank_k == 3


def test_get_cli_config_parses_enable_reranking_false(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("ENABLE_RERANKING", "false")

    config = get_cli_config()

    assert config.enable_reranking is False
