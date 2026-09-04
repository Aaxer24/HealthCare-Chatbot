import time

import src.service as service
from src.cache import TTLCache, make_cache_key, normalize_prompt
from tests.conftest import make_config


def test_cache_returns_stored_value():
    cache = TTLCache(maxsize=4, ttl_seconds=60)
    cache.set("k", {"answer": "hello"})
    assert cache.get("k") == {"answer": "hello"}
    assert cache.stats()["hits"] == 1


def test_cache_misses_unknown_key():
    cache = TTLCache(maxsize=4, ttl_seconds=60)
    assert cache.get("nope") is None
    assert cache.stats()["misses"] == 1


def test_cache_expires_entries_after_ttl():
    cache = TTLCache(maxsize=4, ttl_seconds=0)  # already expired on read
    cache.set("k", "v")
    time.sleep(0.01)
    assert cache.get("k") is None


def test_cache_evicts_least_recently_used():
    cache = TTLCache(maxsize=2, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")           # 'a' is now most recently used, so 'b' should go
    cache.set("c", 3)

    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None


def test_normalize_prompt_collapses_case_and_whitespace():
    assert normalize_prompt("  What   IS  anemia? ") == "what is anemia?"


def test_cache_key_ignores_case_and_spacing():
    kwargs = dict(model_name="m", retrieval_k=8, rerank_k=5, enable_reranking=True)
    assert make_cache_key("What is anemia?", [], **kwargs) == make_cache_key("what is   anemia?", [], **kwargs)


def test_cache_key_changes_when_config_changes():
    base = dict(model_name="m", retrieval_k=8, rerank_k=5, enable_reranking=True)
    key = make_cache_key("q", [], **base)

    # Each of these genuinely changes the answer, so each must bust the cache.
    assert key != make_cache_key("q", [], **{**base, "retrieval_k": 4})
    assert key != make_cache_key("q", [], **{**base, "enable_reranking": False})
    assert key != make_cache_key("q", [], **{**base, "model_name": "other"})
    assert key != make_cache_key("q", [], index_signature="rebuilt", **base)


def test_cache_key_distinguishes_different_history():
    kwargs = dict(model_name="m", retrieval_k=8, rerank_k=5, enable_reranking=True)
    no_history = make_cache_key("explain more", [], **kwargs)
    with_history = make_cache_key("explain more", [("what is anemia", "Anemia is...")], **kwargs)
    assert no_history != with_history


def test_second_identical_question_is_served_from_cache(monkeypatch):
    """The whole point of the cache: a repeat question must not call the LLM."""
    calls = {"count": 0}

    def counting_classify(prompt, chat_history, config):
        calls["count"] += 1
        return "GENERAL_CHAT"

    monkeypatch.setattr(service, "classify_message", counting_classify)
    monkeypatch.setattr(service, "answer_general_chat", lambda prompt, config: "Hello!")
    monkeypatch.setattr(service, "get_index_signature", lambda: "test-index")
    # Fresh cache so this test does not inherit entries from another test.
    monkeypatch.setattr(service, "get_answer_cache", lambda *a, **k: TTLCache(maxsize=8, ttl_seconds=60))

    config = make_config(enable_cache=True)
    first = service.generate_chat_response("hi there", [], config)
    assert first["cached"] is False
    assert calls["count"] == 1


def test_cache_hit_skips_llm_entirely(monkeypatch):
    calls = {"count": 0}
    shared_cache = TTLCache(maxsize=8, ttl_seconds=60)

    def counting_classify(prompt, chat_history, config):
        calls["count"] += 1
        return "GENERAL_CHAT"

    monkeypatch.setattr(service, "classify_message", counting_classify)
    monkeypatch.setattr(service, "answer_general_chat", lambda prompt, config: "Hello!")
    monkeypatch.setattr(service, "get_index_signature", lambda: "test-index")
    monkeypatch.setattr(service, "get_answer_cache", lambda *a, **k: shared_cache)

    config = make_config(enable_cache=True)
    service.generate_chat_response("hi there", [], config)
    second = service.generate_chat_response("HI  THERE", [], config)  # same after normalisation

    assert second["cached"] is True
    assert second["answer"] == "Hello!"
    assert calls["count"] == 1, "classifier should not run again on a cache hit"


def test_cache_disabled_always_recomputes(monkeypatch):
    calls = {"count": 0}

    def counting_classify(prompt, chat_history, config):
        calls["count"] += 1
        return "GENERAL_CHAT"

    monkeypatch.setattr(service, "classify_message", counting_classify)
    monkeypatch.setattr(service, "answer_general_chat", lambda prompt, config: "Hello!")

    config = make_config(enable_cache=False)
    service.generate_chat_response("hi", [], config)
    service.generate_chat_response("hi", [], config)

    assert calls["count"] == 2
