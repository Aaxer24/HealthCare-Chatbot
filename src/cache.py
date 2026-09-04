"""In-memory TTL cache for chat answers.

Groq's free tier caps tokens per day for the whole organisation, and a single
answer costs several LLM calls (classify -> style -> condense -> answer, plus
query rewriting and follow-up generation). Repeated questions are common in a
medical FAQ-style bot, so caching whole responses is the cheapest available
saving: a cache hit costs zero tokens.

Deliberately in-process rather than Redis: the app runs as a single uvicorn
worker in one container, so a shared cache server would add infrastructure for
no benefit. The cache is lost on redeploy, which is correct -- a new image may
carry new prompts or a new index, and stale answers should not survive it.
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any

from src.config import LOGGER


class TTLCache:
    """Small thread-safe LRU cache with per-entry expiry.

    uvicorn serves requests from a thread pool, so every mutation is guarded by
    a lock; without it two concurrent requests can corrupt the OrderedDict.
    """

    def __init__(self, maxsize: int = 512, ttl_seconds: int = 86_400) -> None:
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None

            expires_at, value = entry
            if time.time() >= expires_at:
                # Expired: drop it and report a miss so the caller recomputes.
                del self._store[key]
                self.misses += 1
                return None

            self._store.move_to_end(key)  # mark as most recently used
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.time() + self.ttl_seconds, value)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)  # evict least recently used

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._store),
                "maxsize": self.maxsize,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
            }


_answer_cache: TTLCache | None = None
_cache_lock = threading.Lock()


def get_answer_cache(maxsize: int = 512, ttl_seconds: int = 86_400) -> TTLCache:
    global _answer_cache
    with _cache_lock:
        if _answer_cache is None:
            _answer_cache = TTLCache(maxsize=maxsize, ttl_seconds=ttl_seconds)
        return _answer_cache


def normalize_prompt(prompt: str) -> str:
    """Collapse whitespace and case so trivially different phrasings share a key.

    Only safe because the key also carries the chat history and config; this is
    normalisation, not fuzzy matching -- "what is anemia?" and "What is anemia?"
    are the same question, but "anemia treatment" is deliberately not.
    """
    return " ".join(prompt.split()).strip().lower()


def make_cache_key(
    prompt: str,
    chat_history: list[tuple[str, str]],
    *,
    model_name: str,
    retrieval_k: int,
    rerank_k: int,
    enable_reranking: bool,
    index_signature: str = "",
) -> str:
    """Build a cache key covering everything that can change the answer.

    Config and the index signature are part of the key so that changing k,
    toggling the reranker, switching model, or rebuilding the vector store all
    invalidate previous entries automatically instead of serving answers that
    the current pipeline would no longer produce.
    """
    payload = {
        "prompt": normalize_prompt(prompt),
        # Only the user turns matter for identity; assistant text is derived
        # from them and including it would make otherwise-identical
        # conversations miss the cache because of tiny wording differences.
        "history": [normalize_prompt(user) for user, _ in chat_history],
        "model": model_name,
        "retrieval_k": retrieval_k,
        "rerank_k": rerank_k,
        "reranking": enable_reranking,
        "index": index_signature,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def log_cache_event(hit: bool, key: str) -> None:
    LOGGER.info("answer cache %s (key=%s)", "HIT" if hit else "MISS", key[:12])
