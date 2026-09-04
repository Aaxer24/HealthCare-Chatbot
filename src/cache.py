"""In-memory TTL cache for chat answers, keyed on prompt + history + config.

A cache hit skips every LLM call for that turn, which matters a lot on
Groq's free tier. Not Redis-backed since we're a single container/worker;
cache just resets on redeploy, which is fine.
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any

from src.config import LOGGER


class TTLCache:
    """Thread-safe LRU cache with per-entry expiry."""

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
                del self._store[key]
                self.misses += 1
                return None

            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.time() + self.ttl_seconds, value)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

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
    """"What is anemia?" and "what is anemia?" should hit the same key."""
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
    """Key covers everything that can change the answer, so a config change
    or index rebuild naturally invalidates old entries."""
    payload = {
        "prompt": normalize_prompt(prompt),
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
