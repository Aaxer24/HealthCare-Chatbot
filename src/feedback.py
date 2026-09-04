"""Thumbs-up/down persistence.

Negative feedback is what scripts/review_feedback.py stages into the golden
eval set, so this is really the input to that improvement loop.

JSON Lines, append-only -- one JSON object per line so a partial/interrupted
write can't corrupt earlier records.

NOTE: container filesystem is ephemeral. This path must be a mounted volume
on the server or feedback disappears on every redeploy (see ci.yml).
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.config import BASE_DIR, LOGGER

DEFAULT_FEEDBACK_PATH = BASE_DIR / "eval" / "feedback" / "feedback.jsonl"

_write_lock = threading.Lock()


def get_feedback_path() -> Path:
    return Path(os.getenv("FEEDBACK_PATH", str(DEFAULT_FEEDBACK_PATH)))


def record_feedback(
    question: str,
    answer: str,
    rating: str,
    *,
    sources: list[dict] | None = None,
    comment: str = "",
    message_type: str = "",
) -> dict:
    """Appends one record and returns it. Raises ValueError on a bad rating
    so the API can 422 instead of storing garbage."""
    if rating not in {"up", "down"}:
        raise ValueError(f"rating must be 'up' or 'down', got {rating!r}")

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rating": rating,
        "question": question,
        "answer": answer,
        "message_type": message_type,
        "comment": comment,
        "sources": [
            {"source": s.get("source", ""), "snippet": s.get("snippet", "")}
            for s in (sources or [])
        ],
    }

    path = get_feedback_path()
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info("feedback recorded: %s for %r", rating, question[:60])
    return record


def load_feedback(path: Path | None = None) -> list[dict]:
    """Skips corrupt lines instead of failing the whole read."""
    path = path or get_feedback_path()
    if not path.exists():
        return []

    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            LOGGER.warning("Skipping malformed feedback line %d in %s", line_number, path)
    return records


def feedback_summary(path: Path | None = None) -> dict:
    records = load_feedback(path)
    up = sum(1 for r in records if r.get("rating") == "up")
    down = sum(1 for r in records if r.get("rating") == "down")
    total = up + down
    return {
        "total": total,
        "up": up,
        "down": down,
        "satisfaction_rate": round(up / total, 3) if total else 0.0,
    }
