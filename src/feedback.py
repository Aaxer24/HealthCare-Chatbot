"""Persistence for thumbs-up/down feedback on answers.

This is the entry point of the improvement loop described in the README: a
thumbs-down marks a question the pipeline handled badly, and those questions are
exactly the ones worth adding to the golden evaluation set. Growing the eval set
from real user pain -- rather than from questions we imagined -- is what keeps
RAGAS scores meaningful over time.

Stored as JSON Lines because the file is append-only and may be written while
something else reads it; one self-contained JSON object per line means a partial
write can never corrupt earlier records.

IMPORTANT: the container filesystem is ephemeral. On the server this path must
be a mounted volume, or every redeploy silently discards collected feedback.
See the deploy step in .github/workflows/ci.yml.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.config import BASE_DIR, LOGGER

DEFAULT_FEEDBACK_PATH = BASE_DIR / "eval" / "feedback" / "feedback.jsonl"

# Appends happen from uvicorn's thread pool; a lock keeps two concurrent
# writes from interleaving inside a single line.
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
    """Append one feedback record. Returns the record written.

    Raises ValueError on an invalid rating so the API can answer 422 rather than
    silently storing garbage that the review step would later have to filter.
    """
    if rating not in {"up", "down"}:
        raise ValueError(f"rating must be 'up' or 'down', got {rating!r}")

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rating": rating,
        "question": question,
        "answer": answer,
        "message_type": message_type,
        "comment": comment,
        # Kept so a reviewer can see which chunks produced a bad answer without
        # having to re-run the pipeline.
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
    """Read all feedback records, skipping any corrupt lines.

    A single malformed line (truncated write, manual edit) should not make the
    whole history unreadable, so bad lines are logged and skipped.
    """
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
