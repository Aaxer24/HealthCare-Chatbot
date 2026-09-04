import json

import pytest

import src.feedback as feedback


@pytest.fixture
def feedback_file(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("FEEDBACK_PATH", str(path))
    return path


def test_record_feedback_writes_a_json_line(feedback_file):
    feedback.record_feedback("What is anemia?", "Anemia is...", "up")

    lines = feedback_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["question"] == "What is anemia?"
    assert record["rating"] == "up"
    assert "timestamp_utc" in record


def test_record_feedback_appends_rather_than_overwrites(feedback_file):
    feedback.record_feedback("q1", "a1", "up")
    feedback.record_feedback("q2", "a2", "down")
    assert len(feedback_file.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_record_feedback_rejects_invalid_rating(feedback_file):
    with pytest.raises(ValueError):
        feedback.record_feedback("q", "a", "maybe")


def test_record_feedback_keeps_sources_for_review(feedback_file):
    feedback.record_feedback(
        "q", "a", "down",
        sources=[{"source": "health-3.pdf, page 2", "snippet": "text", "score": "0.9"}],
    )
    record = json.loads(feedback_file.read_text(encoding="utf-8").strip())
    assert record["sources"] == [{"source": "health-3.pdf, page 2", "snippet": "text"}]


def test_load_feedback_returns_empty_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_PATH", str(tmp_path / "nope.jsonl"))
    assert feedback.load_feedback() == []


def test_load_feedback_skips_corrupt_lines(feedback_file):
    """One bad line must not make the whole history unreadable."""
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    feedback_file.write_text(
        json.dumps({"rating": "up", "question": "good"}) + "\n"
        + "{ this is not json\n"
        + json.dumps({"rating": "down", "question": "also good"}) + "\n",
        encoding="utf-8",
    )
    records = feedback.load_feedback()
    assert len(records) == 2


def test_feedback_summary_counts_and_rate(feedback_file):
    feedback.record_feedback("q1", "a", "up")
    feedback.record_feedback("q2", "a", "up")
    feedback.record_feedback("q3", "a", "down")

    summary = feedback.feedback_summary()
    assert summary == {"total": 3, "up": 2, "down": 1, "satisfaction_rate": 0.667}


def test_feedback_summary_handles_no_feedback(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_PATH", str(tmp_path / "none.jsonl"))
    assert feedback.feedback_summary()["satisfaction_rate"] == 0.0
