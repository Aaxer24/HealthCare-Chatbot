import pytest

import src.ocr as ocr
import src.service as service
from src.cache import TTLCache
from tests.conftest import make_config


def test_clean_ocr_text_drops_punctuation_only_lines():
    """Scan edges and table borders produce lines of pure punctuation."""
    raw = "Haemoglobin 9.2 g/dL\n|||---|||\nWBC 11400\n...\n"
    assert ocr.clean_ocr_text(raw) == "Haemoglobin 9.2 g/dL\nWBC 11400"


def test_clean_ocr_text_strips_each_line():
    assert ocr.clean_ocr_text("  Hb 9.2  \n   WBC 11400 ") == "Hb 9.2\nWBC 11400"


def test_clean_ocr_text_truncates_very_long_output():
    assert len(ocr.clean_ocr_text("word " * 5000)) <= ocr.MAX_OCR_CHARS


def test_clean_ocr_text_handles_empty():
    assert ocr.clean_ocr_text("") == ""


def test_is_useful_rejects_near_empty_output():
    assert not ocr.is_useful("Hb 9")
    assert ocr.is_useful("Haemoglobin 9.2 g/dL, WBC 11400, Platelets 210000")


def test_extract_rejects_oversized_image():
    with pytest.raises(ValueError, match="larger than"):
        ocr.extract_text_from_image(b"x" * (ocr.MAX_IMAGE_BYTES + 1))


def test_extract_returns_empty_for_empty_bytes():
    assert ocr.extract_text_from_image(b"") == ""


def test_document_text_is_wrapped_with_clear_delimiters():
    """The model must be able to tell the user's own report apart from the
    retrieved reference material."""
    question = service.build_question_with_document("What does this mean?", "Hb 9.2 g/dL")
    assert "UPLOADED DOCUMENT" in question
    assert "Hb 9.2 g/dL" in question
    assert "What does this mean?" in question


def test_build_question_returns_prompt_unchanged_without_document():
    assert service.build_question_with_document("What is anemia?", "") == "What is anemia?"
    assert service.build_question_with_document("What is anemia?", "   ") == "What is anemia?"


def test_upload_bypasses_routing(monkeypatch):
    """An uploaded report is itself the signal that this is a medical question.
    Classifying "what does this mean?" alone would deflect it as OUT_OF_SCOPE --
    the same failure the router had with bare follow-ups."""
    def must_not_classify(*a, **k):
        raise AssertionError("routing should be bypassed when a document is attached")

    monkeypatch.setattr(service, "classify_message", must_not_classify)
    monkeypatch.setattr(service, "answer_style_instruction", lambda *a, **k: "Be clear.")
    monkeypatch.setattr(service, "generate_follow_ups", lambda *a, **k: [])
    monkeypatch.setattr(service, "answer_question", lambda q, h, c: ("Your haemoglobin is low.", []))

    result = service.generate_chat_response(
        "what does this mean?", [], make_config(enable_cache=False), document_text="Hb 9.2 g/dL"
    )
    assert result["message_type"] == "MEDICAL_QUESTION"


def test_same_question_about_different_documents_does_not_share_cache(monkeypatch):
    """Two different reports must never collide on one cached answer."""
    shared_cache = TTLCache(maxsize=8, ttl_seconds=60)
    monkeypatch.setattr(service, "get_index_signature", lambda: "test-index")
    monkeypatch.setattr(service, "get_answer_cache", lambda *a, **k: shared_cache)
    monkeypatch.setattr(service, "answer_style_instruction", lambda *a, **k: "Be clear.")
    monkeypatch.setattr(service, "generate_follow_ups", lambda *a, **k: [])

    answers = iter(["Low haemoglobin.", "High glucose."])
    monkeypatch.setattr(service, "answer_question", lambda q, h, c: (next(answers), []))

    config = make_config(enable_cache=True)
    first = service.generate_chat_response("what does this mean?", [], config, document_text="Hb 9.2")
    second = service.generate_chat_response("what does this mean?", [], config, document_text="Glucose 240")

    assert first["answer"] == "Low haemoglobin."
    assert second["answer"] == "High glucose."
    assert second["cached"] is False


def test_same_question_same_document_does_hit_cache(monkeypatch):
    shared_cache = TTLCache(maxsize=8, ttl_seconds=60)
    monkeypatch.setattr(service, "get_index_signature", lambda: "test-index")
    monkeypatch.setattr(service, "get_answer_cache", lambda *a, **k: shared_cache)
    monkeypatch.setattr(service, "answer_style_instruction", lambda *a, **k: "Be clear.")
    monkeypatch.setattr(service, "generate_follow_ups", lambda *a, **k: [])

    calls = {"n": 0}

    def counting_answer(q, h, c):
        calls["n"] += 1
        return "Low haemoglobin.", []

    monkeypatch.setattr(service, "answer_question", counting_answer)

    config = make_config(enable_cache=True)
    service.generate_chat_response("what does this mean?", [], config, document_text="Hb 9.2")
    second = service.generate_chat_response("what does this mean?", [], config, document_text="Hb 9.2")

    assert second["cached"] is True
    assert calls["n"] == 1
