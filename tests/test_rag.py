from langchain_core.documents import Document

from src.rag import clean_snippet, format_source, source_previews


def test_format_source_includes_page_number_as_1_indexed():
    doc = Document(page_content="x", metadata={"source": "C:/data/health-1.pdf", "page": 0})
    assert format_source(doc) == "health-1.pdf, page 1"


def test_format_source_without_page():
    doc = Document(page_content="x", metadata={"source": "health-1.pdf"})
    assert format_source(doc) == "health-1.pdf"


def test_format_source_unknown_when_missing():
    doc = Document(page_content="x", metadata={})
    assert format_source(doc) == "Unknown source"


def test_clean_snippet_collapses_whitespace():
    assert clean_snippet("a   b\n\nc") == "a b c"


def test_clean_snippet_truncates_at_word_boundary():
    text = "word " * 200  # far past the default 520-char limit
    snippet = clean_snippet(text, limit=20)
    assert snippet.endswith("...")
    assert len(snippet) <= 24  # 20 + "..." plus rounding to a whole word


def test_source_previews_deduplicates_and_skips_unknown_source():
    docs = [
        Document(page_content="Same text here", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="Same text here", metadata={"source": "a.pdf", "page": 0}),  # duplicate
        Document(page_content="Other text", metadata={}),  # unknown source, skipped
        Document(page_content="Different", metadata={"source": "b.pdf", "page": 1, "rerank_score": 3.14159}),
    ]

    previews = source_previews(docs)

    assert len(previews) == 2
    assert previews[0]["source"] == "a.pdf, page 1"
    assert previews[1]["source"] == "b.pdf, page 2"
    assert previews[1]["score"] == "3.14"
