from langchain_core.documents import Document

from src.ingest import normalize_documents, normalize_pdf_text


def test_ligatures_are_expanded():
    assert normalize_pdf_text("speciﬁc") == "specific"
    assert normalize_pdf_text("suﬃcient") == "sufficient"
    assert normalize_pdf_text("inﬂammation") == "inflammation"


def test_hyphenation_across_line_break_is_rejoined():
    assert normalize_pdf_text("enven-\noming") == "envenoming"
    assert normalize_pdf_text("classifi-\n  cation") == "classification"


def test_hyphenated_term_not_joined_when_next_line_starts_uppercase():
    """Only lowercase continuations are joined, so "Type-\nII" keeps its dash
    rather than silently becoming a different token."""
    assert "-" in normalize_pdf_text("Type-\nII diabetes")


def test_justified_spacing_is_collapsed():
    assert normalize_pdf_text("Snake  bite  should  be") == "Snake bite should be"


def test_newlines_are_preserved():
    """Line structure carries meaning for lists and table rows, and the chunk
    splitter's separators depend on it."""
    result = normalize_pdf_text("first line\nsecond line")
    assert result == "first line\nsecond line"


def test_blank_lines_between_paragraphs_survive():
    assert normalize_pdf_text("para one\n\npara two") == "para one\n\npara two"


def test_trailing_spaces_before_newline_are_trimmed():
    assert normalize_pdf_text("line with tail   \nnext") == "line with tail\nnext"


def test_empty_input_is_safe():
    assert normalize_pdf_text("") == ""


def test_normalize_documents_updates_in_place():
    docs = [Document(page_content="speciﬁc  text", metadata={"source": "a.pdf"})]
    normalize_documents(docs)
    assert docs[0].page_content == "specific text"
    assert docs[0].metadata["source"] == "a.pdf", "metadata must be preserved"


def test_ligature_followed_by_spurious_spacing_is_joined():
    """These PDFs render "ﬂ  uid" for "fluid" -- the most common single defect
    in the corpus (202 occurrences of that one word alone)."""
    assert normalize_pdf_text("ﬂ  uid") == "fluid"
    assert normalize_pdf_text("speciﬁ  c") == "specific"
    assert normalize_pdf_text("difﬁ  culty") == "difficulty"
    assert normalize_pdf_text("ﬁ  rst") == "first"


def test_ligature_at_end_of_word_before_capital_is_not_joined():
    """Only lowercase continuations join, so a ligature ending a sentence does
    not swallow the next word."""
    assert normalize_pdf_text("the beneﬁt  Then") == "the benefit Then"


def test_realistic_extraction_artefact():
    """A line taken from health-4.pdf, combining every defect at once."""
    raw = "clinically relevant enven-\noming and speciﬁ  c forms of  treatment"
    assert normalize_pdf_text(raw) == "clinically relevant envenoming and specific forms of treatment"
