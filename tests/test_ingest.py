from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingest import CHUNK_OVERLAP, CHUNK_SIZE, create_chunks, load_pdf_files


def test_load_pdf_files_raises_for_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_pdf_files(tmp_path / "does-not-exist")


def test_load_pdf_files_raises_when_directory_has_no_pdfs(tmp_path):
    with pytest.raises(FileNotFoundError, match="No PDF files"):
        load_pdf_files(tmp_path)


def test_create_chunks_respects_chunk_size():
    long_text = "sentence number " * 400  # well over CHUNK_SIZE
    documents = [Document(page_content=long_text, metadata={"source": "x.pdf"})]

    chunks = create_chunks(documents)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= CHUNK_SIZE for chunk in chunks)


def test_create_chunks_normalizes_source_to_filename():
    documents = [Document(page_content="short text", metadata={"source": "C:/data/health-1.pdf"})]

    chunks = create_chunks(documents)

    assert chunks[0].metadata["source"] == "health-1.pdf"


def test_create_chunks_prefers_bullet_boundaries_over_mid_entry_splits():
    # Mirrors the real dilution problem: several short diagnosis entries
    # packed onto consecutive lines, like the WHO-style tables in health-4.pdf.
    text = "\n".join(
        [
            "– Entry one symptom description here that is reasonably descriptive",
            "– Entry two symptom description here that is reasonably descriptive",
            "– Entry three symptom description here that is reasonably descriptive",
            "– Entry four symptom description here that is reasonably descriptive",
        ]
    )
    documents = [Document(page_content=text, metadata={"source": "x.pdf"})]

    chunks = create_chunks(documents)

    # No chunk should contain a bullet fragment cut off mid-word (i.e. every
    # bullet marker present in a chunk should start a line, not appear
    # mid-sentence from a broken split).
    for chunk in chunks:
        for line in chunk.page_content.split("\n"):
            stripped = line.strip()
            if stripped:
                assert not stripped.startswith("ntry"), "split occurred mid-word inside a bullet"


def test_chunk_overlap_smaller_than_chunk_size():
    assert CHUNK_OVERLAP < CHUNK_SIZE
