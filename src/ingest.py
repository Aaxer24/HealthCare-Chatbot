import json
import re
import unicodedata
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import DATA_PATH, DB_FAISS_PATH, EMBEDDING_MODEL

# Smaller than a typical prose chunk size on purpose: several source PDFs (e.g.
# WHO-style clinical handbooks) pack many short, unrelated entries per page
# (differential-diagnosis lists, symptom tables). A large chunk_size merges
# several unrelated conditions into one embedding, diluting it so a specific
# query (e.g. "asthma symptoms") doesn't rank the chunk highly even though the
# text is present. A smaller chunk keeps each entry's embedding focused.
CHUNK_SIZE = 450
CHUNK_OVERLAP = 80


def load_pdf_files(data_path: Path) -> list[Document]:
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    pdf_files = sorted(data_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_path}")

    loader = DirectoryLoader(
        str(data_path),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    return loader.load()


# PDF text extraction leaves artefacts that silently break retrieval. Measured
# on this corpus (22,851 chunks) before normalisation:
#   - 31.2% of chunks had a word hyphenated across a line break
#   - 12.5% had runs of 2+ spaces from justified typesetting
#   -  8.1% contained ligature glyphs
# Concretely, "classification" appeared cleanly in 30 chunks but as the broken
# ligature form in another 13 -- roughly a third of its occurrences were
# invisible to exact keyword (BM25) search and diluted in the embedding.
LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}


def normalize_pdf_text(text: str) -> str:
    """Repair common PDF extraction artefacts before chunking.

    Applied at ingestion rather than query time because the index must contain
    clean text: normalising the query alone cannot match a document that stored
    "speciﬁ c" as a ligature.
    """
    if not text:
        return text

    # These PDFs frequently render a ligature then insert spurious spacing before
    # the rest of the word: "ﬂ  uid", "speciﬁ  c", "difﬁ  culty". Sampling the
    # corpus found 199 distinct occurrences and every frequent one was
    # intra-word ("fluid" 202x, "first" 143x, "specific" 52x, "deficiency",
    # "inflammatory"), so the gap is closed whenever a lowercase continuation
    # follows. Done before the plain replacement below, which handles the rest.
    text = re.sub(
        r"([" + "".join(LIGATURES) + r"])[^\S\n]{1,3}(?=[a-z])",
        lambda match: LIGATURES[match.group(1)],
        text,
    )

    for ligature, replacement in LIGATURES.items():
        text = text.replace(ligature, replacement)

    # Compatibility-normalise anything else exotic (odd spaces, sub/superscripts)
    # into plain ASCII-ish equivalents where a mapping exists.
    text = unicodedata.normalize("NFKC", text)

    # Re-join words hyphenated across a line break: "enven-\noming" -> "envenoming".
    # Requires letters on both sides so genuine hyphenated terms spanning a
    # line ("X-\nray") still join correctly, while a trailing dash before a
    # number or bullet is left alone.
    text = re.sub(r"([A-Za-z])-\s*\n\s*([a-z])", r"\1\2", text)

    # Collapse the wide inter-word gaps justified typesetting produces, but
    # never across newlines -- line structure carries meaning for lists and
    # table rows, and the chunk splitter relies on it.
    text = re.sub(r"[^\S\n]{2,}", " ", text)

    # Trim trailing spaces left on each line by the collapse above.
    text = re.sub(r"[^\S\n]+\n", "\n", text)

    return text


def normalize_documents(documents: list[Document]) -> list[Document]:
    for document in documents:
        document.page_content = normalize_pdf_text(document.page_content)
    return documents


def create_chunks(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # "\n" before the bullet markers so list items break on their own
        # lines first; the bullet markers as a fallback catch bullets that
        # PDF extraction ran together on one line (common in dense tables).
        separators=["\n\n", "\n", "– ", "• ", "■ ", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        if "source" in chunk.metadata:
            chunk.metadata["source"] = Path(str(chunk.metadata["source"])).name

    return chunks


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(data_path: Path = DATA_PATH, db_path: Path = DB_FAISS_PATH) -> dict:
    """Load PDFs, chunk them, embed, and persist a FAISS index. Returns the metadata dict written."""
    documents = load_pdf_files(data_path)
    print(f"Loaded {len(documents)} PDF pages from {data_path}")

    documents = normalize_documents(documents)
    print("Normalized PDF text (ligatures, hyphenation, justified spacing)")

    text_chunks = create_chunks(documents)
    print(f"Created {len(text_chunks)} text chunks")

    embedding_model = get_embedding_model()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(str(db_path))

    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "document_count": len(documents),
        "chunk_count": len(text_chunks),
        # Bumped whenever the text pipeline changes, so a stale index built by
        # an older normaliser is identifiable from metadata alone.
        "text_normalization": "v1-ligatures-dehyphen-spaces",
    }
    (db_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved FAISS vector store at {db_path}")
    return metadata
