import json
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
    }
    (db_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved FAISS vector store at {db_path}")
    return metadata
