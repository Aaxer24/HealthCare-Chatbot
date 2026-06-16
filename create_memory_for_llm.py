from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdf_files(data_path: Path):
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


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        if "source" in chunk.metadata:
            chunk.metadata["source"] = Path(str(chunk.metadata["source"])).name

    return chunks


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def main():
    documents = load_pdf_files(DATA_PATH)
    print(f"Loaded {len(documents)} PDF pages from {DATA_PATH}")

    text_chunks = create_chunks(documents)
    print(f"Created {len(text_chunks)} text chunks")

    embedding_model = get_embedding_model()
    DB_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)

    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(str(DB_FAISS_PATH))
    print(f"Saved FAISS vector store at {DB_FAISS_PATH}")


if __name__ == "__main__":
    main()
