import logging
import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"
VECTORSTORE_METADATA_PATH = DB_FAISS_PATH / "metadata.json"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_HISTORY_TURNS = 6

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("medical_chatbot")


@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str
    model_name: str
    retrieval_k: int
    rerank_k: int
    temperature: float
    enable_reranking: bool


def get_secret(name: str, default: str | None = None) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name, default)


def get_bool_secret(name: str, default: str = "true") -> bool:
    value = str(get_secret(name, default) or default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_config() -> AppConfig | None:
    groq_api_key = get_secret("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Missing GROQ_API_KEY. Add it to .env or Streamlit secrets before starting the app.")
        return None

    retrieval_k = int(get_secret("RETRIEVAL_K", "8") or "8")
    rerank_k = int(get_secret("RERANK_K", "5") or "5")

    return AppConfig(
        groq_api_key=groq_api_key,
        model_name=get_secret("GROQ_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
        retrieval_k=retrieval_k,
        rerank_k=min(rerank_k, retrieval_k),
        temperature=float(get_secret("MODEL_TEMPERATURE", "0.1") or "0.1"),
        enable_reranking=get_bool_secret("ENABLE_RERANKING", "true"),
    )
