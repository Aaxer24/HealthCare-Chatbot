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
DEFAULT_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_HISTORY_TURNS = 6

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("medical_chatbot")

# LangChain reads LANGCHAIN_TRACING_V2/API_KEY/PROJECT itself, just set a default project name
os.environ.setdefault("LANGCHAIN_PROJECT", "healthcare-chatbot")
if os.getenv("LANGCHAIN_TRACING_V2", "").lower() in {"1", "true", "yes", "on"} and not os.getenv("LANGCHAIN_API_KEY"):
    LOGGER.warning("LANGCHAIN_TRACING_V2 is enabled but LANGCHAIN_API_KEY is not set; LangSmith tracing will fail silently.")
elif os.getenv("LANGCHAIN_TRACING_V2", "").lower() in {"1", "true", "yes", "on"}:
    LOGGER.info("LangSmith tracing enabled for project '%s'.", os.environ["LANGCHAIN_PROJECT"])


@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str
    model_name: str
    retrieval_k: int
    rerank_k: int
    temperature: float
    enable_reranking: bool
    api_base_url: str
    enable_cache: bool = True
    cache_ttl_seconds: int = 86_400
    cache_max_size: int = 512
    enable_hybrid_search: bool = True
    # query rewriting, multi-query and follow-ups each cost an extra Groq call --
    # can toggle off individually if the daily quota gets tight
    enable_query_rewriting: bool = True
    multi_query_count: int = 2
    enable_follow_ups: bool = True


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
        api_base_url=(get_secret("API_BASE_URL", "http://127.0.0.1:8000") or "http://127.0.0.1:8000").rstrip("/"),
        enable_cache=get_bool_secret("ENABLE_CACHE", "true"),
        cache_ttl_seconds=int(get_secret("CACHE_TTL_SECONDS", "86400") or "86400"),
        cache_max_size=int(get_secret("CACHE_MAX_SIZE", "512") or "512"),
        enable_hybrid_search=get_bool_secret("ENABLE_HYBRID_SEARCH", "true"),
        enable_query_rewriting=get_bool_secret("ENABLE_QUERY_REWRITING", "true"),
        multi_query_count=int(get_secret("MULTI_QUERY_COUNT", "2") or "2"),
        enable_follow_ups=get_bool_secret("ENABLE_FOLLOW_UPS", "true"),
    )


def get_cli_config() -> AppConfig:
    """Config loader for CLI/non-Streamlit entrypoints (ingestion, evaluation).

    Reads only from the environment/.env (no st.secrets), and raises instead
    of rendering a Streamlit error, since there is no Streamlit runtime here.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it to .env before running this script.")

    retrieval_k = int(os.getenv("RETRIEVAL_K", "8"))
    rerank_k = int(os.getenv("RERANK_K", "5"))
    return AppConfig(
        groq_api_key=groq_api_key,
        model_name=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        retrieval_k=retrieval_k,
        rerank_k=min(rerank_k, retrieval_k),
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0.1")),
        enable_reranking=os.getenv("ENABLE_RERANKING", "true").lower() in {"1", "true", "yes", "on"},
        api_base_url=os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
        enable_cache=os.getenv("ENABLE_CACHE", "true").lower() in {"1", "true", "yes", "on"},
        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "86400")),
        cache_max_size=int(os.getenv("CACHE_MAX_SIZE", "512")),
        enable_hybrid_search=os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() in {"1", "true", "yes", "on"},
        enable_query_rewriting=os.getenv("ENABLE_QUERY_REWRITING", "true").lower() in {"1", "true", "yes", "on"},
        multi_query_count=int(os.getenv("MULTI_QUERY_COUNT", "2")),
        enable_follow_ups=os.getenv("ENABLE_FOLLOW_UPS", "true").lower() in {"1", "true", "yes", "on"},
    )
