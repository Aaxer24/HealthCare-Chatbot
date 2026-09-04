import hashlib
import json
import html
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import ConfigDict
from sentence_transformers import CrossEncoder

from src.config import (
    DB_FAISS_PATH,
    EMBEDDING_MODEL,
    LOGGER,
    RERANKER_MODEL,
    VECTORSTORE_METADATA_PATH,
    AppConfig,
)
from src.prompts import CONDENSE_QUESTION_PROMPT, SYSTEM_PROMPT


def document_identity(doc: Document) -> str:
    """Key for de-duping the same chunk across retrievers. Hashed content +
    source/page, since two pages can share boilerplate text."""
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    digest = hashlib.sha1(doc.page_content.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{source}:{page}:{digest}"


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    rrf_k: int = 60,
    top_n: int = 16,
) -> list[Document]:
    """Merges ranked lists via RRF: score = sum(1 / (rrf_k + rank)).
    Using ranks instead of raw scores because BM25 and cosine similarity
    aren't on comparable scales. A doc found by both lists ranks higher."""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = document_identity(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            documents.setdefault(key, doc)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [documents[key] for key, _ in ranked[:top_n]]


class HybridRetriever(BaseRetriever):
    """Semantic (FAISS/MMR) + keyword (BM25) retrieval, fused with RRF.

    Semantic search alone misses exact terms like drug names or dosages --
    BM25 catches those. Reranker downstream sorts out the final order.
    """

    semantic_retriever: BaseRetriever
    keyword_retriever: BaseRetriever
    top_n: int = 16
    rrf_k: int = 60
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        semantic_docs = self.semantic_retriever.invoke(query)
        try:
            keyword_docs = self.keyword_retriever.invoke(query)
        except Exception:
            # never let BM25 take retrieval down with it
            LOGGER.warning("BM25 retrieval failed; falling back to semantic only", exc_info=True)
            keyword_docs = []

        return reciprocal_rank_fusion([semantic_docs, keyword_docs], self.rrf_k, self.top_n)


class RerankingRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    reranker: CrossEncoder
    top_n: int = 5
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        documents = self.base_retriever.invoke(query)
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda item: float(item[1]), reverse=True)

        reranked_docs: list[Document] = []
        for doc, score in ranked[: self.top_n]:
            doc.metadata["rerank_score"] = float(score)
            reranked_docs.append(doc)
        return reranked_docs


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    if not DB_FAISS_PATH.exists():
        raise FileNotFoundError(
            f"Vector store not found at {DB_FAISS_PATH}. Run create_memory_for_llm.py first."
        )
    if not VECTORSTORE_METADATA_PATH.exists():
        raise RuntimeError(
            "Vector store metadata is missing. Rebuild the index with create_memory_for_llm.py."
        )

    metadata = json.loads(VECTORSTORE_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("embedding_model") != EMBEDDING_MODEL:
        raise RuntimeError(
            "Vector store was built with a different embedding model. "
            "Run create_memory_for_llm.py to rebuild it."
        )

    return FAISS.load_local(
        str(DB_FAISS_PATH),
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


@lru_cache(maxsize=1)
def get_all_documents() -> list[Document]:
    """All chunks in the FAISS index -- LangChain has no public accessor for
    this, so we read the in-memory docstore directly."""
    vectorstore = get_vectorstore()
    docstore_dict = getattr(vectorstore.docstore, "_dict", None)
    if not docstore_dict:
        raise RuntimeError(
            "Could not read documents from the FAISS docstore; BM25 hybrid search is unavailable."
        )
    return list(docstore_dict.values())


@lru_cache(maxsize=1)
def get_bm25_retriever() -> BM25Retriever:
    """~175MB and ~1s to build on this corpus, so just built at startup
    instead of persisted to disk."""
    documents = get_all_documents()
    retriever = BM25Retriever.from_documents(documents)
    LOGGER.info("BM25 index built over %d chunks", len(documents))
    return retriever


@lru_cache(maxsize=1)
def get_index_signature() -> str:
    """Hash of the index metadata, used in cache keys so a rebuild
    invalidates old cached answers automatically."""
    if not VECTORSTORE_METADATA_PATH.exists():
        return "no-index"
    raw = VECTORSTORE_METADATA_PATH.read_text(encoding="utf-8")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=8)
def get_llm(model_name: str, groq_api_key: str, temperature: float, max_tokens: int | None = None) -> ChatGroq:
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        groq_api_key=groq_api_key,
        timeout=30,
        max_retries=2,
        max_tokens=max_tokens,
    )


def complete_text(
    config: AppConfig,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """Single-prompt completion for short helper tasks (query rewriting,
    follow-ups) -- simpler than router.invoke_llm_text's system/user pair."""
    llm = get_llm(config.model_name, config.groq_api_key, temperature, max_tokens)
    return str(llm.invoke(prompt).content).strip()


class QueryExpandingRetriever(BaseRetriever):
    """Rewrites the query, optionally fans out to variations, then fuses.
    Always retrieves with the original query too -- a bad rewrite can only
    add noise, never lose the user's real question."""

    inner_retriever: BaseRetriever
    config: AppConfig
    rewrite: bool = True
    variation_count: int = 0
    top_n: int = 16
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        from src.query_expansion import generate_query_variations, rewrite_query

        queries = [query]

        if self.rewrite:
            rewritten = rewrite_query(query, self.config)
            if rewritten.strip().lower() != query.strip().lower():
                queries.append(rewritten)

        if self.variation_count > 0:
            queries.extend(generate_query_variations(query, self.config, self.variation_count))

        if len(queries) == 1:
            return self.inner_retriever.invoke(query)

        result_lists = []
        for candidate in queries:
            try:
                result_lists.append(self.inner_retriever.invoke(candidate))
            except Exception:
                LOGGER.warning("Retrieval failed for expanded query %r", candidate, exc_info=True)

        if not result_lists:
            return []
        return reciprocal_rank_fusion(result_lists, top_n=self.top_n)


def build_retriever(config: AppConfig) -> BaseRetriever:
    vectorstore = get_vectorstore()
    base_retriever: BaseRetriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.retrieval_k, "fetch_k": max(config.retrieval_k * 4, 20)},
    )

    if config.enable_hybrid_search:
        keyword_retriever = get_bm25_retriever()
        keyword_retriever.k = config.retrieval_k
        base_retriever = HybridRetriever(
            semantic_retriever=base_retriever,
            keyword_retriever=keyword_retriever,
            top_n=config.retrieval_k * 2,  # wider pool; reranker narrows it back down
        )

    if config.enable_query_rewriting or config.multi_query_count > 0:
        base_retriever = QueryExpandingRetriever(
            inner_retriever=base_retriever,
            config=config,
            rewrite=config.enable_query_rewriting,
            variation_count=config.multi_query_count,
            top_n=config.retrieval_k * 2,
        )

    if not config.enable_reranking:
        return base_retriever

    return RerankingRetriever(
        base_retriever=base_retriever,
        reranker=get_reranker(),
        top_n=config.rerank_k,
    )


def build_chain(config: AppConfig) -> ConversationalRetrievalChain:
    llm = get_llm(config.model_name, config.groq_api_key, config.temperature)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=build_retriever(config),
        return_source_documents=True,
        output_key="answer",
        condense_question_prompt=PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT),
        combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(SYSTEM_PROMPT)},
    )


def format_source(doc: Document) -> str:
    source = Path(str(doc.metadata.get("source", "Unknown source"))).name
    page = doc.metadata.get("page")
    page_label = f", page {int(page) + 1}" if isinstance(page, int) else ""
    return f"{source}{page_label}"


def clean_snippet(text: str, limit: int = 520) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rsplit(" ", 1)[0] + "..."


def source_previews(documents: Iterable[Document]) -> list[dict[str, str]]:
    previews: list[dict[str, str]] = []
    seen: set[str] = set()
    for doc in documents:
        source = format_source(doc)
        snippet = clean_snippet(doc.page_content)
        key = f"{source}:{snippet[:80]}"
        if source == "Unknown source" or key in seen:
            continue
        previews.append(
            {
                "source": source,
                "snippet": snippet,
                "score": f"{doc.metadata.get('rerank_score'):.2f}"
                if isinstance(doc.metadata.get("rerank_score"), float)
                else "",
            }
        )
        seen.add(key)
    return previews


def render_source_previews(previews: list[dict[str, str]]) -> None:
    if not previews:
        return

    with st.expander("Reference documents", expanded=False):
        for preview in previews:
            score = f" | relevance {preview['score']}" if preview.get("score") else ""
            st.markdown(f"**{html.escape(preview['source'])}**{score}")
            st.caption(html.escape(preview["snippet"]))


def answer_question(prompt: str, chat_history: list[tuple[str, str]], config: AppConfig) -> tuple[str, list[dict[str, str]]]:
    chain = build_chain(config)
    response = chain.invoke({"question": prompt, "chat_history": chat_history})
    answer = response.get("answer", "").strip()
    previews = source_previews(response.get("source_documents", []))
    return answer, previews
