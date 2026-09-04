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
    """Stable key for de-duplicating the same chunk returned by two retrievers.

    Content is hashed rather than compared directly so the key stays small, and
    source/page are included because two different pages can legitimately share
    boilerplate text.
    """
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    digest = hashlib.sha1(doc.page_content.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{source}:{page}:{digest}"


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    rrf_k: int = 60,
    top_n: int = 16,
) -> list[Document]:
    """Merge several ranked lists into one using Reciprocal Rank Fusion.

    RRF scores a document as sum(1 / (rrf_k + rank)) across the lists it appears
    in. It is used here instead of blending raw scores because BM25 scores and
    cosine similarities are on completely different scales and are not
    comparable; ranks are. A document found by *both* retrievers accumulates
    score from both and therefore rises, which is exactly the desired behaviour.
    """
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
    """Runs semantic (FAISS/MMR) and keyword (BM25) retrieval, then fuses.

    Semantic search alone misses exact tokens -- drug names, dosages, test
    codes -- because an embedding of "amoxicillin 500 mg" is close to many other
    antibiotic passages. BM25 matches those literally. Neither is reliable
    alone, so both run and RRF merges them; the cross-encoder reranker
    downstream then decides final order.
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
            # BM25 must never be able to take down retrieval; semantic results
            # alone are still a usable answer path.
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
    """Every chunk held in the FAISS index, needed to build the BM25 index.

    LangChain does not expose a public accessor for this, so the in-memory
    docstore is read directly; the guard keeps the failure legible if a future
    LangChain version changes that internal.
    """
    vectorstore = get_vectorstore()
    docstore_dict = getattr(vectorstore.docstore, "_dict", None)
    if not docstore_dict:
        raise RuntimeError(
            "Could not read documents from the FAISS docstore; BM25 hybrid search is unavailable."
        )
    return list(docstore_dict.values())


@lru_cache(maxsize=1)
def get_bm25_retriever() -> BM25Retriever:
    """Keyword index over the same chunks as FAISS.

    Measured on this corpus (22,851 chunks): ~175 MB resident and ~1 s to build,
    so it is built once at startup rather than persisted to disk -- the added
    build complexity would not pay for itself at this scale.
    """
    documents = get_all_documents()
    retriever = BM25Retriever.from_documents(documents)
    LOGGER.info("BM25 index built over %d chunks", len(documents))
    return retriever


@lru_cache(maxsize=1)
def get_index_signature() -> str:
    """Short hash of the vector store metadata.

    Included in answer cache keys so that rebuilding the index (new chunk size,
    new documents, different embedding model) automatically invalidates cached
    answers instead of serving results the current index would not produce.
    """
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
    """Single-prompt LLM completion for short helper tasks.

    Distinct from router.invoke_llm_text, which sends a system/user message
    pair; here the whole instruction is one prompt, which is all the query
    expansion and follow-up helpers need.
    """
    llm = get_llm(config.model_name, config.groq_api_key, temperature, max_tokens)
    return str(llm.invoke(prompt).content).strip()


class QueryExpandingRetriever(BaseRetriever):
    """Rewrites the query, optionally fans out to variations, then fuses results.

    Sits *below* reranking on purpose: the reranker keeps scoring against the
    user's original question, so expansion can only widen the candidate pool,
    never change what relevance is measured against.

    The original query is always retrieved with, alongside any rewrite or
    variation, so a poor rewrite can add noise but can never lose the user's
    actual question.
    """

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
            # Deliberately wider than retrieval_k: hybrid search exists to widen
            # the candidate pool, and the cross-encoder below is what narrows it
            # back down precisely. Passing only retrieval_k here would discard
            # keyword hits before the reranker ever scored them.
            top_n=config.retrieval_k * 2,
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
