import json
import html
from pathlib import Path
from typing import Iterable

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
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
    RERANKER_MODEL,
    VECTORSTORE_METADATA_PATH,
    AppConfig,
)
from src.prompts import CONDENSE_QUESTION_PROMPT, SYSTEM_PROMPT


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


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
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


@st.cache_resource(show_spinner=False)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, groq_api_key: str, temperature: float) -> ChatGroq:
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        groq_api_key=groq_api_key,
        timeout=30,
        max_retries=2,
    )


def build_retriever(config: AppConfig) -> BaseRetriever:
    vectorstore = get_vectorstore()
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.retrieval_k, "fetch_k": max(config.retrieval_k * 4, 20)},
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
