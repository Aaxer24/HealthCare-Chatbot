import html
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv(find_dotenv())
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_HISTORY_TURNS = 6
CASUAL_RESPONSES = {
    "hi": "Hi! How can I help you today?",
    "hii": "Hi! How can I help you today?",
    "hello": "Hello! What would you like to ask?",
    "hey": "Hey! How can I help?",
    "good morning": "Good morning! How can I help you today?",
    "good afternoon": "Good afternoon! How can I help you today?",
    "good evening": "Good evening! How can I help you today?",
    "how are you": "I'm doing well, thanks for asking. How can I help you today?",
    "how are you doing": "I'm doing well, thanks for asking. What would you like to know?",
    "how r u": "I'm doing well, thanks for asking. How can I help?",
    "how are u": "I'm doing well, thanks for asking. How can I help?",
    "what are you doing": "I'm here and ready to help with health-related questions or general conversation.",
    "who are you": "I'm a medical knowledge chatbot that can answer health-related questions from trusted documents and provide sources.",
    "what can you do": "I can answer health-related questions using the medical documents in this project, provide citations, and also handle simple general conversation.",
    "thanks": "You're welcome.",
    "thank you": "You're welcome.",
    "ok": "Okay. Let me know what you would like to ask.",
    "okay": "Okay. Let me know what you would like to ask.",
    "bye": "Take care.",
    "goodbye": "Take care.",
}

SYSTEM_PROMPT = """
You are a careful medical information assistant for educational support.
Use only the supplied context and the chat history to answer the user's question.

Safety rules:
- Do not diagnose, prescribe medication, or replace a licensed clinician.
- If the user describes urgent warning signs such as chest pain, severe breathing trouble,
  stroke symptoms, severe allergic reaction, suicidal thoughts, overdose, uncontrolled
  bleeding, or loss of consciousness, advise urgent local emergency care immediately.
- If the context is insufficient, say what is missing and ask a concise follow-up question.
- If the answer is not supported by the context, say you do not know from the documents.
- Explain uncertainty clearly and avoid overstating confidence.
- Include practical next steps and when to seek professional care when relevant.

Context:
{context}

Question:
{question}

Answer with these sections:
1. Short answer
2. What the documents say
3. What to do next
"""

CONDENSE_QUESTION_PROMPT = """
Given the chat history and a follow-up question, rewrite the follow-up into a standalone
medical information question. Preserve important details such as age, symptoms, duration,
severity, medicines, and conditions. Do not answer the question.

Chat history:
{chat_history}

Follow-up question:
{question}

Standalone question:
"""


@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str
    model_name: str
    retrieval_k: int
    temperature: float


def get_secret(name: str, default: str | None = None) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name, default)


def get_config() -> AppConfig | None:
    groq_api_key = get_secret("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Missing GROQ_API_KEY. Add it to .env or Streamlit secrets before starting the app.")
        return None

    return AppConfig(
        groq_api_key=groq_api_key,
        model_name=get_secret("GROQ_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
        retrieval_k=int(get_secret("RETRIEVAL_K", "5") or "5"),
        temperature=float(get_secret("MODEL_TEMPERATURE", "0.1") or "0.1"),
    )


@st.cache_resource(show_spinner=False)
def get_vectorstore() -> FAISS:
    if not DB_FAISS_PATH.exists():
        raise FileNotFoundError(
            f"Vector store not found at {DB_FAISS_PATH}. Run create_memory_for_llm.py first."
        )

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(DB_FAISS_PATH),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, groq_api_key: str, temperature: float) -> ChatGroq:
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        groq_api_key=groq_api_key,
        timeout=30,
        max_retries=2,
    )


def build_chain(config: AppConfig) -> ConversationalRetrievalChain:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.retrieval_k, "fetch_k": max(config.retrieval_k * 4, 12)},
    )
    llm = get_llm(config.model_name, config.groq_api_key, config.temperature)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
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


def unique_sources(documents: Iterable[Document]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for doc in documents:
        source = format_source(doc)
        if source not in seen:
            sources.append(source)
            seen.add(source)
    return sources


def trim_history() -> None:
    st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY_TURNS:]


def format_timestamp() -> str:
    return datetime.now().strftime("%I:%M %p")


def render_header() -> None:
    st.set_page_config(
        page_title="Medical Knowledge Chatbot",
        page_icon="medical_symbol",
        layout="centered",
    )
    st.markdown(
        """
        <style>
        .block-container { max-width: 920px; padding-top: 1.5rem; }
        .app-title { font-size: 2rem; font-weight: 700; margin-bottom: .25rem; }
        .app-subtitle { color: #4b5563; margin-bottom: 1rem; }
        .safety-note {
            border: 1px solid #bfd7ea;
            background: #f4faff;
            color: #123047;
            border-radius: 8px;
            padding: .8rem 1rem;
            margin-bottom: 1rem;
            font-size: .95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="app-title">Medical Knowledge Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">RAG-powered answers from your curated medical PDFs, with citations.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="safety-note">
        This assistant provides educational information from uploaded documents. It is not a diagnosis
        or a substitute for a clinician. For severe or rapidly worsening symptoms, seek local emergency care.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(config: AppConfig | None) -> None:
    with st.sidebar:
        st.header("System")
        st.caption("Production checks")
        st.write(f"Vector store: {'ready' if DB_FAISS_PATH.exists() else 'missing'}")
        st.write(f"Model: `{config.model_name if config else DEFAULT_MODEL}`")
        st.write(f"Retrieval depth: `{config.retrieval_k if config else 5}`")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))
        st.caption(message.get("timestamp", ""))


def render_sources(sources: list[str]) -> None:
    cleaned_sources = [source for source in sources if source and source != "Unknown source"]
    if not cleaned_sources:
        return

    with st.expander("Sources", expanded=False):
        for source in cleaned_sources:
            st.markdown(f"- {html.escape(source)}")


def stream_answer(answer: str) -> None:
    placeholder = st.empty()
    streamed_answer = ""
    for word in answer.split():
        streamed_answer += word + " "
        placeholder.markdown(streamed_answer + "▌")
        time.sleep(0.035)
    placeholder.markdown(answer)


def initialize_session() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chat_history", [])


def casual_response(prompt: str) -> str | None:
    normalized = " ".join(prompt.lower().strip().split())
    normalized = normalized.rstrip(".!?")
    if normalized in CASUAL_RESPONSES:
        return CASUAL_RESPONSES[normalized]

    greeting_words = {"hi", "hii", "hello", "hey", "heyy"}
    thanks_words = {"thanks", "thank", "thx"}
    tokens = set(normalized.replace(",", " ").split())

    if tokens & greeting_words and len(tokens) <= 4:
        return "Hi! How can I help you today?"
    if "how are you" in normalized or "how are u" in normalized or "how r u" in normalized:
        return "I'm doing well, thanks for asking. How can I help you today?"
    if "what can you do" in normalized or "help me" == normalized:
        return "I can answer health-related questions using the medical documents in this project, provide citations, and also handle simple general conversation."
    if tokens & thanks_words and len(tokens) <= 5:
        return "You're welcome."
    if normalized in {"bro", "broo"}:
        return "Yes bro, tell me what you want to ask."

    return None


def answer_question(prompt: str, config: AppConfig) -> tuple[str, list[str]]:
    chain = build_chain(config)
    response = chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
    answer = response.get("answer", "").strip()
    sources = unique_sources(response.get("source_documents", []))
    return answer, sources


def main() -> None:
    render_header()
    initialize_session()
    config = get_config()
    render_sidebar(config)

    for message in st.session_state.messages:
        render_message(message)

    prompt = st.chat_input("Ask about symptoms, conditions, tests, or treatments...")
    if not prompt:
        return

    user_message = {"role": "user", "content": prompt, "timestamp": format_timestamp()}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    casual_answer = casual_response(prompt)
    if casual_answer:
        assistant_message = {
            "role": "assistant",
            "content": casual_answer,
            "sources": [],
            "timestamp": format_timestamp(),
        }
        st.session_state.messages.append(assistant_message)
        with st.chat_message("assistant"):
            stream_answer(casual_answer)
            st.caption(assistant_message["timestamp"])
        return

    if config is None:
        return

    with st.chat_message("assistant"):
        with st.spinner("Searching trusted documents..."):
            try:
                answer, sources = answer_question(prompt, config)
            except Exception as exc:
                LOGGER.exception("Failed to answer question")
                st.error(f"Unable to generate an answer right now: {exc}")
                return

        stream_answer(answer)
        render_sources(sources)
        timestamp = format_timestamp()
        st.caption(timestamp)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "timestamp": timestamp}
    )
    st.session_state.chat_history.append((prompt, answer))
    trim_history()


if __name__ == "__main__":
    main()
