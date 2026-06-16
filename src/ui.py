import html
import time
from datetime import datetime

import streamlit as st

from src.config import DB_FAISS_PATH, DEFAULT_MODEL, AppConfig
from src.rag import render_source_previews


def format_timestamp() -> str:
    return datetime.now().strftime("%I:%M %p")


def render_header() -> None:
    st.set_page_config(
        page_title="Medical Knowledge Chatbot",
        page_icon="M",
        layout="centered",
    )
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #0b0f14;
            --panel: #121821;
            --panel-border: #243042;
            --text: #eef5ff;
            --muted: #9aa8bb;
            --accent: #38bdf8;
            --accent-2: #34d399;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: var(--app-bg);
            color: var(--text);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"], #MainMenu, footer { display: none; }
        .block-container {
            max-width: 1040px;
            padding: 2.25rem 2rem 6.5rem;
        }
        [data-testid="stSidebar"] > div:first-child {
            background: #10151f;
            border-right: 1px solid #222b3a;
            padding-top: 1.5rem;
        }
        .app-hero {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }
        .app-kicker {
            color: var(--accent-2);
            font-size: .78rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }
        .app-title {
            color: var(--text);
            font-size: 2.15rem;
            line-height: 1.15;
            font-weight: 760;
            margin: 0;
        }
        .app-subtitle {
            color: var(--muted);
            max-width: 680px;
            font-size: 1rem;
            line-height: 1.6;
            margin-top: .55rem;
        }
        .status-pill {
            border: 1px solid rgba(52, 211, 153, .35);
            background: rgba(52, 211, 153, .08);
            color: #b8f7d8;
            border-radius: 999px;
            padding: .42rem .7rem;
            font-size: .86rem;
            white-space: nowrap;
        }
        .safety-note {
            border: 1px solid rgba(56, 189, 248, .32);
            background: linear-gradient(135deg, rgba(56, 189, 248, .10), rgba(52, 211, 153, .08));
            color: #d9ecff;
            border-radius: 8px;
            padding: .85rem 1rem;
            margin: 0 0 1.25rem;
            font-size: .95rem;
            line-height: 1.6;
        }
        .empty-state {
            border: 1px solid var(--panel-border);
            background: var(--panel);
            border-radius: 8px;
            padding: 1rem;
            color: var(--muted);
            margin-top: 1rem;
        }
        .sidebar-brand {
            border-bottom: 1px solid #253044;
            padding-bottom: 1rem;
            margin-bottom: 1rem;
        }
        .sidebar-title {
            color: var(--text);
            font-weight: 760;
            font-size: 1.15rem;
        }
        .sidebar-caption {
            color: var(--muted);
            font-size: .88rem;
            margin-top: .2rem;
        }
        .metric-card {
            border: 1px solid #253044;
            background: #151c28;
            border-radius: 8px;
            padding: .75rem;
            margin-bottom: .65rem;
        }
        .metric-label {
            color: var(--muted);
            font-size: .78rem;
            margin-bottom: .2rem;
        }
        .metric-value {
            color: var(--text);
            font-size: .95rem;
            font-weight: 650;
            overflow-wrap: anywhere;
        }
        [data-testid="stChatMessage"] {
            border: 1px solid var(--panel-border);
            background: var(--panel);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: .85rem 0;
            box-shadow: 0 12px 30px rgba(0, 0, 0, .16);
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stMarkdownContainer"] p {
            font-size: 1rem;
            line-height: 1.65;
        }
        [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] {
            color: #7f8da3;
        }
        [data-testid="stExpander"] {
            border: 1px solid #29364a;
            background: #101722;
            border-radius: 8px;
        }
        .stButton > button {
            border-radius: 8px;
            border: 1px solid #334155;
            background: #172033;
            color: var(--text);
            font-weight: 650;
            min-height: 2.7rem;
        }
        .stButton > button:hover {
            border-color: var(--accent);
            color: #dff7ff;
            background: #1a2940;
        }
        [data-testid="stChatInput"] {
            border-radius: 999px;
            border: 1px solid #334155;
            background: #151b26;
            box-shadow: 0 12px 30px rgba(0, 0, 0, .28);
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, .12);
        }
        .streaming-answer {
            white-space: pre-wrap;
            font-size: 1rem;
            line-height: 1.6;
            font-weight: 400;
        }
        .typing-cursor { opacity: .85; }
        @media (max-width: 760px) {
            .block-container { padding: 1.2rem 1rem 6rem; }
            .app-hero { display: block; }
            .status-pill {
                display: inline-block;
                margin-top: .75rem;
            }
            .app-title { font-size: 1.65rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="app-hero">
            <div>
                <div class="app-kicker">Clinical document assistant</div>
                <h1 class="app-title">Medical Knowledge Chatbot</h1>
                <div class="app-subtitle">
                    Ask grounded healthcare questions and get concise answers with document citations.
                </div>
            </div>
            <div class="status-pill">Knowledge base ready</div>
        </div>
        """,
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
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-title">MedAssist RAG</div>
                <div class="sidebar-caption">Local FAISS retrieval with Groq responses</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        vector_status = "Ready" if DB_FAISS_PATH.exists() else "Missing"
        model_name = config.model_name if config else DEFAULT_MODEL
        retrieval_k = config.retrieval_k if config else 5
        rerank_status = "On" if config and config.enable_reranking else "Off"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Vector store</div>
                <div class="metric-value">{vector_status}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Model</div>
                <div class="metric-value">{html.escape(model_name)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Retrieval depth</div>
                <div class="metric-value">{retrieval_k} chunks</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Reranking</div>
                <div class="metric-value">{rerank_status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()


def render_empty_state() -> None:
    if st.session_state.messages:
        return
    st.markdown(
        """
        <div class="empty-state">
        Try asking about symptoms, conditions, tests, treatment options, prevention, or general health concepts.
        For deeper answers, ask the assistant to explain in detail.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_source_previews(message.get("sources", []))
        st.caption(message.get("timestamp", ""))


def stream_answer(answer: str) -> None:
    placeholder = st.empty()
    streamed_answer = ""
    for word in answer.split():
        streamed_answer += word + " "
        safe_answer = html.escape(streamed_answer)
        placeholder.markdown(
            f'<div class="streaming-answer">{safe_answer}<span class="typing-cursor">|</span></div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.035)
    placeholder.markdown(answer)
