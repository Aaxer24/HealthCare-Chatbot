import uuid

import streamlit as st

from src.api_client import call_chat_api
from src.config import LOGGER, get_config
from src.rag import render_source_previews
from src.ui import (
    format_timestamp,
    render_empty_state,
    render_header,
    render_message,
    render_sidebar,
    stream_answer,
)


def initialize_session() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chat_history", [])


def build_message(role: str, content: str, sources: list[dict] | None = None) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "sources": sources or [],
        "timestamp": format_timestamp(),
    }


def trim_history() -> None:
    from src.config import MAX_HISTORY_TURNS

    st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY_TURNS:]


def render_assistant_response(answer: str, sources: list[dict] | None = None) -> None:
    assistant_message = build_message("assistant", answer, sources)
    with st.chat_message("assistant"):
        stream_answer(answer)
        render_source_previews(assistant_message["sources"])
        st.caption(assistant_message["timestamp"])
    st.session_state.messages.append(assistant_message)


def main() -> None:
    render_header()
    initialize_session()
    config = get_config()
    render_sidebar(config)

    for message in st.session_state.messages:
        render_message(message)

    render_empty_state()

    prompt = st.chat_input("Ask about symptoms, conditions, tests, or treatments...")
    if not prompt:
        return

    user_message = build_message("user", prompt)
    st.session_state.messages.append(user_message)
    render_message(user_message)

    if config is None:
        return

    with st.spinner("Thinking..."):
        try:
            result = call_chat_api(prompt, st.session_state.chat_history, config)
            answer = result["answer"]
            sources = result["sources"]
        except Exception as exc:
            LOGGER.exception("Failed to answer prompt")
            st.error(f"Unable to generate an answer right now: {exc}")
            return

    render_assistant_response(answer, sources)
    if result["message_type"] == "MEDICAL_QUESTION":
        st.session_state.chat_history.append((prompt, answer))
        trim_history()


if __name__ == "__main__":
    main()
