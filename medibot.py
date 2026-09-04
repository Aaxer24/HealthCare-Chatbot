import uuid

import streamlit as st

from src.api_client import call_chat_api, submit_feedback
from src.config import LOGGER, get_config
from src.ocr import OCRUnavailable, extract_text_from_image, is_useful, ocr_available
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
    st.session_state.setdefault("pending_prompt", None)  # set by a follow-up button click
    st.session_state.setdefault("rated", {})  # message_id -> "up"/"down", so buttons don't double-fire
    st.session_state.setdefault("document_text", "")  # current OCR'd upload, if any


def build_message(
    role: str,
    content: str,
    sources: list[dict] | None = None,
    follow_ups: list[str] | None = None,
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "sources": sources or [],
        "follow_ups": follow_ups or [],
        "timestamp": format_timestamp(),
    }


def trim_history() -> None:
    from src.config import MAX_HISTORY_TURNS

    st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY_TURNS:]


def render_feedback_controls(message: dict, config) -> None:
    """Thumbs up/down under an assistant answer."""
    message_id = message["id"]
    already = st.session_state.rated.get(message_id)

    if already:
        st.caption("Thanks for the feedback." if already == "up" else "Thanks - we'll use this to improve.")
        return

    question = message.get("question", "")
    left, right, _ = st.columns([1, 1, 8])

    with left:
        if st.button("👍", key=f"up_{message_id}", help="This answer was helpful"):
            submit_feedback(question, message["content"], "up", config,
                            sources=message["sources"], message_type=message.get("message_type", ""))
            st.session_state.rated[message_id] = "up"
            st.rerun()

    with right:
        if st.button("👎", key=f"down_{message_id}", help="This answer was not helpful"):
            submit_feedback(question, message["content"], "down", config,
                            sources=message["sources"], message_type=message.get("message_type", ""))
            st.session_state.rated[message_id] = "down"
            st.rerun()


def render_follow_ups(message: dict) -> None:
    follow_ups = message.get("follow_ups") or []
    if not follow_ups:
        return

    st.caption("You might also ask:")
    for index, suggestion in enumerate(follow_ups):
        if st.button(suggestion, key=f"followup_{message['id']}_{index}"):
            st.session_state.pending_prompt = suggestion
            st.rerun()


def render_document_uploader() -> None:
    """Image upload + OCR, pinned in the sidebar so it's always visible and in
    the same place -- inline in the chat flow it used to scroll out of view as
    the conversation grew. Extracted text attaches to the next question."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Upload a report or prescription**")

        if not ocr_available():
            st.caption("Unavailable: OCR engine not installed on this server.")
            return

        uploaded = st.file_uploader(
            "Photo or scan (optional)",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
            key="document_upload",
            label_visibility="collapsed",
        )

        if uploaded is None:
            st.session_state.document_text = ""
            st.caption("No document attached.")
            return

        try:
            with st.spinner("Reading the document..."):
                text = extract_text_from_image(uploaded.getvalue())
        except (OCRUnavailable, ValueError) as exc:
            st.warning(str(exc))
            st.session_state.document_text = ""
            return

        if not is_useful(text):
            st.warning("Very little text could be read. Try a clearer, well-lit photo.")
            st.session_state.document_text = ""
            return

        st.session_state.document_text = text
        st.success(f"Attached -- {len(text)} characters read. Ask about it below.")
        with st.expander("Show extracted text"):
            st.text(text)
        st.caption("OCR can misread values -- confirm results with your clinician.")


def render_assistant_response(message: dict, config) -> None:
    with st.chat_message("assistant"):
        stream_answer(message["content"])
        render_source_previews(message["sources"])
        st.caption(message["timestamp"])
        render_feedback_controls(message, config)
        render_follow_ups(message)


def handle_prompt(prompt: str, config) -> None:
    user_message = build_message("user", prompt)
    st.session_state.messages.append(user_message)
    render_message(user_message)

    if config is None:
        return

    document_text = st.session_state.get("document_text", "")

    with st.spinner("Thinking..."):
        try:
            result = call_chat_api(prompt, st.session_state.chat_history, config, document_text)
        except Exception as exc:
            LOGGER.exception("Failed to answer prompt")
            st.error(f"Unable to generate an answer right now: {exc}")
            return

    assistant_message = build_message(
        "assistant", result["answer"], result["sources"], result.get("follow_ups")
    )
    # Carried so feedback records what was actually asked, not just the answer.
    assistant_message["question"] = prompt
    assistant_message["message_type"] = result["message_type"]
    st.session_state.messages.append(assistant_message)

    render_assistant_response(assistant_message, config)

    if result["message_type"] == "MEDICAL_QUESTION":
        st.session_state.chat_history.append((prompt, result["answer"]))
        trim_history()


def main() -> None:
    render_header()
    initialize_session()
    config = get_config()
    render_sidebar(config)
    render_document_uploader()

    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                render_source_previews(message["sources"])
                st.caption(message["timestamp"])
                render_feedback_controls(message, config)
                render_follow_ups(message)
        else:
            render_message(message)

    render_empty_state()

    # A clicked follow-up takes priority over the input box on this rerun.
    prompt = st.session_state.pending_prompt or st.chat_input(
        "Ask about symptoms, conditions, tests, or treatments..."
    )
    st.session_state.pending_prompt = None

    if prompt:
        handle_prompt(prompt, config)


if __name__ == "__main__":
    main()
