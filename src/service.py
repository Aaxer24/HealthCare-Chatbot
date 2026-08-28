from src.config import AppConfig
from src.router import (
    answer_general_chat,
    answer_out_of_scope,
    answer_style_instruction,
    classify_message,
)
from src.rag import answer_question


def generate_chat_response(
    prompt: str,
    chat_history: list[tuple[str, str]],
    config: AppConfig,
) -> dict:
    message_type = classify_message(prompt, chat_history, config)

    if message_type == "GENERAL_CHAT":
        answer = answer_general_chat(prompt, config)
        return {
            "message_type": message_type,
            "answer": answer,
            "sources": [],
        }

    if message_type == "OUT_OF_SCOPE":
        answer = answer_out_of_scope(prompt, config)
        return {
            "message_type": message_type,
            "answer": answer,
            "sources": [],
        }

    style_instruction = answer_style_instruction(prompt, chat_history, config)
    question = f"{prompt}\n\nResponse style instruction: {style_instruction}"
    answer, sources = answer_question(question, chat_history, config)
    return {
        "message_type": message_type,
        "answer": answer,
        "sources": sources,
    }
