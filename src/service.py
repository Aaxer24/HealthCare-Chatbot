import hashlib

from src.cache import get_answer_cache, log_cache_event, make_cache_key
from src.config import AppConfig
from src.followups import generate_follow_ups
from src.router import (
    answer_general_chat,
    answer_out_of_scope,
    answer_style_instruction,
    classify_message,
)
from src.rag import answer_question, get_index_signature


def _cache_key_for(
    prompt: str,
    chat_history: list[tuple[str, str]],
    config: AppConfig,
    document_text: str = "",
) -> str:
    # hash the doc into the key so two different reports never share a cached answer
    document_signature = (
        hashlib.sha256(document_text.strip().encode("utf-8")).hexdigest()[:16]
        if document_text.strip()
        else ""
    )
    return make_cache_key(
        prompt,
        chat_history,
        model_name=config.model_name,
        retrieval_k=config.retrieval_k,
        rerank_k=config.rerank_k,
        enable_reranking=config.enable_reranking,
        index_signature=f"{get_index_signature()}:{document_signature}",
    )


def build_question_with_document(prompt: str, document_text: str) -> str:
    """Wraps the doc text in a clear block so the model can tell it apart
    from the retrieved reference material."""
    if not document_text.strip():
        return prompt
    return (
        "UPLOADED DOCUMENT (text extracted from the user's file):\n"
        "---\n"
        f"{document_text.strip()}\n"
        "---\n\n"
        f"Their question: {prompt}"
    )


def generate_chat_response(
    prompt: str,
    chat_history: list[tuple[str, str]],
    config: AppConfig,
    document_text: str = "",
) -> dict:
    # cache hit = zero LLM calls for this turn, our biggest lever against the token quota
    cache = get_answer_cache(config.cache_max_size, config.cache_ttl_seconds) if config.enable_cache else None
    cache_key = ""
    if cache is not None:
        cache_key = _cache_key_for(prompt, chat_history, config, document_text)
        cached = cache.get(cache_key)
        if cached is not None:
            log_cache_event(True, cache_key)
            return {**cached, "cached": True}
        log_cache_event(False, cache_key)

    if document_text.strip():
        # an uploaded report IS the signal this is medical -- classifying "what
        # does this mean?" alone would deflect it as OUT_OF_SCOPE
        message_type = "MEDICAL_QUESTION"
    else:
        message_type = classify_message(prompt, chat_history, config)

    if message_type == "GENERAL_CHAT":
        result = {
            "message_type": message_type,
            "answer": answer_general_chat(prompt, config),
            "sources": [],
            "follow_ups": [],
        }
    elif message_type == "OUT_OF_SCOPE":
        result = {
            "message_type": message_type,
            "answer": answer_out_of_scope(prompt, config),
            "sources": [],
            "follow_ups": [],
        }
    else:
        style_instruction = answer_style_instruction(prompt, chat_history, config)
        question = build_question_with_document(prompt, document_text)
        question = f"{question}\n\nResponse style instruction: {style_instruction}"
        answer, sources = answer_question(question, chat_history, config)
        result = {
            "message_type": message_type,
            "answer": answer,
            "sources": sources,
            "follow_ups": generate_follow_ups(prompt, answer, config),
        }

    if cache is not None:
        cache.set(cache_key, result)

    return {**result, "cached": False}
