from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config import AppConfig
from src.prompts import (
    ANSWER_STYLE_PROMPT,
    GENERAL_CLASSIFIER_PROMPT,
    GENERAL_RESPONSE_PROMPT,
    OUT_OF_SCOPE_RESPONSE_PROMPT,
)
from src.rag import get_llm


def invoke_llm_text(llm: ChatGroq, system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content).strip()


def classify_message(prompt: str, config: AppConfig) -> str:
    llm = get_llm(config.model_name, config.groq_api_key, 0.0)
    label = invoke_llm_text(llm, GENERAL_CLASSIFIER_PROMPT, prompt).upper()
    if "OUT_OF_SCOPE" in label:
        return "OUT_OF_SCOPE"
    if "GENERAL_CHAT" in label and "MEDICAL_QUESTION" not in label:
        return "GENERAL_CHAT"
    return "MEDICAL_QUESTION"


def answer_general_chat(prompt: str, config: AppConfig) -> str:
    llm = get_llm(config.model_name, config.groq_api_key, 0.2)
    return invoke_llm_text(llm, GENERAL_RESPONSE_PROMPT, prompt)


def answer_out_of_scope(prompt: str, config: AppConfig) -> str:
    llm = get_llm(config.model_name, config.groq_api_key, 0.1)
    return invoke_llm_text(llm, OUT_OF_SCOPE_RESPONSE_PROMPT, prompt)


def answer_style_instruction(prompt: str, config: AppConfig) -> str:
    llm = get_llm(config.model_name, config.groq_api_key, 0.0)
    style = invoke_llm_text(llm, ANSWER_STYLE_PROMPT, prompt).upper()

    if "EXPLAIN" in style:
        return (
            "The user wants an explanation. Give a detailed but readable answer with clear "
            "paragraphs, explain the reasoning/mechanism when supported, and include examples."
        )
    if "PRACTICAL" in style:
        return (
            "The user wants practical guidance. Focus on safe next steps, what to monitor, "
            "and when to contact a healthcare professional."
        )
    return "The user wants a concise answer. Be clear, natural, and avoid unnecessary detail."
