import json
from urllib import error, request

from src.config import LOGGER, AppConfig


def call_chat_api(
    prompt: str,
    chat_history: list[tuple[str, str]],
    config: AppConfig,
    document_text: str = "",
) -> dict:
    payload = {
        "prompt": prompt,
        "document_text": document_text,
        "chat_history": [
            {"role": "user", "content": user_message}
            for user_message, _ in chat_history
        ],
    }

    # Preserve alternating user/assistant history for the API contract.
    payload["chat_history"] = []
    for user_message, assistant_message in chat_history:
        payload["chat_history"].append({"role": "user", "content": user_message})
        payload["chat_history"].append({"role": "assistant", "content": assistant_message})

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=f"{config.api_base_url}/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed with status {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"Could not reach API at {config.api_base_url}. "
            "Make sure the FastAPI server is running."
        ) from exc


def submit_feedback(
    question: str,
    answer: str,
    rating: str,
    config: AppConfig,
    *,
    sources: list[dict] | None = None,
    message_type: str = "",
) -> bool:
    """Send a thumbs up/down. Never raises -- worst case we just lose one record."""
    payload = {
        "question": question,
        "answer": answer,
        "rating": rating,
        "sources": sources or [],
        "message_type": message_type,
    }
    req = request.Request(
        url=f"{config.api_base_url}/feedback",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception:
        LOGGER.warning("Could not submit feedback", exc_info=True)
        return False
