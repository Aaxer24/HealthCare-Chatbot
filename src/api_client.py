import json
from urllib import error, request

from src.config import AppConfig


def call_chat_api(
    prompt: str,
    chat_history: list[tuple[str, str]],
    config: AppConfig,
) -> dict:
    payload = {
        "prompt": prompt,
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
