from src.config import AppConfig


def make_config(**overrides) -> AppConfig:
    defaults = dict(
        groq_api_key="test-key",
        model_name="openai/gpt-oss-120b",
        retrieval_k=8,
        rerank_k=5,
        temperature=0.1,
        enable_reranking=True,
        api_base_url="http://127.0.0.1:8000",
    )
    defaults.update(overrides)
    return AppConfig(**defaults)
