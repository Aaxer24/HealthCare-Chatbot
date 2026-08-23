from pathlib import Path

from src.config import DB_FAISS_PATH, VECTORSTORE_METADATA_PATH, get_cli_config


def validate_runtime() -> None:
    """Fail fast on missing production prerequisites."""
    config = get_cli_config()
    if not config.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required.")

    if not DB_FAISS_PATH.exists():
        raise RuntimeError(
            f"Vectorstore directory not found at {DB_FAISS_PATH}. "
            "Build the FAISS index before starting the container."
        )

    expected_files = [
        DB_FAISS_PATH / "index.faiss",
        DB_FAISS_PATH / "index.pkl",
        VECTORSTORE_METADATA_PATH,
    ]
    missing = [str(path) for path in expected_files if not path.exists()]
    if missing:
        raise RuntimeError(
            "Vectorstore is incomplete. Missing required files: " + ", ".join(missing)
        )


if __name__ == "__main__":
    validate_runtime()
