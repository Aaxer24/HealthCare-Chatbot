import os

from dotenv import find_dotenv, load_dotenv

from src.config import DB_FAISS_PATH, DEFAULT_MODEL, AppConfig
from src.rag import answer_question
from src.router import answer_style_instruction


load_dotenv(find_dotenv())


def get_cli_config() -> AppConfig:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it to .env before running this script.")

    retrieval_k = int(os.getenv("RETRIEVAL_K", "8"))
    rerank_k = int(os.getenv("RERANK_K", "5"))
    return AppConfig(
        groq_api_key=groq_api_key,
        model_name=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        retrieval_k=retrieval_k,
        rerank_k=min(rerank_k, retrieval_k),
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0.1")),
        enable_reranking=os.getenv("ENABLE_RERANKING", "true").lower() in {"1", "true", "yes", "on"},
    )


def main():
    if not DB_FAISS_PATH.exists():
        raise FileNotFoundError(f"Vector store not found at {DB_FAISS_PATH}. Run create_memory_for_llm.py first.")

    config = get_cli_config()
    user_query = input("Write your query: ").strip()
    if not user_query:
        print("No query provided.")
        return

    style_instruction = answer_style_instruction(user_query, config)
    question = f"{user_query}\n\nResponse style instruction: {style_instruction}"
    answer, sources = answer_question(question, [], config)

    print("\nRESULT\n")
    print(answer.strip())
    print("\nSOURCE DOCUMENTS\n")
    for source in sources:
        print(f"- {source['source']}")
        print(f"  {source['snippet']}")


if __name__ == "__main__":
    main()
