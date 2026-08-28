from dotenv import find_dotenv, load_dotenv

from src.config import DB_FAISS_PATH, get_cli_config
from src.rag import answer_question
from src.router import answer_style_instruction


load_dotenv(find_dotenv())


def main():
    if not DB_FAISS_PATH.exists():
        raise FileNotFoundError(f"Vector store not found at {DB_FAISS_PATH}. Run create_memory_for_llm.py first.")

    config = get_cli_config()
    user_query = input("Write your query: ").strip()
    if not user_query:
        print("No query provided.")
        return

    style_instruction = answer_style_instruction(user_query, [], config)
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
