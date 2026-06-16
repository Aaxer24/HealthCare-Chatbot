import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent
DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

PROMPT_TEMPLATE = """
You are a careful medical information assistant. Answer only from the context.
If the context does not contain enough information, say you do not know from the documents.
Do not diagnose or prescribe treatment. For emergency symptoms, advise urgent local care.

Context:
{context}

Question:
{question}

Answer:
"""


def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(DB_FAISS_PATH),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def build_qa_chain():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it to .env before running this script.")

    llm = ChatGroq(
        model_name=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0.1")),
        groq_api_key=groq_api_key,
        timeout=30,
        max_retries=2,
    )
    db = load_vectorstore()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TEMPLATE)},
    )


def format_source(doc):
    source = Path(str(doc.metadata.get("source", "Unknown source"))).name
    page = doc.metadata.get("page")
    return f"{source}, page {int(page) + 1}" if isinstance(page, int) else source


def main():
    if not DB_FAISS_PATH.exists():
        raise FileNotFoundError(f"Vector store not found at {DB_FAISS_PATH}. Run create_memory_for_llm.py first.")

    qa_chain = build_qa_chain()
    user_query = input("Write your query: ").strip()
    if not user_query:
        print("No query provided.")
        return

    response = qa_chain.invoke({"query": user_query})
    print("\nRESULT\n")
    print(response["result"].strip())
    print("\nSOURCE DOCUMENTS\n")
    for doc in response["source_documents"]:
        print(f"- {format_source(doc)}")


if __name__ == "__main__":
    main()
