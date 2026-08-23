from dotenv import find_dotenv, load_dotenv

from src.ingest import build_vectorstore

load_dotenv(find_dotenv())


if __name__ == "__main__":
    build_vectorstore()
