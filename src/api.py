from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import LOGGER, get_cli_config
from src.rag import get_bm25_retriever, get_embedding_model, get_reranker, get_vectorstore
from src.service import generate_chat_response
from src.startup_checks import validate_runtime


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    document_text: str = ""  # OCR text from an uploaded report, if any


class SourcePreview(BaseModel):
    source: str
    snippet: str
    score: str = ""


class FeedbackRequest(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    rating: Literal["up", "down"]
    sources: list[SourcePreview] = Field(default_factory=list)
    comment: str = ""
    message_type: str = ""


class ChatResponse(BaseModel):
    message_type: Literal["GENERAL_CHAT", "OUT_OF_SCOPE", "MEDICAL_QUESTION"]
    answer: str
    sources: list[SourcePreview]
    cached: bool = False
    follow_ups: list[str] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_runtime()
    # Load the embedding model, reranker, and FAISS index now instead of
    # lazily on the first request. On a scale-to-zero platform (Cloud Run),
    # this pushes that cost into the container's startup/readiness window --
    # which docker/entrypoint.sh already blocks on before starting Streamlit
    # -- instead of a real user's first chat message eating a 15-30s delay.
    get_embedding_model()
    get_reranker()
    get_vectorstore()
    try:  # BM25 builds in ~1s, doing it here keeps it off the first request too
        get_bm25_retriever()
    except Exception:
        LOGGER.warning("BM25 pre-warm failed; hybrid search will fall back to semantic only", exc_info=True)
    yield


app = FastAPI(
    title="Healthcare Chatbot API",
    version="1.0.0",
    description="FastAPI endpoint for the medical RAG chatbot.",
    root_path="/api",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest) -> dict:
    """Record a thumbs up/down. Feeds scripts/review_feedback.py."""
    from src.feedback import record_feedback

    try:
        record_feedback(
            question=request.question,
            answer=request.answer,
            rating=request.rating,
            sources=[s.model_dump() for s in request.sources],
            comment=request.comment,
            message_type=request.message_type,
        )
        return {"status": "recorded"}
    except Exception as exc:
        LOGGER.exception("Failed to record feedback")
        raise HTTPException(status_code=500, detail="Unable to record feedback right now.") from exc


@app.get("/feedback/summary")
def feedback_summary_endpoint() -> dict:
    from src.feedback import feedback_summary

    return feedback_summary()


@app.get("/cache/stats")
def cache_stats() -> dict:
    """Cache hit rate, so you can tell if it's actually saving tokens."""
    from src.cache import get_answer_cache

    return get_answer_cache().stats()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        config = get_cli_config()
        pairs = [
            (request.chat_history[idx].content, request.chat_history[idx + 1].content)
            for idx in range(0, len(request.chat_history) - 1, 2)
            if request.chat_history[idx].role == "user"
            and request.chat_history[idx + 1].role == "assistant"
        ]
        result = generate_chat_response(request.prompt, pairs, config, request.document_text)
        return ChatResponse(**result)
    except Exception as exc:
        # Log the real error server-side, but never echo internal exception
        # details (stack traces, file paths, API error bodies) back to the
        # client -- that's an information-disclosure risk on a public endpoint.
        LOGGER.exception("API chat request failed")
        raise HTTPException(status_code=500, detail="Unable to generate an answer right now.") from exc
