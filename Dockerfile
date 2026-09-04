# syntax=docker/dockerfile:1
#
# Single-container image: runs the FastAPI backend (internal-only, port 8000)
# and the Streamlit UI (published on $PORT) together in one container -- see
# docker/entrypoint.sh. Built for Google Cloud Run, which only runs one
# container on one port per service and injects which port via $PORT.

# ---- builder ------------------------------------------------------------
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- runtime --------------------------------------------------------------
FROM python:3.10-slim AS runtime

# tesseract-ocr powers the report/prescription upload feature (src/ocr.py).
# Only the English language data is installed; the full language set is ~500MB
# and this corpus is English-only.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    API_BASE_URL=http://127.0.0.1:8000 \
    PORT=8080

WORKDIR /app

# App code and the pre-built vector index (see README for how to rebuild
# vectorstore/ before building this image -- it is NOT regenerated at build
# time, since PDF parsing + embedding takes on the order of tens of minutes).
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/
COPY docker/entrypoint.sh ./docker/entrypoint.sh
COPY medibot.py ./
COPY vectorstore/ ./vectorstore/

# Pre-download the embedding + reranker weights at build time so the
# container never needs Hugging Face Hub access at runtime (only Groq's API)
# and doesn't re-download ~200MB of weights on every restart.
RUN python -c "\
from langchain_huggingface import HuggingFaceEmbeddings; \
from sentence_transformers import CrossEncoder; \
HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

RUN chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /home/appuser/.streamlit \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8080\")}/_stcore/health')" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker/entrypoint.sh"]
