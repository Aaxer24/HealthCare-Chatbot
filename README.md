# Medical Knowledge Chatbot

An interactive healthcare chatbot that uses retrieval-augmented generation over curated medical PDFs. The app retrieves relevant passages from a FAISS vector store, sends grounded context to LLaMA-4 through Groq, and returns conversational answers with source citations.

## Features

- RAG over trusted medical PDFs stored in `data/`
- Streamlit chat interface with persistent session history
- LLaMA-4 via Groq for low-latency responses
- Medical safety prompt with emergency-care guidance and uncertainty handling
- Source citations with document names and page numbers
- MMR retrieval to improve context diversity
- Cached embeddings, vector store, and model clients for faster responses
- Environment-based configuration for deployment

## Project Structure

```text
.
├── data/                       # Source medical PDFs
├── vectorstore/db_faiss/        # Generated FAISS index
├── create_memory_for_llm.py     # Builds the vector store from PDFs
├── create_memory_with_llm.py    # CLI smoke test for the RAG pipeline
├── medibot.py                  # Streamlit application
├── requirements.txt
└── .env.example
```

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Configure secrets.

```powershell
Copy-Item .env.example .env
```

Add your Groq API key to `.env`.

4. Build or rebuild the vector store.

```powershell
python create_memory_for_llm.py
```

5. Run the app.

```powershell
streamlit run medibot.py
```

## Configuration

These values can be set in `.env` or Streamlit secrets.

| Variable | Purpose | Default |
| --- | --- | --- |
| `GROQ_API_KEY` | Groq API key | Required |
| `GROQ_MODEL` | Chat model | `llama-3.3-70b-versatile` |
| `MODEL_TEMPERATURE` | Response randomness | `0.1` |
| `RETRIEVAL_K` | Number of retrieved chunks | `5` |
| `LOG_LEVEL` | Python logging level | `INFO` |

## Medical Safety

This chatbot is for educational information only. It does not diagnose conditions, prescribe treatment, or replace a licensed clinician. The prompt instructs the model to recommend urgent local care for emergency warning signs and to say when the documents do not support an answer.

## Resume Highlights

- Developed an interactive medical chatbot using RAG over trusted medical PDFs, providing symptom-based guidance with source citations.
- Built a real-time Streamlit conversational interface integrated with LLaMA-4 through Groq for fast responses.
- Reduced hallucination risk with strict grounding prompts, follow-up question rewriting, MMR retrieval, bounded chat history, and source citation.
