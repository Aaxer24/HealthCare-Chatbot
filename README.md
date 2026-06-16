# Medical Knowledge Chatbot

An interactive healthcare chatbot that uses retrieval-augmented generation over curated medical PDFs. The app retrieves relevant passages from a FAISS vector store, reranks the results, sends grounded context to Groq, and returns conversational answers with source citations and snippet previews.

## Features

- RAG over trusted medical PDFs stored in `data/`
- Streamlit chat interface with session history
- Groq chat model for low-latency responses
- BGE embeddings for stronger semantic retrieval
- MMR retrieval plus cross-encoder reranking for better passage quality
- Medical safety prompt with emergency-care guidance and uncertainty handling
- Source previews showing document names, pages, relevance scores, and snippets
- General-chat router so greetings and capability questions do not search PDFs
- Feedback buttons for answer quality collection
- Modular code structure for easier maintenance and deployment

## Project Structure

```text
.
|-- data/                       # Source medical PDFs
|-- vectorstore/db_faiss/        # Generated FAISS index
|-- src/
|   |-- config.py                # Settings, paths, environment variables
|   |-- prompts.py               # RAG, routing, and style prompts
|   |-- rag.py                   # Embeddings, FAISS, reranking, citations
|   |-- router.py                # General-chat and answer-depth routing
|   `-- ui.py                    # Streamlit styling and UI helpers
|-- create_memory_for_llm.py     # Builds the vector store from PDFs
|-- create_memory_with_llm.py    # CLI smoke test for the RAG pipeline
|-- medibot.py                   # Streamlit entrypoint
|-- requirements.txt
`-- .env.example
```

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

In this local project, your current environment is Conda-style, so these commands also work:

```powershell
.\venv\python.exe -m pip install -r requirements.txt
```

2. Configure secrets.

```powershell
Copy-Item .env.example .env
```

Add your Groq API key to `.env`.

3. Build or rebuild the vector store.

```powershell
.\venv\python.exe create_memory_for_llm.py
```

Rebuild is required after changing PDFs or changing the embedding model.

4. Run the app.

```powershell
.\venv\python.exe -m streamlit run medibot.py
```

## Configuration

These values can be set in `.env` or Streamlit secrets.

| Variable | Purpose | Default |
| --- | --- | --- |
| `GROQ_API_KEY` | Groq API key | Required |
| `GROQ_MODEL` | Chat model | `llama-3.3-70b-versatile` |
| `MODEL_TEMPERATURE` | Response randomness | `0.1` |
| `RETRIEVAL_K` | Initial retrieved chunks before reranking | `8` |
| `RERANK_K` | Final chunks after reranking | `5` |
| `ENABLE_RERANKING` | Enable cross-encoder reranking | `true` |
| `LOG_LEVEL` | Python logging level | `INFO` |

## Medical Safety

This chatbot is for educational information only. It does not diagnose conditions, prescribe treatment, or replace a licensed clinician. The prompt instructs the model to recommend urgent local care for emergency warning signs and to say when the documents do not support an answer.

## Resume Highlights

- Developed an interactive medical chatbot using RAG over trusted medical PDFs, providing symptom-based guidance with source citations and snippet previews.
- Improved answer grounding with BGE embeddings, MMR retrieval, cross-encoder reranking, follow-up question rewriting, and source attribution.
- Built a real-time Streamlit conversational interface integrated with Groq, general-chat routing, answer-depth routing, and feedback collection.
