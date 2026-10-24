# DocuSearch — Document RAG API

A small, portfolio-ready API for indexing PDF/TXT documents, retrieving semantically relevant passages, and answering questions with references to source chunks. Built with FastAPI, Sentence Transformers, ChromaDB, optional Ollama, pytest, Docker and GitHub Actions.

> This is a starter implementation. Run the tests and validate the full workflow locally before presenting it as a completed project on a CV.

## Features

- PDF/TXT ingestion with file-size and extension validation
- Text chunking with overlap
- Semantic vector search using `sentence-transformers/all-MiniLM-L6-v2`
- Persistent ChromaDB collection
- `/ask` endpoint that uses a local Ollama model when available and falls back to source excerpts when it is not
- OpenAPI documentation, error handling, automated unit tests and CI workflow
- Docker image and Compose configuration

## Run locally (Python 3.11 recommended)

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs>. The first document operation downloads the embedding model from Hugging Face, so the first run needs internet access and can take a few minutes.

## Optional local LLM

Install Ollama separately, then pull a small model:

```bash
ollama pull qwen2.5:3b
```

By default the API calls `http://localhost:11434`. Configure `OLLAMA_URL` and `OLLAMA_MODEL` as environment variables if needed. If Ollama is unreachable, `/ask` returns relevant source excerpts instead of pretending a generated answer was produced.

## Docker

```bash
docker compose up --build
```

Then open <http://localhost:8000/docs>. Compose starts Ollama as an optional generation service. Pull the model in the Ollama container before using generated answers:

```bash
docker compose exec ollama ollama pull qwen2.5:3b
```

The embedding model is downloaded on first use inside the API container. The vector database is persisted in a named volume.

## API examples

Upload a PDF or TXT:

```bash
curl -X POST http://localhost:8000/documents \
  -F 'file=@sample.txt'
```

Semantic search:

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"How does the system handle errors?","top_k":3}'
```

Ask a question grounded in retrieved documents:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does the system handle errors?","top_k":3}'
```

Health check: `GET /health`.

## Tests and lint

```bash
pytest -q
ruff check app tests
```

The included tests cover text extraction and chunking. Before calling this production-ready, add integration tests for upload/search/ask, authentication, rate limiting, robust PDF limits, structured logging, and monitoring.

## Evaluation idea

Create a small benchmark of 15–20 questions and manually label the expected source chunks. Compute Recall@3 (fraction of questions for which at least one expected chunk appears in the top three results). Report the actual measured result in the README; do not invent a score.

## Architecture

```text
Client -> FastAPI -> PDF/TXT extraction -> chunking -> Sentence Transformers
                                              |                  |
                                              +--------------> ChromaDB
Client -> /ask -> semantic retrieval -> context + question -> Ollama (optional)
                                               \\-> excerpt fallback when Ollama is unavailable
```

## Limitations

- Text-based PDFs only; scanned PDFs require OCR.
- No authentication or multi-user isolation.
- Local model downloads and inference can require significant disk space and memory.
- Source references point to filenames and chunk indices, not PDF page numbers.
- Retrieval relevance and answer quality must be evaluated on a real benchmark.
