"""FastAPI entry point for DocuSearch."""
from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core import index_document, search_documents

app = FastAPI(
    title="DocuSearch API",
    description="Document ingestion and semantic search with optional local LLM answers.",
    version="1.0.0",
)
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(10 * 1024 * 1024)))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "docus earch".replace(" ", "")}


@app.post("/documents", tags=["documents"])
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or "document.txt"
    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=415, detail="Only PDF and TXT files are supported.")
    content = await file.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_BYTES} bytes.")
    try:
        return index_document(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document indexing failed.") from exc


@app.post("/search", tags=["search"])
def search(request: SearchRequest) -> dict[str, Any]:
    try:
        matches = search_documents(request.query, request.top_k)
        return {"query": request.query, "results": matches, "count": len(matches)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Search failed. Check the embedding model and index.") from exc


@app.post("/ask", tags=["search"])
async def ask(request: AskRequest) -> dict[str, Any]:
    try:
        sources = search_documents(request.question, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not sources:
        return {"question": request.question, "answer": "No documents indexed yet. Upload a PDF or TXT first.", "sources": []}

    context = "\n\n".join(f"[Source {i + 1}: {s['filename']} — chunk {s['chunk_index']}]\n{s['text']}"
                            for i, s in enumerate(sources))
    prompt = ("Answer the question using only the context below. If the context does not contain the answer, "
              "say you cannot determine it from the documents. Cite sources as [Source 1], [Source 2], etc.\n\n"
              f"CONTEXT:\n{context}\n\nQUESTION: {request.question}")
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(f"{OLLAMA_URL.rstrip('/')}/api/generate",
                                         json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
            response.raise_for_status()
            answer = response.json().get("response", "").strip()
            mode = "ollama"
    except (httpx.HTTPError, ValueError, KeyError):
        # A transparent extractive fallback keeps the API usable without Ollama.
        answer = "LLM unavailable; here are the most relevant excerpts:\n\n" + "\n\n".join(
            f"[Source {i + 1}] {s['text']}" for i, s in enumerate(sources))
        mode = "extractive_fallback"
    return {"question": request.question, "answer": answer, "answer_mode": mode,
            "sources": [{"filename": s["filename"], "chunk_index": s["chunk_index"], "distance": s["distance"]}
                        for s in sources]}
