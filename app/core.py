"""Document ingestion and semantic retrieval services."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(DATA_DIR / "chroma")))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

_client = None
_collection = None
_embedder = None


def get_collection():
    """Lazily initialize Chroma so importing the API does not download models."""
    global _client, _collection
    if _collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        text = content.decode("utf-8", errors="replace")
    elif suffix == ".pdf":
        import io

        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise ValueError("Unsupported file type. Upload a .pdf or .txt file.")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError("No extractable text found in this document.")
    return text


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap must be in [0, size).")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Prefer a natural boundary when it is not too far from the target.
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + size // 2:
                end = boundary + (1 if text[boundary] == "\n" else 2)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def index_document(filename: str, content: bytes) -> dict[str, Any]:
    text = extract_text(filename, content)
    chunks = chunk_text(text)
    document_id = hashlib.sha256(filename.encode() + b"\0" + content).hexdigest()[:24]
    collection = get_collection()
    # Re-uploading the same document replaces its previous chunks.
    existing = collection.get(where={"document_id": document_id})
    if existing.get("ids"):
        collection.delete(ids=existing["ids"])
    embeddings = get_embedder().encode(chunks, normalize_embeddings=True).tolist()
    ids = [f"{document_id}-{i}" for i in range(len(chunks))]
    metadatas = [{"document_id": document_id, "filename": Path(filename).name, "chunk_index": i}
                 for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return {"document_id": document_id, "filename": Path(filename).name, "chunks_indexed": len(chunks)}


def search_documents(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("Query must not be empty.")
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")
    collection = get_collection()
    if collection.count() == 0:
        return []
    vector = get_embedder().encode([query], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=vector, n_results=min(top_k, collection.count()),
                              include=["documents", "metadatas", "distances"])
    matches = []
    for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        matches.append({"text": doc, "filename": meta["filename"], "document_id": meta["document_id"],
                        "chunk_index": meta["chunk_index"], "distance": float(distance)})
    return matches
