"""
ChromaDB persistent vector store for BLAS code chunks.
"""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

from embedder import embed_query, embed_chunks

load_dotenv()

COLLECTION_NAME = "blas_code"
CHROMA_PATH = Path(__file__).resolve().parent / "chroma_db"

_client = None
_collection_cache = None


def _chroma_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection(reset: bool = False):
    """
    Create or load the ChromaDB collection. Cached after first load.
    Uses cosine similarity for retrieval.
    """
    global _collection_cache
    if reset:
        _collection_cache = None
        try:
            _chroma_client().delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    if _collection_cache is None:
        _collection_cache = _chroma_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection_cache


def add_chunks(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    collection=None,
):
    """Add chunk documents and embeddings to the collection."""
    if collection is None:
        collection = get_collection()
    ids = [f"{c['file_path']}:{c['start_line']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "file_path": c["file_path"],
            "file_name": c["file_name"],
            "routine_name": c["routine_name"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "precision": c["precision"],
            "operation_type": c["operation_type"],
            "description": c["description"][:500] if c["description"] else "",
            "chunk_type": c["chunk_type"],
        }
        for c in chunks
    ]
    # ChromaDB metadata values must be str, int, float, bool
    for m in metadatas:
        m["start_line"] = int(m["start_line"])
        m["end_line"] = int(m["end_line"])
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def search_with_embedding(
    query_embedding: list[float],
    k: int = 5,
    filters: Optional[dict] = None,
    collection=None,
) -> list[dict[str, Any]]:
    """
    Search collection with pre-computed embedding. No API calls.
    """
    if collection is None:
        collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=filters,
        include=["documents", "metadatas", "distances"],
    )
    formatted = []
    if results["ids"] and results["ids"][0]:
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = max(0.0, 1.0 - dist)
            formatted.append(
                {"id": doc_id, "text": doc, "metadata": meta, "similarity": similarity}
            )
    return formatted


def search(
    query: str,
    k: int = 5,
    filters: Optional[dict] = None,
    collection=None,
) -> list[dict[str, Any]]:
    """
    Embed query, search collection, return formatted results.
    Converts cosine distance to 0-1 similarity (higher = more similar).
    """
    query_embedding = embed_query(query)
    return search_with_embedding(query_embedding, k=k, filters=filters, collection=collection)


def get_full_file(file_path: str, collection=None) -> list[dict[str, Any]]:
    """
    Retrieve all chunks for a file, sorted by line number.
    """
    if collection is None:
        collection = get_collection()
    results = collection.get(
        where={"file_path": file_path},
        include=["documents", "metadatas"],
    )
    chunks = []
    if results["ids"]:
        for doc_id, doc, meta in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        ):
            chunks.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta,
                }
            )
        chunks.sort(key=lambda x: x["metadata"]["start_line"])
    return chunks
