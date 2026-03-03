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


def _chroma_client():
    return chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(reset: bool = False):
    """
    Create or load the ChromaDB collection.
    Uses cosine similarity for retrieval.
    """
    client = _chroma_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        # Alternative for ChromaDB 1.x: configuration={"hnsw": {"space": "cosine"}}
    )


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


def search(
    query: str,
    k: int = 5,
    filters: Optional[dict] = None,
    collection=None,
) -> list[dict[str, Any]]:
    """
    Embed query, search collection, return formatted results.
    Converts cosine distance to 0-1 similarity (higher = more similar).
    ChromaDB returns distance; for cosine, similarity = 1 - distance.
    """
    if collection is None:
        collection = get_collection()
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=filters,
        include=["documents", "metadatas", "distances"],
    )
    formatted = []
    if results["ids"] and results["ids"][0]:
        for i, (doc_id, doc, meta, dist) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            # Cosine distance: 0 = identical, 2 = opposite. similarity = 1 - distance
            similarity = max(0.0, 1.0 - dist)
            formatted.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta,
                    "similarity": similarity,
                }
            )
    return formatted


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
