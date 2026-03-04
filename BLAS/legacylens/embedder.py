"""
Voyage Code 2 embedding wrapper.
Use input_type='document' for chunks, input_type='query' for search queries.
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
import voyageai

load_dotenv()

MODEL = "voyage-code-2"
BATCH_SIZE = 128


_client: "voyageai.Client | None" = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY not set in .env")
        _client = voyageai.Client(api_key=api_key)
    return _client


def embed_chunks(texts: list[str], batch_size: int = BATCH_SIZE) -> list[list[float]]:
    """
    Batch embed documents for indexing. Uses input_type='document'.
    """
    if not texts:
        return []
    client = _get_client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        if i > 0:
            time.sleep(25)  # Throttle for rate limits (e.g. 3 RPM without payment)
        batch = texts[i : i + batch_size]
        result = client.embed(
            texts=batch,
            model=MODEL,
            input_type="document",
        )
        all_embeddings.extend(result.embeddings)
    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single search query. Uses input_type='query'.
    """
    client = _get_client()
    result = client.embed(
        texts=[query],
        model=MODEL,
        input_type="query",
    )
    return result.embeddings[0]
