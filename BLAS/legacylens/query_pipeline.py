"""
Query pipeline with per-step timing, cache, and latency optimization.
Target: <3s end-to-end.
"""

import time
from typing import Any, Optional

from embedder import embed_query
from retriever import get_collection, search_with_embedding
from generator import generate_answer


# Simple in-memory cache: query string -> (results, answer)
# Key: normalized query (strip + lower for case-insensitive exact match)
# For exact match we use the raw stripped query to avoid false hits
_cache: dict[str, tuple[list[dict], str]] = {}
_CACHE_MAX_SIZE = 200


def _cache_key(query: str) -> str:
    return query.strip()


def _cache_get(query: str) -> Optional[tuple[list[dict], str]]:
    key = _cache_key(query)
    return _cache.get(key)


def _cache_set(query: str, results: list[dict], answer: str) -> None:
    key = _cache_key(query)
    if len(_cache) >= _CACHE_MAX_SIZE:
        # Evict oldest (first inserted) - dict preserves order in 3.7+
        first = next(iter(_cache))
        del _cache[first]
    _cache[key] = (results, answer)


def run_query(
    query: str,
    k: int = 5,
    feature: Optional[str] = None,
    filters: Optional[dict] = None,
) -> tuple[list[dict], str, dict[str, float]]:
    """
    Run full query pipeline: embed -> ChromaDB search -> Claude generation.
    Returns (results, answer, timing_ms) with per-step breakdown.
    """
    timing: dict[str, float] = {
        "embed_ms": 0.0,
        "chroma_ms": 0.0,
        "claude_ms": 0.0,
        "cache_hit": False,
        "total_ms": 0.0,
    }
    t0 = time.perf_counter()

    # Cache check
    cached = _cache_get(query)
    if cached is not None:
        results, answer = cached
        timing["cache_hit"] = True
        timing["total_ms"] = (time.perf_counter() - t0) * 1000
        return results, answer, timing

    # 1. Embed
    t1 = time.perf_counter()
    query_embedding = embed_query(query)
    timing["embed_ms"] = (time.perf_counter() - t1) * 1000

    # 2. ChromaDB search (no API, uses persisted collection)
    t2 = time.perf_counter()
    collection = get_collection()
    results = search_with_embedding(query_embedding, k=k, filters=filters, collection=collection)
    timing["chroma_ms"] = (time.perf_counter() - t2) * 1000

    # 3. Claude generation
    t3 = time.perf_counter()
    answer = generate_answer(query, results, feature)
    timing["claude_ms"] = (time.perf_counter() - t3) * 1000

    timing["total_ms"] = (time.perf_counter() - t0) * 1000

    # Cache result
    _cache_set(query, results, answer)

    return results, answer, timing


def format_timing(timing: dict[str, float]) -> str:
    """Format timing breakdown for display."""
    parts = []
    if timing.get("cache_hit"):
        parts.append("cache hit")
    else:
        parts.append(f"embed={timing['embed_ms']:.0f}ms")
        parts.append(f"chroma={timing['chroma_ms']:.0f}ms")
        parts.append(f"claude={timing['claude_ms']:.0f}ms")
    parts.append(f"total={timing['total_ms']:.0f}ms")
    return " | ".join(parts)
