"""
LegacyLens API — FastAPI wrapper for query and search.
Run: uvicorn api:app --reload
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from retriever import search
from generator import generate_answer

app = FastAPI(title="LegacyLens", description="Query BLAS codebase in natural language")


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    feature: str | None = None


class QueryResponse(BaseModel):
    results: list[dict]
    answer: str
    latency: dict[str, float]


@app.post("/query", response_model=QueryResponse)
def api_query(req: QueryRequest):
    """Run a natural language query against the BLAS codebase."""
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    start = time.perf_counter()
    results = search(req.query, k=req.k)
    search_elapsed = time.perf_counter() - start
    gen_start = time.perf_counter()
    answer = generate_answer(req.query, results, req.feature)
    gen_elapsed = time.perf_counter() - gen_start
    return QueryResponse(
        results=results,
        answer=answer,
        latency={"total": search_elapsed + gen_elapsed, "search": search_elapsed, "generate": gen_elapsed},
    )


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the barebones terminal UI."""
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(404, "index.html not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}
