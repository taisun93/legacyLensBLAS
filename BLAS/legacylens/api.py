"""
LegacyLens API — FastAPI wrapper for query and search.
Run: uvicorn api:app --reload
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from query_pipeline import run_query, format_timing
from retriever import get_collection

app = FastAPI(title="LegacyLens", description="Query BLAS codebase in natural language")


@app.on_event("startup")
def startup():
    """Pre-warm ChromaDB collection so first query doesn't pay load cost."""
    get_collection()


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    feature: str | None = None


class QueryResponse(BaseModel):
    results: list[dict]
    answer: str
    latency: dict[str, float]
    timing_breakdown: str


@app.post("/query", response_model=QueryResponse)
def api_query(req: QueryRequest):
    """Run a natural language query against the BLAS codebase."""
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    results, answer, timing = run_query(req.query, k=req.k, feature=req.feature)
    latency = {
        "total_ms": timing["total_ms"],
        "embed_ms": timing["embed_ms"],
        "chroma_ms": timing["chroma_ms"],
        "claude_ms": timing["claude_ms"],
        "cache_hit": timing["cache_hit"],
    }
    return QueryResponse(
        results=results,
        answer=answer,
        latency=latency,
        timing_breakdown=format_timing(timing),
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


@app.get("/warmup")
def warmup():
    """Pre-load ChromaDB collection so first query doesn't pay cold-start cost."""
    get_collection()
    return {"status": "ready"}
