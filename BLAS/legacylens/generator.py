"""
Claude-based answer generation from retrieved BLAS code chunks.
"""

import os
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5"  # Fastest model for low latency
MAX_TOKENS = 128

_client: Optional[Anthropic] = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        _client = Anthropic(api_key=api_key)
    return _client

# Base system prompt for BLAS expert behavior (kept short for latency)
BASE_SYSTEM = """Fortran/BLAS expert. Answer from the code context only. Be concise (2-4 sentences). Cite file and line numbers. Mention BLAS naming (S/D/C/Z, GEMM, DOT, etc.) when relevant. If context doesn't answer the question, say "not found"."""

# Feature-specific system prompt overrides
FEATURE_PROMPTS = {
    "explain": """You are a Fortran and BLAS expert. Explain the retrieved routine(s) in plain English.
- What does the routine do?
- What inputs does it take?
- What does it return or compute?
- Any edge cases or important notes?
Cite file path and line numbers.""",

    "docs": """You are a Fortran and BLAS expert. Generate structured documentation for the retrieved routine(s).
Format: Purpose, Arguments (table), Returns, Example usage, Notes on precision.
Cite file path and line numbers.""",

    "translate": """You are a Fortran and BLAS expert. Suggest NumPy/SciPy equivalents for the retrieved routine(s).
Provide example code where applicable.
Note when Fortran/BLAS is still preferable (e.g., performance).
Cite file path and line numbers.""",

    "patterns": """You are a Fortran and BLAS expert. Identify shared patterns across the retrieved routines.
- Argument conventions
- Computation structure
- Design similarities
Cite file path and line numbers.""",
}


def _build_context(chunks: list[dict]) -> str:
    """Build context string from search results."""
    parts = []
    for i, r in enumerate(chunks, 1):
        meta = r.get("metadata", r) if isinstance(r.get("metadata"), dict) else {}
        file_name = meta.get("file_name", "?")
        start = meta.get("start_line", "?")
        end = meta.get("end_line", "?")
        routine = meta.get("routine_name", "")
        precision = meta.get("precision", "")
        op_type = meta.get("operation_type", "")
        text = r.get("text", "")
        header = f"--- Result {i}: {file_name} (lines {start}-{end})"
        if routine:
            header += f" [{routine} | {precision} | {op_type}]"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def generate_answer(
    query: str,
    chunks: list[dict],
    feature: Optional[str] = None,
) -> str:
    """
    Generate an answer from Claude using retrieved chunks.
    chunks: list of dicts with 'text', 'metadata', 'similarity'
    feature: one of explain, docs, translate, patterns, or None for default
    """
    if not chunks:
        return "No relevant code found. Try a different query."

    # Low-confidence warning when avg similarity < 0.5
    avg_sim = sum(c.get("similarity", 0) for c in chunks) / len(chunks)
    warning = ""
    if avg_sim < 0.5:
        warning = "[Low confidence: retrieval scores are low. Answer may be less relevant.]\n\n"

    system = FEATURE_PROMPTS.get(feature, BASE_SYSTEM) if feature else BASE_SYSTEM
    context = _build_context(chunks)
    user_content = f"""Question: {query}

Context:

{context}

Answer (cite file:lines)."""

    client = _get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )

    text = response.content[0].text if response.content else ""
    return warning + text
