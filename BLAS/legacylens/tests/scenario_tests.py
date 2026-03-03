"""
LegacyLens scenario tests — 6 queries from requirements.md, adapted for BLAS.
Run: py run_tests.py
Requires: ingest completed (chroma_db populated), .env with VOYAGE_API_KEY and ANTHROPIC_API_KEY.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retriever import search
from generator import generate_answer

# 6 scenario queries (BLAS-adapted from requirements.md)
SCENARIOS = [
    {
        "query": "Where is the main entry point of this program?",
        "expect_in_top5": None,  # BLAS is a library; no single main
    },
    {
        "query": "What routines modify the input vector or array?",
        "expect_in_top5": ["DAXPY", "DCOPY", "DSCAL", "DSWAP", "DGER"],
    },
    {
        "query": "Explain what DGEMM does",
        "expect_in_top5": ["DGEMM"],
    },
    {
        "query": "Find routines that print error messages or write output",
        "expect_in_top5": ["XERBLA"],
    },
    {
        "query": "What are the dependencies of DGEMM?",
        "expect_in_top5": ["XERBLA", "LSAME", "DGEMM"],
    },
    {
        "query": "Show me error handling patterns in this codebase",
        "expect_in_top5": ["XERBLA"],
    },
]


def run_scenario(i: int, scenario: dict) -> tuple[bool, str]:
    """Run one scenario; return (passed, message)."""
    query = scenario["query"]
    expect = scenario.get("expect_in_top5")

    try:
        results = search(query, k=5)
    except Exception as e:
        return False, f"Search failed: {e}"

    if not results:
        return False, "No search results"

    top5_routines = [r.get("metadata", {}).get("routine_name", "") for r in results]

    # Check expected routines in top-5
    if expect:
        found = any(exp in top5_routines for exp in expect)
        if not found:
            return False, f"Expected one of {expect} in top-5, got {top5_routines[:5]}"

    # Generate answer (skip if no Anthropic key)
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            answer = generate_answer(query, results)
            if not answer or "No relevant" in answer:
                return False, "Empty or fallback answer"
        except Exception as e:
            return False, f"Generation failed: {e}"

    return True, "OK"


def main():
    print("LegacyLens Scenario Tests")
    print("=" * 50)
    if not os.getenv("VOYAGE_API_KEY"):
        print("ERROR: VOYAGE_API_KEY not set. Run ingest first.")
        return 1
    passed = 0
    failed = 0
    for i, scenario in enumerate(SCENARIOS, 1):
        ok, msg = run_scenario(i, scenario)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {i}. [{status}] {scenario['query'][:50]}...")
        if not ok:
            print(f"      {msg}")
    print("=" * 50)
    print(f"Result: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
