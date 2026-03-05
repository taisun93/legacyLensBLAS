"""
Retrieval precision evaluation against 15 ground-truth (query, expected routine) pairs.
Run after ingestion. Prints per-query pass/fail and precision %.
"""

# 15 ground truth pairs from plan (Stack Overflow–style queries → expected BLAS routine)
GROUND_TRUTH = [
    ("double precision general matrix multiply", "DGEMM"),
    ("single precision matrix vector multiply", "SGEMV"),
    ("double precision dot product", "DDOT"),
    ("euclidean norm of a vector", "DNRM2"),
    ("triangular matrix vector multiply", "DTRMV"),
    ("triangular solve", "DTRSV"),
    ("rank one update outer product", "DGER"),
    ("symmetric matrix vector multiply", "DSYMV"),
    ("scale a vector by a scalar", "DSCAL"),
    ("copy one vector to another", "DCOPY"),
    ("add scaled vector axpy", "DAXPY"),
    ("swap two vectors", "DSWAP"),
    ("index of maximum absolute value", "IDAMAX"),
    ("sum of absolute values", "DASUM"),
    ("complex double precision matrix multiply", "ZGEMM"),
]


def run_evaluation(k: int = 5) -> float:
    """
    Run retrieval for each ground-truth query; check if expected routine is in top-k.
    Returns precision as a fraction in [0, 1].
    """
    from embedder import embed_query
    from retriever import get_collection, search_with_embedding

    collection = get_collection()
    hits = 0
    total = len(GROUND_TRUTH)

    for query, expected_routine in GROUND_TRUTH:
        emb = embed_query(query)
        results = search_with_embedding(emb, k=k, collection=collection)
        top_routines = [
            (r.get("metadata") or {}).get("routine_name") or ""
            for r in results
        ]
        found = expected_routine.upper() in [r.upper() for r in top_routines]
        if found:
            hits += 1
        yield query, expected_routine, found, top_routines

    yield None, None, hits, total  # sentinel: (None, None, hits, total)


def main() -> float:
    """Print per-query pass/fail and final precision %. Returns precision as fraction."""
    print("Retrieval Precision (ground truth, top-5)")
    print("=" * 50)
    gen = run_evaluation(k=5)
    hits, total = 0, len(GROUND_TRUTH)
    for item in gen:
        query, expected, found, rest = item
        if query is None:
            hits, total = found, rest  # sentinel: found=hits, rest=total
            break
        status = "PASS" if found else "FAIL"
        print(f"  [{status}] {query!r} -> {expected} (top-5: {rest})")
    precision_pct = (hits / total * 100) if total else 0.0
    print("=" * 50)
    print(f"Precision: {hits}/{total} = {precision_pct:.1f}%")
    target = 70.0
    print(f"Target >={target}%: {'PASS' if precision_pct >= target else 'FAIL'}")
    return hits / total if total else 0.0


if __name__ == "__main__":
    main()
