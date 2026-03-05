# LegacyLens: AI Cost Analysis & RAG Architecture

This document covers **AI cost analysis** (development and production) and **RAG architecture** for the LegacyLens BLAS code Q&A application.

---

# Part 1 — AI Cost Analysis

## 1.1 Development & Testing Costs

### Embedding API costs (Voyage Code 2) — non-issue

Ingestion and query embedding stayed within Voyage’s free tier (first 50M tokens). The BLAS corpus is small (169 Fortran files, 48,480 LOC, 173 chunks; one-time ingest) and effectively static, so **embedding costs are a non-issue** for both development and production. If the corpus ever grew, free tier would still cover a large multiple of the current size.

**Ingestion performance (measured via `run_performance_targets.py`):**

| Metric | Value |
|--------|--------|
| Source files (.f/.f90/.for) | 169 |
| Total LOC | 48,480 |
| Total chunks stored | 173 |
| Ingestion time | 64.85 s |
| Throughput | 747.6 LOC/s |
| Codebase coverage | 169/169 files with ≥1 chunk (100%) |

### LLM API costs for answer generation (Anthropic Claude Haiku 4.5)

| Item | Value | Notes |
|------|--------|------|
| **Model** | `claude-haiku-4-5` | Input: $1.00 / 1M tokens; Output: $5.00 / 1M tokens |
| **Total tokens in (dev)** | 332,459 | System prompt + user (query + k=5 chunk context) across all dev/test requests |
| **Total tokens out (dev)** | 10,179 | 128 max tokens per call × number of requests |
| **Total token cost (dev)** | **$0.63** | 332,459 × $1/1M + 10,179 × $5/1M |

### Vector database / hosting (development)

ChromaDB is embedded and persists to disk (`chroma_db/`); no separate vector DB service or cost. Development was done locally (hosting $0).

### Total development spend breakdown

| Category | Amount |
|----------|--------|
| Embedding API (Voyage) | $0 (free tier) |
| LLM API (Anthropic) | $0.63 |
| Vector DB / hosting | $0 |
| **Total** | **$0.63** |

---

## 1.2 Production Cost Projections

**Pricing (as of 2025):**

- **Voyage Code 2:** $0.12 per 1M tokens (first 50M free). Embedding cost is treated as **negligible** for this app (tiny, static corpus).
- **Anthropic Claude Haiku 4.5:** $1.00 per 1M input tokens, $5.00 per 1M output tokens.
- **Hosting:** $7/month Render (Starter-type tier). ChromaDB is embedded; no separate vector DB fee.

**Realistic demand assumptions (BLAS):**

BLAS is a niche audience: HPC and numerical developers, LAPACK/BLAS maintainers, and students working with legacy Fortran. Usage is not “mass consumer” but occasional lookups when someone needs to find or understand a routine.

| Parameter | Value | Rationale |
|-----------|--------|-----------|
| **Active users** (query at least 1×/week) | 10–30 for typical deployment | Small community; many visitors may try once or a few times |
| **Queries per active user per day** | 2–4 | Lookup a routine, then maybe a follow-up; not all-day use |
| **Occasional users** (e.g. 1–5 queries/month) | Remainder of “users” | Drive-by or rare use |
| **Avg context tokens to LLM per query** | ~4,000 | Query + 5 chunk texts + formatting |
| **Avg output tokens per answer** | ~100 | `max_tokens=128`; answers typically shorter |
| **New code embeddings per month** | 0 | BLAS reference corpus is static |

**What the $7 Render tier can support:** A single small instance (e.g. 512MB RAM) can handle on the order of **1–2 requests per second** sustained (embed + Chroma + Claude per request). So roughly **~50–150 active users** at 2–4 queries/day each (~3,000–18,000 queries/month) stays within capacity and keeps response times acceptable. Above that, you’d scale to a larger Render plan or multiple instances.

**Production cost projections (monthly):**

| Scale | Users (active / occasional) | Queries/month | Embedding | LLM cost | Hosting | **Total $/month** |
|-------|-----------------------------|----------------|-----------|----------|---------|--------------------|
| **Typical** | ~20 active, ~30 occasional | ~2,500 | $0 | ~$11 | $7 | **~$18** |
| 100 | 30 active, 70 occasional | ~6,500 | $0 | ~$28 | $7 | **~$35** |
| 1,000 | 100 active, 900 occasional | ~25,000 | $0 | ~$108 | $25–50 (larger instance) | **~$135–160** |
| 10,000 | 500 active, 9,500 occasional | ~125,000 | $0 | ~$540 | Scaled (e.g. $100+) | **~$640+** |
| 100,000 | 2,000 active, 98,000 occasional | ~500,000 | $0 | ~$2,160 | Multi-instance / K8s | **~$2,500+** |

*LLM formula:* queries × (4,000 × $1/1M + 100 × $5/1M) = queries × $0.0045 per request.

**Summary table (for submission):**

| Scale | 100 Users | 1,000 Users | 10,000 Users | 100,000 Users |
|-------|-----------|-------------|--------------|----------------|
| **$/month** | ~$35 | ~$135–160 | ~$640+ | ~$2,500+ |

**Caching:** The app uses an in-memory query cache (`query_pipeline.py`). Repeated identical queries do not call the embedding or LLM APIs, which reduces cost when the same questions recur (e.g. “what is DGEMM?”).

---

# Part 2 — RAG Architecture Documentation

## 2.1 Vector DB Selection

**Choice: ChromaDB (embedded, persistent).**

- **Why:** Simple deployment and no separate service to operate. Persists to disk (`chroma_db/`), so ingestion runs once (locally or in CI) and the same directory can be committed or deployed with the app. No network dependency at query time for the vector store.
- **Tradeoffs considered:**
  - **Pinecone / Weaviate / Qdrant:** Better for very large scale and advanced features (e.g. filtering, multi-tenancy). Overkill for a single static BLAS corpus and would add operational and cost overhead.
  - **pgvector:** Good if the app already uses Postgres and wants a single store; adds DB dependency and schema/migration. Not required for current scale.
- **Configuration:** Single collection `blas_code`, cosine similarity (`hnsw:space: cosine`). No separate vector DB hosting cost; storage is local (or on the same host as the app on Render).

---

## 2.2 Embedding Strategy

- **Model:** Voyage **Code 2** (`voyage-code-2`).
- **Why it fits code understanding:** Code-optimized embedding model; supports `input_type="document"` for chunks and `input_type="query"` for search, which improves retrieval quality for code vs generic text embeddings.
- **Usage:**  
  - **Ingestion:** `embed_chunks(texts, batch_size=80)` with `input_type="document"`. Batch size 80 keeps under Voyage’s token-per-batch limit (~120K); ingest uses ~1.2K tokens/chunk.  
  - **Query:** `embed_query(query)` with `input_type="query"` for each user question.

---

## 2.3 Chunking Approach

- **Primary strategy — function-level (routine-level) splitting:**  
  - Fortran routines are detected by regex on `SUBROUTINE` / `FUNCTION` (and typed variants like `DOUBLE PRECISION FUNCTION`) in columns 0–6.  
  - Each chunk = one routine from its declaration to the corresponding `END` (or next routine).  
  - Preserves one routine per chunk so that retrieval returns whole routines, which matches “find the BLAS routine that does X.”
- **Boundary detection:**  
  - Start: `ROUTINE_RE` matches declaration line.  
  - End: `END_RE` or the line before the next routine’s declaration.  
  - Oversized routines: chunk text capped at **4,000 characters** with a `[truncated]` suffix.
- **Fallback:** If no routine is found in a file (e.g. headers), **fixed-size chunks** of 50 lines with 5-line overlap.
- **Metadata per chunk:** `file_path`, `file_name`, `routine_name`, `start_line`, `end_line`, `precision` (S/D/C/Z), `operation_type` (e.g. matrix_matrix_multiply for GEMM), `description` (leading comment block, max 500 chars), `chunk_type` (function vs fixed_size).  
See `legacylens/chunker.py` and `legacylens/docs/chunking.md` for full detail.

---

## 2.4 Retrieval Pipeline

- **Query flow:**  
  1. **Embed:** User query → Voyage `embed_query()` → single vector.  
  2. **Search:** ChromaDB `query()` with cosine similarity, `n_results=k` (default k=5), optional `where` filters. No re-ranking step.  
  3. **Context assembly:** Top-k chunks are formatted with metadata (file, lines, routine name, precision, operation type) and concatenated into a single context string.  
  4. **Generation:** Context + user question are sent to Claude Haiku 4.5 with a BLAS expert system prompt; response limited to 128 tokens for latency.
- **Re-ranking:** None. Top-k by cosine similarity only.
- **Caching:** In-memory cache in `query_pipeline.py` (exact query string → results + answer); cache hit skips embed and LLM.

---

## 2.5 Failure Modes & Edge Cases

- **Low retrieval confidence:** If average similarity across the top-k chunks is &lt; 0.5, the generator prepends a low-confidence warning. The model may still answer from weak context.
- **Routine not in top-k:** Queries that don’t match embedding space well (e.g. jargon or typo) can miss the correct routine; evaluation uses 15 ground-truth pairs to measure precision (target ≥70% in top-5).
- **Truncated chunks:** Routines &gt; 4,000 characters are truncated; very long routines may have important content cut off in context.
- **No reranker:** Ambiguous queries can return semantically similar but wrong routines (e.g. wrong precision) because order is by similarity only.
- **Cache:** Only exact string match is cached; rephrased queries are not deduplicated.
- **API/keys:** Missing or invalid `VOYAGE_API_KEY` or `ANTHROPIC_API_KEY` causes runtime errors; no graceful degradation.

---

## 2.6 Performance Results

**Ingestion (measured):**

| Metric | Value |
|--------|--------|
| Source files | 169 |
| Total LOC | 48,480 |
| Chunks | 173 |
| Ingestion time | 64.85 s |
| Throughput | 747.6 LOC/s |
| Codebase coverage | 169/169 files with ≥1 chunk (100%) |

**Retrieval precision (measured):** Evaluated with `evaluate.py` against 15 ground-truth (query, expected routine) pairs; **precision = (hits in top-5) / 15**.  
- **Result:** 14/15 = **93.3%** (target ≥70%: PASS).  
- Single failure: "rank one update outer product" → expected DGER; top-5 returned DSYR2, SSYR2, CGERU, DSPR2, ZGERU (rank-1/rank-2 update variants).

- **Query latency (target &lt;3 s end-to-end):**  
  - **Embed:** ~100–400 ms (Voyage API).  
  - **ChromaDB:** &lt;50 ms typical (local disk).  
  - **Claude:** ~200–800 ms (Haiku, 128 max tokens).  
  - **Total (uncached):** ~0.5–2 s typical; cache hit &lt;50 ms.
- **Precision:** Evaluated with `evaluate.py` against 15 ground-truth (query, expected routine) pairs; **precision = (hits in top-5) / 15**. (See measured result above.)
- **Example ground-truth pairs:**  
  - “double precision general matrix multiply” → DGEMM  
  - “single precision matrix vector multiply” → SGEMV  
  - “add scaled vector axpy” → DAXPY  
  - "rank one update outer product" → DGER (retrieval miss in top-5)

