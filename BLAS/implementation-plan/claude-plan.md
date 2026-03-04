# LegacyLens — Implementation Spec
## For Claude Code / Cursor

---

## Project Summary

Build a CLI tool that lets developers query the BLAS Fortran codebase in natural language.
User types a question, gets back relevant code snippets with file/line references and a plain English answer.

**Stack:** Python, LangChain, ChromaDB (embedded), Voyage Code 2 embeddings, Claude (claude-sonnet-4-20250514), Rich CLI

---

## Project Structure

```
legacylens/
├── ingest.py         # One-time pipeline: read files → chunk → embed → store
├── query.py          # CLI entry point
├── chunker.py        # Fortran-aware splitting logic
├── embedder.py       # Voyage Code 2 wrapper
├── retriever.py      # ChromaDB read/write
├── generator.py      # Claude answer generation
├── features.py       # 4 code understanding features
├── evaluate.py       # Precision measurement against ground truth
├── blas/             # Downloaded BLAS .f source files
├── chroma_db/        # Persisted vector store (gitignore)
├── .env              # VOYAGE_API_KEY, ANTHROPIC_API_KEY
├── requirements.txt
└── README.md
```

---

## Step 1 — Acquire BLAS Source

Clone the Reference LAPACK repo and copy BLAS/SRC into `./blas/src/`.
Target: ~50-70 `.f` files.

```
https://github.com/Reference-LAPACK/lapack
Copy: lapack/BLAS/SRC → ./blas/src/
```

---

## Step 2 — Ingestion Pipeline (`ingest.py`)

One-time script. Runs once, persists ChromaDB to disk. Does not re-run on every query.

**Steps in order:**
1. Recursively find all `.f`, `.f90`, `.for` files in `./blas`
2. Read each file (handle encoding with latin-1 fallback)
3. Chunk each file using `chunker.py`
4. Print first chunk to terminal as sanity check before embedding
5. Batch embed all chunks using `embedder.py` (batch size 128)
6. Store in ChromaDB collection via `retriever.py`
7. Print total chunk count on completion

**Performance target:** Full BLAS corpus in under 5 minutes.

---

## Step 3 — Chunking (`chunker.py`)

**Primary strategy — function-level splitting:**
- Detect subroutine/function boundaries by regex matching `SUBROUTINE`, `FUNCTION`, and typed variants (`DOUBLE PRECISION FUNCTION`, etc.) at column 0-6
- Each routine from its declaration to its `END` statement = one chunk
- Cap chunks at 4000 characters, truncate with `[truncated]` note if exceeded

**Fallback — fixed-size:**
- If no routines found in a file (headers, utilities), split into 50-line chunks with 5-line overlap

**Metadata to extract per chunk:**
- `file_path` — full path
- `file_name` — basename
- `routine_name` — e.g. `DGEMM`
- `start_line`, `end_line` — integers
- `precision` — inferred from first letter of routine name (S=single, D=double, C=complex, Z=double complex)
- `operation_type` — inferred from routine name suffix (GEMM=matrix_matrix_multiply, DOT=dot_product, NRM2=norm, etc.)
- `description` — first comment block above/within the routine, max 500 chars
- `chunk_type` — `"function"` or `"fixed_size"`

---

## Step 4 — Embeddings (`embedder.py`)

**Model:** `voyage-code-2`

Two distinct functions:
- `embed_chunks(texts)` — batch embed documents, use `input_type="document"`
- `embed_query(query)` — single query embed, use `input_type="query"`

This asymmetric distinction is important for retrieval quality — do not use the same input_type for both.

---

## Step 5 — Vector Storage (`retriever.py`)

**ChromaDB in persistent mode** — store to `./chroma_db/`, survive restarts.

Collection config: cosine similarity (`hnsw:space: cosine`).

Key functions:
- `get_collection(reset=False)` — create or load collection, optional wipe for re-ingestion
- `search(query, k=5, filters=None)` — embed query, search, return formatted results with relevance score (convert cosine distance to 0-1 similarity)
- `get_full_file(file_path)` — retrieve all chunks for a file sorted by line number, for drill-down

---

## Step 6 — Answer Generation (`generator.py`)

**Model:** `claude-sonnet-4-20250514`, max_tokens=1000, batch (not streaming).

System prompt instructs Claude to:
- Act as a Fortran/BLAS expert
- Always cite file path and line numbers
- Explain BLAS naming conventions (precision prefix, operation type) when relevant
- Explicitly say "not found" rather than hallucinate when context is irrelevant

Add a low-confidence warning when average retrieval score < 0.5 — prepend a visible warning to the answer rather than hiding it.

---

## Step 7 — CLI (`query.py`)

Use `rich` library for all output formatting.

**Two modes:**

Interactive mode (no args): REPL loop, accepts queries until `quit`

Single query mode: `python query.py "your question here"`

**Output per query:**
1. For each of top-k results show:
   - Result number, relevance score (color coded: green ≥0.8, yellow ≥0.6, red <0.6)
   - File name, line range, routine name
   - Precision and operation type metadata
   - Syntax-highlighted Fortran code snippet (truncated to 800 chars for display)
2. Generated answer in a panel below results
3. Query latency in seconds
4. Prompt for drill-down: user enters result number to see full file content

**CLI flags:**
- `-k` / `--top-k` — number of results, default 5
- `--feature` — one of `explain`, `docs`, `translate`, `patterns`

---

## Step 8 — Code Understanding Features (`features.py`)

All four features follow the same pattern: retrieve top-5 chunks for the query, pass to Claude with a feature-specific system prompt.

**Feature 1 — Code Explanation (`--feature explain`)**
Prompt: explain what the routine does in plain English, what inputs it takes, what it returns, edge cases.

**Feature 2 — Documentation Generation (`--feature docs`)**
Prompt: generate structured docs — Purpose, Arguments table, Returns, Example usage, Notes on precision.

**Feature 3 — Translation Hints (`--feature translate`)**
Prompt: suggest NumPy/SciPy equivalents with example code, note when Fortran is still preferable.

**Feature 4 — Pattern Detection (`--feature patterns`)**
Prompt: identify shared patterns across the retrieved routines — argument conventions, computation structure, design similarities.

---

## Step 9 — Evaluation (`evaluate.py`)

15 ground truth pairs: (natural language query, expected BLAS routine name).

Source queries from Stack Overflow [blas] tag — real questions with known answers.

Suggested ground truth set:
```
"double precision general matrix multiply" → DGEMM
"single precision matrix vector multiply" → SGEMV
"double precision dot product" → DDOT
"euclidean norm of a vector" → DNRM2
"triangular matrix vector multiply" → DTRMV
"triangular solve" → DTRSV
"rank one update outer product" → DGER
"symmetric matrix vector multiply" → DSYMV
"scale a vector by a scalar" → DSCAL
"copy one vector to another" → DCOPY
"add scaled vector axpy" → DAXPY
"swap two vectors" → DSWAP
"index of maximum absolute value" → IDAMAX
"sum of absolute values" → DASUM
"complex double precision matrix multiply" → ZGEMM
```

For each pair: run search, check if expected routine appears in top-5, count hits.
Print per-query pass/fail, final precision %, pass/fail against 70% target.

---

## Step 10 — README

Minimum viable sections:
- One-line description
- Setup (clone, venv, pip install, .env)
- Ingest command
- Query commands (interactive, single, with feature flag)
- Architecture table: component → choice → rationale (5 rows)

---

## Key Constraints

- ChromaDB persists to disk — ingestion runs once, queries read from disk
- Voyage Code 2 for BOTH ingestion and queries — never mix models
- Batch response only — no streaming
- Top-k default is 5 — make it a tunable parameter
- `.env` for all API keys — never hardcode
- `chroma_db/` and `.env` in `.gitignore`

---

## Definition of Done

- [ ] `python ingest.py` completes without errors, prints chunk count
- [ ] `python query.py "find double precision matrix multiply"` returns DGEMM in results
- [ ] All 4 `--feature` flags return non-empty output
- [ ] `python evaluate.py` reports ≥70% precision
- [ ] End-to-end query latency under 3 seconds
- [ ] Public GitHub repo with README

---

*Ship MVP first. Optimize chunking only after verifying retrieval works end-to-end.*