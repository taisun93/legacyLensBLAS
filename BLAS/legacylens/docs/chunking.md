# Chunking Strategy

LegacyLens uses syntax-aware splitting for Fortran/BLAS source files. Implementation: `chunker.py`.

---

## Overview

Two strategies are applied in order:

1. **Function-level (primary)** — one chunk per subroutine/function
2. **Fixed-size + overlap (fallback)** — for files with no routine declarations

---

## Primary Strategy: Function-Level

Each subroutine or function is chunked as a unit.

### Detection

- **Boundary regex:** Matches `SUBROUTINE` or `FUNCTION` (and typed variants) in Fortran columns 0–6
- **Typed variants:** `REAL FUNCTION`, `DOUBLE PRECISION FUNCTION`, `COMPLEX FUNCTION`, `LOGICAL FUNCTION`, `INTEGER FUNCTION`
- **Routine extent:** From the declaration line to the corresponding `END` (or `END SUBROUTINE name` / `END FUNCTION name`)

### Fortran 77 Layout

- Columns 1–5: statement label (optional)
- Column 6: continuation character
- Columns 7–72: statement

The regex allows 0–6 leading characters (label or spaces) before the routine keyword.

### Size Limit

- Chunks are capped at **4,000 characters**
- Oversized chunks are truncated with a `[truncated]` note

---

## Fallback Strategy: Fixed-Size + Overlap

Used when a file contains **no** subroutine/function declarations (headers, includes, etc.).

- **Chunk size:** 50 lines
- **Overlap:** 5 lines between consecutive chunks
- **Size limit:** Same 4,000-character cap and truncation as the primary strategy

---

## Metadata per Chunk

| Field | Description |
|-------|-------------|
| `file_path` | Full path to the source file |
| `file_name` | Basename of the file |
| `routine_name` | e.g. `DGEMM`, `DAXPY` (empty for fixed-size chunks) |
| `start_line` | First line of the chunk (1-based) |
| `end_line` | Last line of the chunk (1-based) |
| `precision` | Inferred from routine name: `single` (S), `double` (D), `complex` (C), `double_complex` (Z), or `unknown` |
| `operation_type` | Inferred from suffix: e.g. `matrix_matrix_multiply` (GEMM), `dot_product` (DOT), `norm` (NRM2), `scaled_vector_addition` (AXPY) |
| `description` | First comment block above the routine, max 500 chars |
| `chunk_type` | `"function"` or `"fixed_size"` |

---

## Operation Type Inference

BLAS routine names follow `{precision}{operation}`. The chunker infers `operation_type` from the suffix. Examples:

- GEMM → matrix_matrix_multiply
- GEMV → matrix_vector_multiply
- DOT → dot_product
- NRM2 → norm
- AXPY → scaled_vector_addition
- COPY → vector_copy
- SCAL → vector_scale
- TRSV → triangular_solve_vector

See `OPERATION_TYPES` in `chunker.py` for the full mapping.

---

## Parameters

| Constant | Value | Purpose |
|----------|-------|---------|
| `CHUNK_SIZE_LIMIT` | 4000 | Max characters per chunk |
| `FIXED_CHUNK_LINES` | 50 | Lines per fixed-size chunk |
| `FIXED_CHUNK_OVERLAP` | 5 | Overlap between fixed-size chunks |
| `DESCRIPTION_MAX` | 500 | Max length of extracted description |

---

## BLAS Usage

For the Reference LAPACK BLAS corpus, every source file contains at least one routine. The primary (function-level) strategy covers all chunks; the fixed-size fallback is unused in practice.
