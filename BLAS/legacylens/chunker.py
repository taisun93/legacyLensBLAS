"""
Fortran-aware chunking for BLAS source files.
Primary: function-level splitting. Fallback: fixed-size chunks.
"""

import re
from pathlib import Path
from typing import Any


# BLAS operation type inference from routine name suffix
OPERATION_TYPES: dict[str, str] = {
    "GEMM": "matrix_matrix_multiply",
    "GEMV": "matrix_vector_multiply",
    "GBMV": "banded_matrix_vector_multiply",
    "HEMM": "hermitian_matrix_multiply",
    "HEMV": "hermitian_matrix_vector_multiply",
    "HER": "hermitian_rank1_update",
    "HER2": "hermitian_rank2_update",
    "HERK": "hermitian_rankk_update",
    "HER2K": "hermitian_rank2k_update",
    "HPMV": "hermitian_packed_matrix_vector_multiply",
    "HPR": "hermitian_packed_rank1_update",
    "HPR2": "hermitian_packed_rank2_update",
    "SYMM": "symmetric_matrix_multiply",
    "SYMV": "symmetric_matrix_vector_multiply",
    "SYR": "symmetric_rank1_update",
    "SYR2": "symmetric_rank2_update",
    "SYRK": "symmetric_rankk_update",
    "SYR2K": "symmetric_rank2k_update",
    "SPMV": "symmetric_packed_matrix_vector_multiply",
    "SPR": "symmetric_packed_rank1_update",
    "SPR2": "symmetric_packed_rank2_update",
    "TRMM": "triangular_matrix_multiply",
    "TRMV": "triangular_matrix_vector_multiply",
    "TRSM": "triangular_solve_matrix",
    "TRSV": "triangular_solve_vector",
    "TBMV": "triangular_banded_matrix_vector_multiply",
    "TBSV": "triangular_banded_solve",
    "TPMV": "triangular_packed_matrix_vector_multiply",
    "TPSV": "triangular_packed_solve",
    "DOT": "dot_product",
    "DOTC": "dot_product_conjugate",
    "DOTU": "dot_product_unconjugate",
    "NRM2": "norm",
    "ASUM": "sum_of_absolute_values",
    "IAMAX": "index_of_maximum_absolute_value",
    "AXPY": "scaled_vector_addition",
    "COPY": "vector_copy",
    "SWAP": "vector_swap",
    "SCAL": "vector_scale",
    "ROT": "plane_rotation",
    "ROTG": "generate_plane_rotation",
    "ROTM": "modified_plane_rotation",
    "ROTMG": "generate_modified_plane_rotation",
    "GER": "rank1_update",
    "GERC": "rank1_update_conjugate",
    "GERU": "rank1_update_unconjugate",
    "SBMV": "symmetric_banded_matrix_vector_multiply",
    "SKEWSYMM": "skew_symmetric_matrix_multiply",
    "SKEWSYMV": "skew_symmetric_matrix_vector_multiply",
    "SKEWSYR2": "skew_symmetric_rank2_update",
    "SKEWSYR2K": "skew_symmetric_rank2k_update",
}

# Fortran 77: cols 1-5 label, 6 continuation, 7+ statement
# Match SUBROUTINE/FUNCTION (and typed variants) in first 7 columns
ROUTINE_RE = re.compile(
    r"^\s{0,6}"
    r"(SUBROUTINE|REAL\s+FUNCTION|DOUBLE\s+PRECISION\s+FUNCTION|"
    r"COMPLEX\s+FUNCTION|LOGICAL\s+FUNCTION|INTEGER\s+FUNCTION|FUNCTION)\s+"
    r"([A-Z0-9]+)",
    re.IGNORECASE,
)

# END statement for routine (standalone or END SUBROUTINE/FUNCTION, in cols 1-6 area)
END_RE = re.compile(r"^\s{0,6}END(\s+(SUBROUTINE|FUNCTION)\s+\w+)?\s*$", re.IGNORECASE)

CHUNK_SIZE_LIMIT = 4000
FIXED_CHUNK_LINES = 50
FIXED_CHUNK_OVERLAP = 5
DESCRIPTION_MAX = 500


def _infer_precision(routine_name: str) -> str:
    """Infer precision from first letter: S, D, C, Z."""
    if not routine_name:
        return "unknown"
    c = routine_name[0].upper()
    if c == "S":
        return "single"
    if c == "D":
        return "double"
    if c == "C":
        return "complex"
    if c == "Z":
        return "double_complex"
    return "unknown"


def _infer_operation_type(routine_name: str) -> str:
    """Infer operation type from routine name suffix."""
    name_upper = routine_name.upper()
    for suffix, op_type in OPERATION_TYPES.items():
        if name_upper.endswith(suffix):
            return op_type
    return "unknown"


def _extract_description(lines: list[str], start_idx: int) -> str:
    """Extract first comment block above/within routine, max 500 chars."""
    desc_lines: list[str] = []
    # Look backwards for comment block
    for i in range(start_idx - 1, -1, -1):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("*") or stripped.upper().startswith("C"):
            # Get content after comment marker
            if stripped.startswith("*>"):
                content = stripped[2:].strip()
            elif stripped.startswith("*"):
                content = stripped[1:].strip()
            else:
                content = stripped[1:].strip() if len(stripped) > 1 else ""
            if content and not content.startswith("=") and not content.startswith("\\"):
                desc_lines.insert(0, content)
        else:
            break
    desc = " ".join(desc_lines)[:DESCRIPTION_MAX]
    return desc.strip()


def chunk_file(file_path: Path, content: str) -> list[dict[str, Any]]:
    """
    Chunk a Fortran source file. Returns list of chunk dicts with metadata.
    """
    lines = content.splitlines()
    file_name = file_path.name

    # Find routine boundaries
    routines: list[tuple[int, int, str]] = []  # (start_line, end_line, routine_name)
    i = 0
    while i < len(lines):
        m = ROUTINE_RE.match(lines[i])
        if m:
            routine_name = m.group(2).upper()
            start_line = i + 1  # 1-based
            # Find END or next routine
            end_line = len(lines)  # 1-based, inclusive
            for j in range(i + 1, len(lines)):
                if END_RE.match(lines[j]):
                    end_line = j + 1
                    break
                if ROUTINE_RE.match(lines[j]):
                    end_line = j  # next routine at line j+1; this routine ends at line j
                    break
            routines.append((start_line, end_line, routine_name))
            i = end_line if end_line > i + 1 else i + 1
        else:
            i += 1

    chunks: list[dict[str, Any]] = []

    if routines:
        for start_line, end_line, routine_name in routines:
            chunk_lines = lines[start_line - 1 : end_line]
            chunk_text = "\n".join(chunk_lines)
            if len(chunk_text) > CHUNK_SIZE_LIMIT:
                chunk_text = chunk_text[: CHUNK_SIZE_LIMIT - 15] + "\n[truncated]"
            description = _extract_description(lines, start_line - 1)
            chunks.append(
                {
                    "text": chunk_text,
                    "file_path": str(file_path.resolve()),
                    "file_name": file_name,
                    "routine_name": routine_name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "precision": _infer_precision(routine_name),
                    "operation_type": _infer_operation_type(routine_name),
                    "description": description,
                    "chunk_type": "function",
                }
            )
    else:
        # Fallback: fixed-size chunks
        overlap = FIXED_CHUNK_OVERLAP
        size = FIXED_CHUNK_LINES
        pos = 0
        chunk_num = 0
        while pos < len(lines):
            end = min(pos + size, len(lines))
            chunk_lines = lines[pos:end]
            chunk_text = "\n".join(chunk_lines)
            if len(chunk_text) > CHUNK_SIZE_LIMIT:
                chunk_text = chunk_text[: CHUNK_SIZE_LIMIT - 15] + "\n[truncated]"
            start_line = pos + 1
            end_line = end
            chunks.append(
                {
                    "text": chunk_text,
                    "file_path": str(file_path.resolve()),
                    "file_name": file_name,
                    "routine_name": "",
                    "start_line": start_line,
                    "end_line": end_line,
                    "precision": "unknown",
                    "operation_type": "unknown",
                    "description": "",
                    "chunk_type": "fixed_size",
                }
            )
            chunk_num += 1
            pos = end - overlap if end < len(lines) else len(lines)

    return chunks
