"""
Run all three performance targets locally and log results:
  1. Ingestion throughput: time full ingest, count LOC, print LOC/s and elapsed.
  2. Codebase coverage: compare source .f file count vs chunks in ChromaDB; flag files with zero chunks.
  3. Retrieval precision: run evaluate.py against 15 ground-truth queries; print precision %.

Usage: from legacylens dir, run:
  py run_performance_targets.py

Output: printed to stdout and written to performance_report.txt.
"""

import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure we run from legacylens package context
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def find_fortran_files(root: Path) -> list[Path]:
    """Recursively find .f, .f90, .for files."""
    exts = {".f", ".f90", ".for"}
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def count_loc(files: list[Path], read_file_fn) -> int:
    """Count total lines across all files."""
    total = 0
    for p in files:
        try:
            total += len(read_file_fn(p).splitlines())
        except Exception:
            pass
    return total


def run_ingestion_throughput(log_lines: list[str]) -> tuple[int, int, float, float]:
    """
    Run ingest.py, time it, parse file count and chunk count from stdout.
    Returns (num_source_files, total_chunks, elapsed_seconds, loc_per_second).
    """
    from ingest import BLAS_DIR, read_file

    files = find_fortran_files(BLAS_DIR)
    total_loc = count_loc(files, read_file)
    num_files = len(files)

    log_lines.append("")
    log_lines.append("=== 1. Ingestion Throughput ===")
    log_lines.append(f"Source .f/.f90/.for files: {num_files}")
    log_lines.append(f"Total LOC: {total_loc}")

    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "ingest"],
        cwd=str(SCRIPT_DIR),
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.perf_counter() - start

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode != 0:
        log_lines.append(f"INGEST FAILED (exit {result.returncode})")
        log_lines.append("stderr: " + stderr[:500])
        return num_files, 0, elapsed, 0.0

    # Parse "Found N Fortran files" and "Total chunks: N" or "Total chunk count: N"
    found_match = re.search(r"Found\s+(\d+)\s+Fortran", stdout)
    chunk_match = re.search(r"Total chunks?:\s*(\d+)", stdout) or re.search(
        r"Total chunk count:\s*(\d+)", stdout
    )
    parsed_files = int(found_match.group(1)) if found_match else num_files
    total_chunks = int(chunk_match.group(1)) if chunk_match else 0

    loc_per_sec = (total_loc / elapsed) if elapsed > 0 else 0.0
    log_lines.append(f"Elapsed: {elapsed:.2f}s")
    log_lines.append(f"Total chunks stored: {total_chunks}")
    log_lines.append(f"LOC/sec: {loc_per_sec:.1f}")
    log_lines.append("")

    return parsed_files, total_chunks, elapsed, loc_per_sec


def run_codebase_coverage(
    source_files: list[Path], log_lines: list[str]
) -> list[str]:
    """
    Compare source file count vs chunks in ChromaDB; flag files with zero chunks.
    Returns list of file paths that produced zero chunks.
    """
    from retriever import get_collection

    log_lines.append("=== 2. Codebase Coverage ===")
    source_paths = {str(p.resolve()) for p in source_files}
    log_lines.append(f"Source .f files discovered: {len(source_paths)}")

    try:
        collection = get_collection()
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
    except Exception as e:
        log_lines.append(f"ChromaDB error: {e}")
        return list(source_paths)

    from collections import Counter
    chunks_per_file = Counter(m.get("file_path") for m in metadatas if m.get("file_path"))
    total_chunks = len(metadatas)
    files_with_chunks = len(chunks_per_file)
    log_lines.append(f"Chunks in ChromaDB: {total_chunks}")
    log_lines.append(f"Files with >=1 chunk: {files_with_chunks}")

    zero_chunk = [p for p in source_paths if chunks_per_file.get(p, 0) == 0]
    if zero_chunk:
        log_lines.append(f"Files with ZERO chunks ({len(zero_chunk)}):")
        for p in sorted(zero_chunk):
            log_lines.append(f"  - {p}")
    else:
        log_lines.append("No source files with zero chunks.")
    log_lines.append("")
    return zero_chunk


def run_retrieval_precision(log_lines: list[str]) -> float:
    """Run evaluate.py (15 ground-truth queries), capture precision, append to log."""
    import io
    from contextlib import redirect_stdout

    log_lines.append("=== 3. Retrieval Precision ===")
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            import evaluate
            precision = evaluate.main()
    except Exception as e:
        log_lines.append(f"Evaluate error: {e}")
        return 0.0
    log_lines.append(buf.getvalue())
    log_lines.append("")
    return precision


def main():
    from ingest import BLAS_DIR, read_file

    log_lines: list[str] = []
    log_lines.append("LegacyLens Performance Targets Report")
    log_lines.append(datetime.now().isoformat())
    log_lines.append("=" * 50)

    source_files = find_fortran_files(BLAS_DIR)

    # 1. Ingestion throughput (runs full ingest)
    _, _, _, _ = run_ingestion_throughput(log_lines)

    # 2. Codebase coverage (after ingest)
    run_codebase_coverage(source_files, log_lines)

    # 3. Retrieval precision
    run_retrieval_precision(log_lines)

    report = "\n".join(log_lines)
    print(report)

    out_path = SCRIPT_DIR / "performance_report.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
