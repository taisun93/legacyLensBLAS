"""
One-time ingestion pipeline: read BLAS files → chunk → embed → store in ChromaDB.
Run once; queries read from persisted chroma_db/.
"""

from pathlib import Path

from chunker import chunk_file
from embedder import embed_chunks
from retriever import get_collection, add_chunks

BLAS_DIR = Path(__file__).resolve().parent / "blas"
BATCH_SIZE = 80  # Voyage max 120K tokens/batch; ~1.2K tokens/chunk


def find_fortran_files(root: Path) -> list[Path]:
    """Recursively find .f, .f90, .for files."""
    exts = {".f", ".f90", ".for"}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def read_file(path: Path) -> str:
    """Read file with latin-1 fallback for older Fortran."""
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1", errors="replace")


def main():
    print("LegacyLens Ingestion Pipeline")
    print("=" * 40)

    files = find_fortran_files(BLAS_DIR)
    print(f"Found {len(files)} Fortran files in {BLAS_DIR}")

    all_chunks = []
    for path in files:
        content = read_file(path)
        chunks = chunk_file(path, content)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    if all_chunks:
        # Sanity check: print first chunk
        c = all_chunks[0]
        print("\n--- First chunk (sanity check) ---")
        print(f"File: {c['file_name']} lines {c['start_line']}-{c['end_line']}")
        print(f"Routine: {c['routine_name']} ({c['chunk_type']})")
        print(f"Precision: {c['precision']}, Op: {c['operation_type']}")
        preview = c["text"][:400].replace("\n", "\n  ")
        print(f"Preview:\n  {preview}...")
        print("---\n")

    print(f"Embedding chunks (batch size {BATCH_SIZE})...")
    texts = [ch["text"] for ch in all_chunks]
    embeddings = embed_chunks(texts, batch_size=BATCH_SIZE)
    print(f"Embedded {len(embeddings)} chunks.")

    print("Storing in ChromaDB...")
    collection = get_collection(reset=True)
    add_chunks(all_chunks, embeddings, collection)
    print("Done.")

    print(f"\nTotal chunk count: {len(all_chunks)}")


if __name__ == "__main__":
    main()
