"""
Splits extracted document text into overlapping chunks for embedding.

Reuses the chunk_text(text, chunk_size, overlap) approach from Phase 2
topic 4, sized for real financial documents instead of the small
illustrative sizes used during practice.

Chunk size is chosen deliberately generous: extraction.py's pypdf
output keeps a table's period-context header (e.g. "Three Months
Ended June 27, 2026") naturally adjacent to the line items beneath it
in the source text, and a large enough chunk_size keeps that pairing
intact more often. This is still plain character-based chunking, not
table-aware chunking -- a reasonable default, not a guarantee that
every chunk retains full context.
"""

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 150


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of chunk_size characters. Each
    chunk overlaps the previous one by `overlap` characters, so a fact
    sitting near a chunk boundary is still likely to appear whole in
    at least one chunk rather than getting sliced in half.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
