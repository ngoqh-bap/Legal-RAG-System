from __future__ import annotations

from legal_rag.core.models import Chunk, Document
from legal_rag.utils.ids import stable_id


def chunk_document(doc: Document, *, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    text = doc.text or ""
    chunks: list[Chunk] = []

    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunk_text = text[i:j].strip()
        if chunk_text:
            chunk_id = stable_id(doc.doc_id, str(i), str(j), chunk_text[:40])
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source_path=doc.source_path,
                    text=chunk_text,
                    start_char=i,
                    end_char=j,
                    metadata=dict(doc.metadata),
                )
            )
        if j >= n:
            break
        i = j - chunk_overlap

    return chunks
