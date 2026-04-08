from __future__ import annotations

from pathlib import Path

from legal_rag.core.models import Chunk
from legal_rag.ingest.chunking import chunk_document
from legal_rag.ingest.loaders import discover_files, load_document
from legal_rag.utils.jsonl import write_jsonl


def ingest_to_corpus(
    input_path: str | Path,
    *,
    out_path: str | Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    files = discover_files(input_path)
    all_chunks: list[Chunk] = []
    for f in files:
        doc = load_document(f)
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    write_jsonl(out_path, all_chunks)
    return all_chunks
