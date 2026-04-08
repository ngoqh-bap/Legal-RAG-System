from __future__ import annotations

from legal_rag.providers.embeddings import LocalHashEmbedder
from legal_rag.providers.llm import LocalExtractiveGenerator


def get_embedder(name: str):
    name = (name or "").strip().lower()
    if name in {"local_hash", "hash", "local"}:
        return LocalHashEmbedder()
    raise ValueError(f"Unknown embeddings provider: {name}")


def get_generator(name: str):
    name = (name or "").strip().lower()
    if name in {"local_extractive", "extractive", "local"}:
        return LocalExtractiveGenerator()
    raise ValueError(f"Unknown llm provider: {name}")
