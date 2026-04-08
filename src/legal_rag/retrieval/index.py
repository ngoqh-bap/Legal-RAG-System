from __future__ import annotations

from dataclasses import dataclass

from legal_rag.core.models import Chunk, RetrievedChunk
from legal_rag.providers.embeddings import Embedder


@dataclass
class InMemoryVectorIndex:
    chunks: list[Chunk]
    embeddings: list[list[float]]

    @classmethod
    def build(cls, *, chunks: list[Chunk], embedder: Embedder) -> InMemoryVectorIndex:
        vectors = embedder.embed_texts([c.text for c in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedder returned incorrect number of vectors.")
        return cls(chunks=chunks, embeddings=vectors)

    def search(self, *, query: str, embedder: Embedder, top_k: int) -> list[RetrievedChunk]:
        q = embedder.embed_texts([query])[0]
        scored: list[tuple[float, int]] = []
        for i, v in enumerate(self.embeddings):
            scored.append((_dot(q, v), i))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[RetrievedChunk] = []
        for score, idx in scored[: max(top_k, 0)]:
            out.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))
        return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))
