from __future__ import annotations

from legal_rag.core.models import Answer, Chunk
from legal_rag.providers.embeddings import Embedder
from legal_rag.providers.llm import Generator
from legal_rag.retrieval.index import InMemoryVectorIndex


def answer_question(
    *,
    question: str,
    chunks: list[Chunk],
    embedder: Embedder,
    generator: Generator,
    top_k: int,
) -> Answer:
    index = InMemoryVectorIndex.build(chunks=chunks, embedder=embedder)
    retrieved = index.search(query=question, embedder=embedder, top_k=top_k)
    contexts = [r.chunk.text for r in retrieved]
    answer = generator.generate(question=question, contexts=contexts)
    return Answer(question=question, answer=answer, citations=retrieved)
