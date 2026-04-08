from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class LocalHashEmbedder:
    """
    Deterministic, dependency-free embedder.

    Not semantically meaningful, but great for scaffolding the RAG plumbing
    without requiring GPU models or external APIs.
    """

    dim: int = 256

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_hash_to_unit_vector(t, dim=self.dim) for t in texts]


def _hash_to_unit_vector(text: str, *, dim: int) -> list[float]:
    # Expand sha256 stream to dim floats in [-0.5, 0.5], then normalize.
    buf = b""
    counter = 0
    while len(buf) < dim:
        h = hashlib.sha256()
        h.update(text.encode("utf-8", errors="ignore"))
        h.update(counter.to_bytes(4, "little", signed=False))
        buf += h.digest()
        counter += 1
    vals = [((b / 255.0) - 0.5) for b in buf[:dim]]
    return _l2_normalize(vals)


def _l2_normalize(v: list[float]) -> list[float]:
    n2 = sum(x * x for x in v)
    if n2 <= 0:
        return v
    inv = 1.0 / math.sqrt(n2)
    return [x * inv for x in v]
