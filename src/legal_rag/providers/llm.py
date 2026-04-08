from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Generator(Protocol):
    def generate(self, *, question: str, contexts: list[str]) -> str: ...


@dataclass(frozen=True)
class LocalExtractiveGenerator:
    """
    Local baseline generator that answers by selecting/quoting the most relevant context.

    This is intentionally simple so the project can run without external LLMs.
    """

    max_chars: int = 1200

    def generate(self, *, question: str, contexts: list[str]) -> str:
        if not contexts:
            return "I couldn't find relevant context in the corpus."

        # In a real RAG system you'd pass these contexts to an LLM. Here, we provide a
        # pragmatic default: return the top context (trimmed) and hint that it's extractive.
        best = contexts[0].strip()
        if len(best) > self.max_chars:
            best = best[: self.max_chars].rstrip() + "…"

        return (
            "Extracted answer (top retrieved passage):\n\n"
            f"{best}\n\n"
            "Note: configure an LLM provider for synthesized answers."
        )
