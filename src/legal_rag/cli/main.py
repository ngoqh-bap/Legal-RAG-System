from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from legal_rag.core.models import Chunk
from legal_rag.core.settings import load_settings
from legal_rag.generation.rag import answer_question
from legal_rag.ingest.pipeline import ingest_to_corpus
from legal_rag.providers.registry import get_embedder, get_generator
from legal_rag.utils.jsonl import read_jsonl

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def ingest(
    input_path: str = typer.Argument(..., help="File or directory to ingest (txt/md/pdf)."),
    out: str = typer.Option("data/processed/corpus.jsonl", "--out", help="Output JSONL corpus path."),
    config: str | None = typer.Option(None, "--config", help="Optional YAML config path."),
):
    """Load documents, chunk them, and write a JSONL corpus."""
    settings = load_settings(config)
    chunks = ingest_to_corpus(
        input_path,
        out_path=out,
        chunk_size=settings.ingest.chunk_size,
        chunk_overlap=settings.ingest.chunk_overlap,
    )
    console.print(f"Wrote [bold]{len(chunks)}[/bold] chunks to [bold]{out}[/bold].")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask."),
    corpus: str | None = typer.Option(None, "--corpus", help="Path to corpus JSONL."),
    config: str | None = typer.Option(None, "--config", help="Optional YAML config path."),
    top_k: int | None = typer.Option(None, "--top-k", help="Override top_k retrieval."),
):
    """Run retrieval + generation over an existing corpus JSONL."""
    settings = load_settings(config)
    corpus_path = corpus or settings.paths.default_corpus_path
    p = Path(corpus_path)
    if not p.exists():
        raise typer.BadParameter(f"Corpus not found: {corpus_path}. Run `legal-rag ingest ...` first.")

    chunks = list(read_jsonl(p, Chunk))
    embedder = get_embedder(settings.providers.embeddings)
    generator = get_generator(settings.providers.llm)
    k = top_k if top_k is not None else settings.retrieval.top_k

    result = answer_question(
        question=question,
        chunks=chunks,
        embedder=embedder,
        generator=generator,
        top_k=k,
    )

    console.rule("[bold]Answer[/bold]")
    console.print(result.answer)

    if result.citations:
        table = Table(title="Citations (top retrieved chunks)")
        table.add_column("score", justify="right")
        table.add_column("source")
        table.add_column("chunk_id")
        for r in result.citations:
            table.add_row(f"{r.score:.4f}", r.chunk.source_path, r.chunk.chunk_id)
        console.print(table)
