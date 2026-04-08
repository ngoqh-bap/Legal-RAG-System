# Legal RAG System (Python)

Base Python project structure for a Retrieval-Augmented Generation (RAG) system focused on legal documents.

## Quickstart

Create a virtual environment, install, and run the CLI:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -e ".[dev]"

legal-rag --help
```

## Minimal end-to-end demo

1) Put some `.txt` files into `data/raw/` (example: `data/raw/contract.txt`).

You can also pull raw source documents from Hugging Face datasets into `data/raw/`:

```bash
pip install -e ".[hf]"

# Download to: data/raw/hf/th1nhng0__vietnamese-legal-documents/default/<split>.jsonl
python scripts/hf_download_to_raw.py --dataset "th1nhng0/vietnamese-legal-documents"

# Download to: data/raw/hf/fcsn37__vietnamese-stopwords/default/<split>.jsonl
python scripts/hf_download_to_raw.py --dataset "fcsn37/vietnamese-stopwords"
```

2) Ingest and query:

```bash
legal-rag ingest data/raw --out data/processed/corpus.jsonl
legal-rag ask "What is the termination clause?" --corpus data/processed/corpus.jsonl
```

This repo ships with a **local, dependency-free** default embedder and generator so the pipeline works out of
the box. Swap providers later (OpenAI / sentence-transformers / FAISS, etc.).

## Project layout

- `src/legal_rag/`: library code
- `src/legal_rag/cli/`: `legal-rag` CLI
- `configs/`: config templates
- `data/raw/`: source documents (not committed)
- `data/processed/`: extracted + chunked corpus
- `tests/`: unit + integration tests

## Notes

- **No secrets committed**: copy `.env.example` to `.env` and fill in keys if/when you enable hosted LLMs.