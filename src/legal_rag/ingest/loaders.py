from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from legal_rag.core.models import Document
from legal_rag.utils.ids import stable_id


def discover_files(root: str | Path, exts: Iterable[str] = (".txt", ".md", ".pdf")) -> list[Path]:
    root_path = Path(root)
    if root_path.is_file():
        return [root_path]
    exts_lower = {e.lower() for e in exts}
    files: list[Path] = []
    for p in root_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_lower:
            files.append(p)
    return sorted(files)


def load_document(path: str | Path) -> Document:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = p.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        text = _read_pdf(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    doc_id = stable_id(str(p.resolve()))
    return Document(doc_id=doc_id, source_path=str(p.as_posix()), text=text, metadata={"ext": suffix})


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PDF support requires optional dependency. Install with: pip install -e \".[pdf]\""
        ) from e

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)
