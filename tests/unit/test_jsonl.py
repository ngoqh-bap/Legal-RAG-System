from pathlib import Path

from legal_rag.core.models import Chunk
from legal_rag.utils.jsonl import read_jsonl, write_jsonl


def test_jsonl_roundtrip(tmp_path: Path):
    items = [
        Chunk(chunk_id="c1", doc_id="d", source_path="x", text="hello"),
        Chunk(chunk_id="c2", doc_id="d", source_path="x", text="world"),
    ]
    p = tmp_path / "corpus.jsonl"
    write_jsonl(p, items)
    loaded = list(read_jsonl(p, Chunk))
    assert [c.chunk_id for c in loaded] == ["c1", "c2"]
    assert loaded[0].text == "hello"
