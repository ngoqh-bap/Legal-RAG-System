from legal_rag.core.models import Document
from legal_rag.ingest.chunking import chunk_document


def test_chunk_document_respects_overlap():
    doc = Document(doc_id="d1", source_path="x.txt", text="a" * 1000)
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=50)

    assert len(chunks) >= 5
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 200
    assert chunks[1].start_char == 150
    assert chunks[1].end_char == 350
