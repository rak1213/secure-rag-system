"""Tests for text splitter module."""

from langchain_core.documents import Document

from src.text_splitter import split_documents


class TestSplitDocuments:
    def test_splits_long_document(self):
        long_text = "This is a sentence. " * 200
        docs = [Document(page_content=long_text, metadata={"source": "test.pdf"})]
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1

    def test_preserves_metadata(self):
        docs = [Document(page_content="Short text.", metadata={"source": "test.pdf", "page": 0})]
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=100)
        assert chunks[0].metadata["source"] == "test.pdf"

    def test_short_document_not_split(self):
        docs = [Document(page_content="Short.", metadata={"source": "test.pdf"})]
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=100)
        assert len(chunks) == 1
