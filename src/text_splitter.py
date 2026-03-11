"""Text splitter module for chunking documents."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .logging_config import get_logger

log = get_logger(__name__)


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into smaller chunks for embedding and retrieval.

    Args:
        documents: List of Document objects to split.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of Document objects representing the chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    log.info("text_splitter.done", input_docs=len(documents), output_chunks=len(chunks),
             chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunks
