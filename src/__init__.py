"""RAG (Retrieval Augmented Generation) System for Document Q&A."""

from .document_loader import load_pdf_documents
from .text_splitter import split_documents
from .embeddings import get_embeddings
from .vector_store import get_vector_store, index_documents, is_indexed
from .retriever import format_docs_with_sources
from .rag_chain import RAGChain, RAGResponse
from .config import Settings, get_settings
from .llm_factory import create_llm
from .embedding_factory import create_embeddings
from .logging_config import setup_logging, get_logger

__all__ = [
    "load_pdf_documents",
    "split_documents",
    "get_embeddings",
    "get_vector_store",
    "index_documents",
    "is_indexed",
    "format_docs_with_sources",
    "RAGChain",
    "RAGResponse",
    "Settings",
    "get_settings",
    "create_llm",
    "create_embeddings",
    "setup_logging",
    "get_logger",
]
