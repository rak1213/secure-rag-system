"""Vector store module using ChromaDB."""

import uuid
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .logging_config import get_logger

log = get_logger(__name__)


def get_vector_store(
    embeddings: Embeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "rag_documents",
) -> Chroma:
    """Create or load a ChromaDB vector store."""
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    log.info("vector_store.loaded", persist_directory=persist_directory, collection=collection_name)
    return vector_store


def index_documents(
    vector_store: Chroma,
    documents: list[Document],
    embeddings: Embeddings,
    force_reindex: bool = False,
) -> int:
    """Index documents into the vector store.

    Args:
        vector_store: ChromaDB vector store instance.
        documents: List of Document chunks to index.
        embeddings: Embeddings model to convert text -> vectors.
        force_reindex: If True, clear existing documents first.

    Returns:
        Number of documents indexed.
    """
    existing_count = vector_store._collection.count()

    if existing_count > 0 and not force_reindex:
        log.info("vector_store.index.skip", existing_count=existing_count)
        return existing_count

    if force_reindex and existing_count > 0:
        log.info("vector_store.index.clearing", existing_count=existing_count)
        vector_store._collection.delete(where={})

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    log.info("vector_store.index.embedding", num_chunks=len(texts))

    vectors = embeddings.embed_documents(texts)
    dim = len(vectors[0])

    log.info("vector_store.index.embedded", num_vectors=len(vectors), dimensions=dim)

    ids = [str(uuid.uuid4()) for _ in range(len(texts))]
    vector_store._collection.add(
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
        ids=ids,
    )

    log.info("vector_store.index.done", stored=len(vectors), dimensions=dim)
    return len(documents)


def is_indexed(vector_store: Chroma) -> bool:
    """Check if the vector store has any indexed documents."""
    return vector_store._collection.count() > 0
