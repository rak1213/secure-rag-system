"""Vector store module using ChromaDB."""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def get_vector_store(
    embeddings: Embeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "rag_documents",
) -> Chroma:
    """
    Create or load a ChromaDB vector store.

    Args:
        embeddings: Embeddings instance for vectorizing documents.
        persist_directory: Directory to persist the database.
        collection_name: Name of the collection in ChromaDB.

    Returns:
        Configured Chroma vector store instance.
    """
    # Ensure persist directory exists
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    return vector_store


def index_documents(
    vector_store: Chroma,
    documents: list[Document],
    force_reindex: bool = False,
) -> int:
    """
    Index documents into the vector store.

    Args:
        vector_store: ChromaDB vector store instance.
        documents: List of Document chunks to index.
        force_reindex: If True, clear existing documents first.

    Returns:
        Number of documents indexed.
    """
    # Check if documents already exist
    existing_count = vector_store._collection.count()

    if existing_count > 0 and not force_reindex:
        print(f"Vector store already contains {existing_count} documents. Skipping indexing.")
        print("Use force_reindex=True to re-index documents.")
        return existing_count

    if force_reindex and existing_count > 0:
        print(f"Clearing {existing_count} existing documents...")
        # Clear the collection by deleting all documents
        vector_store._collection.delete(where={})

    print(f"Indexing {len(documents)} document chunks...")
    vector_store.add_documents(documents)
    print(f"Successfully indexed {len(documents)} chunks")

    return len(documents)


def is_indexed(vector_store: Chroma) -> bool:
    """
    Check if the vector store has any indexed documents.

    Args:
        vector_store: ChromaDB vector store instance.

    Returns:
        True if documents are indexed, False otherwise.
    """
    return vector_store._collection.count() > 0
