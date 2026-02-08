"""Vector store module using ChromaDB."""

import uuid
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
    embeddings: Embeddings,
    force_reindex: bool = False,
) -> int:
    """
    Index documents into the vector store.

    This function makes the embedding step EXPLICIT so you can see
    exactly when text gets converted into vectors, and when those
    vectors get stored in the database.

    Args:
        vector_store: ChromaDB vector store instance.
        documents: List of Document chunks to index.
        embeddings: Embeddings model to convert text -> vectors.
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
        vector_store._collection.delete(where={})

    # ── Step A: Extract text from document chunks ──────────────────────
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    print(f"\n┌─── STEP A: TEXT CHUNKS READY ──────────────────────────────────")
    print(f"│ We have {len(texts)} text chunks to embed.")
    print(f"│")
    for i, text in enumerate(texts[:3]):
        preview = text[:70].replace("\n", " ")
        print(f"│   Chunk {i+1}: \"{preview}...\"")
    if len(texts) > 3:
        print(f"│   ... and {len(texts) - 3} more chunks")
    print(f"└───────────────────────────────────────────────────────────────────\n")

    # ── Step B: Convert text chunks into embedding vectors ─────────────
    # THIS is where the magic happens — each text chunk becomes a
    # list of numbers (a vector) that captures its meaning.
    print(f"┌─── STEP B: CREATING EMBEDDINGS (text → vectors) ──────────────")
    print(f"│ Sending {len(texts)} chunks to Jina AI embedding model...")
    print(f"│ Model: jina-embeddings-v3")
    print(f"│")

    vectors = embeddings.embed_documents(texts)  # <-- THIS is the embedding call

    dim = len(vectors[0])
    print(f"│ Result: {len(vectors)} vectors, each with {dim} dimensions")
    print(f"│")
    print(f"│   Chunk 1 vector (first 8 values):")
    print(f"│   {[round(v, 4) for v in vectors[0][:8]]}...")
    if len(vectors) > 1:
        print(f"│   Chunk 2 vector (first 8 values):")
        print(f"│   {[round(v, 4) for v in vectors[1][:8]]}...")
    print(f"│")
    print(f"│ Each chunk is now a point in {dim}-dimensional space.")
    print(f"│ Similar text → nearby points. Different text → far apart points.")
    print(f"└───────────────────────────────────────────────────────────────────\n")

    # ── Step C: Store vectors + text in ChromaDB ───────────────────────
    # We store the vectors alongside the original text so we can
    # retrieve the text later when a similar query vector is found.
    print(f"┌─── STEP C: STORING IN VECTOR DATABASE ────────────────────────")
    print(f"│ Inserting {len(vectors)} vectors + their original text into ChromaDB...")

    ids = [str(uuid.uuid4()) for _ in range(len(texts))]
    vector_store._collection.add(
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"│ Stored {len(vectors)} entries. Each entry contains:")
    print(f"│   - The embedding vector ({dim} floats)")
    print(f"│   - The original text chunk")
    print(f"│   - Metadata (source file, page number)")
    print(f"└───────────────────────────────────────────────────────────────────\n")

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
