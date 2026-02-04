"""
RAG (Retrieval Augmented Generation) System for Document Q&A.

This module provides the main entry point for the RAG system.
It supports both interactive CLI mode and batch query mode.

Usage:
    Interactive mode:
        python main.py

    Batch queries (modify the QUERIES list below):
        python main.py --batch
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.document_loader import load_pdf_documents
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import get_vector_store, index_documents, is_indexed
from src.rag_chain import RAGChain


# Load environment variables from .env file
load_dotenv()


# ============================================================================
# BATCH QUERIES - Modify this list to run multiple queries at once
# ============================================================================
QUERIES = [
    "What is Crosswalk guards?",
    "What to do if moving through an intersection with a green signal?",
   " what to do when approached by an emergency vehicle?"
]


def setup_rag_system(force_reindex: bool = False) -> RAGChain:
    """
    Set up the complete RAG system.

    This includes loading documents, creating embeddings,
    indexing into ChromaDB, and initializing the RAG chain.

    Args:
        force_reindex: If True, re-index documents even if already indexed.

    Returns:
        Configured RAGChain instance ready for queries.
    """
    print("=" * 60)
    print("Setting up RAG System")
    print("=" * 60)

    # Initialize embeddings
    print("\n[1/4] Initializing Jina AI embeddings...")
    embeddings = get_embeddings()

    # Initialize vector store
    print("\n[2/4] Initializing ChromaDB vector store...")
    vector_store = get_vector_store(embeddings)

    # Check if we need to index documents
    if not is_indexed(vector_store) or force_reindex:
        # Load PDF documents
        print("\n[3/4] Loading PDF documents from data/ directory...")
        documents = load_pdf_documents("data")

        # Split documents into chunks
        print("\n[4/4] Splitting documents into chunks...")
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)

        # Index documents
        print("\nIndexing documents into vector store...")
        index_documents(vector_store, chunks, force_reindex=force_reindex)
    else:
        print("\n[3/4] Documents already indexed. Skipping document loading.")
        print("[4/4] Skipping text splitting.")

    # Initialize RAG chain
    print("\nInitializing RAG chain with Gemini...")
    rag_chain = RAGChain(vector_store)

    print("\n" + "=" * 60)
    print("RAG System Ready!")
    print("=" * 60)

    return rag_chain


def run_interactive_mode(rag_chain: RAGChain) -> None:
    """
    Run the RAG system in interactive CLI mode.

    Args:
        rag_chain: Configured RAGChain instance.
    """
    print("\nInteractive Mode - Ask questions about your documents")
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        try:
            query = input("\nYour question: ").strip()

            if not query:
                print("Please enter a question.")
                continue

            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            print("\nSearching documents and generating answer...\n")
            response = rag_chain.query(query)

            print("-" * 50)
            print("ANSWER:")
            print("-" * 50)
            print(response.format_full_response())
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


def run_batch_mode(rag_chain: RAGChain, queries: list[str]) -> None:
    """
    Run the RAG system with a batch of queries.

    Args:
        rag_chain: Configured RAGChain instance.
        queries: List of questions to answer.
    """
    print(f"\nBatch Mode - Processing {len(queries)} queries\n")

    for i, query in enumerate(queries, 1):
        print("=" * 60)
        print(f"Query {i}/{len(queries)}: {query}")
        print("=" * 60)

        response = rag_chain.query(query)

        print("\nANSWER:")
        print(response.format_full_response())
        print()


def main() -> None:
    """Main entry point for the RAG system."""
    parser = argparse.ArgumentParser(
        description="RAG System for Document Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                 # Interactive mode
  python main.py --batch         # Run batch queries (modify QUERIES list in main.py)
  python main.py --reindex       # Force re-indexing of documents
        """,
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode with predefined queries",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of documents",
    )

    args = parser.parse_args()

    # Verify data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        print("Error: 'data/' directory not found.")
        print("Please create a 'data/' directory and add PDF files.")
        return

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("Error: No PDF files found in 'data/' directory.")
        print("Please add PDF files to the 'data/' directory.")
        return

    print(f"Found {len(pdf_files)} PDF file(s) in data/:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    # Setup RAG system
    rag_chain = setup_rag_system(force_reindex=args.reindex)

    # Run in selected mode
    if args.batch:
        run_batch_mode(rag_chain, QUERIES)
    else:
        run_interactive_mode(rag_chain)


if __name__ == "__main__":
    # main()
    rag_chain = setup_rag_system(force_reindex=False)
    run_batch_mode(rag_chain, QUERIES)