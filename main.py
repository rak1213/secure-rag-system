"""
RAG (Retrieval Augmented Generation) System for Document Q&A.

Usage:
    Interactive mode:     python main.py
    Batch queries:        python main.py --batch
    Start API server:     python main.py --serve
    Force re-index:       python main.py --reindex
"""

import argparse
from pathlib import Path

from src.config import get_settings
from src.logging_config import setup_logging, get_logger
from src.llm_factory import create_llm
from src.embedding_factory import create_embeddings
from src.vector_store import get_vector_store, index_documents, is_indexed
from src.document_loader import load_pdf_documents
from src.text_splitter import split_documents
from src.rag_chain import RAGChain

log = get_logger(__name__)


# ============================================================================
# BATCH QUERIES - Modify this list to run multiple queries at once
# ============================================================================
QUERIES = [
    "What is Crosswalk guards?",
    "What to do if moving through an intersection with a green signal?",
    "What to do when approached by an emergency vehicle?",
]


def setup_rag_system(force_reindex: bool = False) -> RAGChain:
    """Set up the complete RAG system."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level, log_format=settings.log_format)

    log.info("setup.start", llm_provider=settings.llm_provider, embedding_provider=settings.embedding_provider)

    # Initialize components via factories
    llm = create_llm(settings)
    embeddings = create_embeddings(settings)
    vector_store = get_vector_store(embeddings)

    # Check if we need to index documents
    if not is_indexed(vector_store) or force_reindex:
        documents = load_pdf_documents("data")
        chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
        index_documents(vector_store, chunks, embeddings, force_reindex=force_reindex)
    else:
        log.info("setup.index.skip", reason="already_indexed")

    rag_chain = RAGChain(
        vector_store=vector_store,
        embeddings=embeddings,
        llm=llm,
        num_chunks=settings.num_retrieval_chunks,
    )

    log.info("setup.complete")
    return rag_chain


def run_interactive_mode(rag_chain: RAGChain) -> None:
    """Run the RAG system in interactive CLI mode."""
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
    """Run the RAG system with a batch of queries."""
    print(f"\nBatch Mode - Processing {len(queries)} queries\n")

    for i, query in enumerate(queries, 1):
        print("=" * 60)
        print(f"Query {i}/{len(queries)}: {query}")
        print("=" * 60)

        response = rag_chain.query(query)

        print("\nANSWER:")
        print(response.format_full_response())
        print()


def run_server() -> None:
    """Start the FastAPI server."""
    import uvicorn
    from src.api.app import create_app

    settings = get_settings()
    app = create_app()

    log.info("server.start", host=settings.api_host, port=settings.api_port)
    print(f"\nStarting API server at http://{settings.api_host}:{settings.api_port}")
    print(f"API docs: http://localhost:{settings.api_port}/docs")
    print(f"Health:   http://localhost:{settings.api_port}/api/v1/health\n")

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


def main() -> None:
    """Main entry point for the RAG system."""
    parser = argparse.ArgumentParser(
        description="RAG System for Document Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                 # Interactive mode
  python main.py --batch         # Run batch queries
  python main.py --serve         # Start API server
  python main.py --reindex       # Force re-indexing of documents
        """,
    )
    parser.add_argument("--batch", action="store_true", help="Run in batch mode with predefined queries")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing of documents")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI REST API server")

    args = parser.parse_args()

    # Server mode doesn't need local data check (handled during startup)
    if args.serve:
        run_server()
        return

    # Verify data directory exists for CLI modes
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
    main()
