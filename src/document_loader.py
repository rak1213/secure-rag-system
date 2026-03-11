"""Document loader module for loading PDF files."""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .logging_config import get_logger

log = get_logger(__name__)


def load_pdf_documents(data_dir: str = "data") -> list[Document]:
    """Load all PDF documents from the specified directory.

    Args:
        data_dir: Path to the directory containing PDF files.

    Returns:
        List of Document objects with page content and metadata.

    Raises:
        FileNotFoundError: If the data directory doesn't exist.
        ValueError: If no PDF files are found in the directory.
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pdf_files = list(data_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {data_dir}")

    all_documents: list[Document] = []

    for pdf_file in pdf_files:
        log.info("document_loader.loading", file=pdf_file.name)
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()

        for doc in documents:
            doc.metadata["source_file"] = pdf_file.name

        all_documents.extend(documents)

    log.info("document_loader.done", pages=len(all_documents), files=len(pdf_files))
    return all_documents
