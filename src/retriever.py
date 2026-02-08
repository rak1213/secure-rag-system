"""Retriever module for formatting retrieved documents with source citations."""

from dataclasses import dataclass


@dataclass
class SourceInfo:
    """
    Source information for a retrieved document.

    Attributes:
        source_file: Name of the source PDF file.
        page_number: Page number in the original document (1-indexed).
    """
    source_file: str
    page_number: int

    def format_citation(self) -> str:
        """Format the source citation for display."""
        return f"[Source: {self.source_file}, Page {self.page_number}]"


def extract_source_info(metadata: dict) -> SourceInfo:
    """
    Extract source information from document metadata.

    Args:
        metadata: Document metadata dictionary.

    Returns:
        SourceInfo with file and page information.
    """
    source_file = metadata.get("source_file", metadata.get("source", "Unknown"))
    # Handle path in source field
    if "/" in str(source_file):
        source_file = str(source_file).split("/")[-1]
    # Page numbers in PyPDF are 0-indexed, convert to 1-indexed
    page_number = metadata.get("page", 0) + 1

    return SourceInfo(source_file=source_file, page_number=page_number)


def format_docs_with_sources(docs: list) -> tuple[str, list[str]]:
    """
    Format retrieved documents as context string with source citations.

    Args:
        docs: List of Document objects from retriever.

    Returns:
        Tuple of (formatted context string, list of unique citation strings).
    """
    context_parts: list[str] = []
    citations: list[str] = []
    seen_citations: set[str] = set()

    for i, doc in enumerate(docs, 1):
        source_info = extract_source_info(doc.metadata)
        citation = source_info.format_citation()

        context_parts.append(
            f"--- Chunk {i} {citation} ---\n{doc.page_content}"
        )

        if citation not in seen_citations:
            citations.append(citation)
            seen_citations.add(citation)

    context = "\n\n".join(context_parts)
    return context, citations
