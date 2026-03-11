"""Shared test fixtures — all tests use mocked LLM/embeddings (no API keys needed)."""

import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.config import Settings
from src.logger import SecurityLogger


@pytest.fixture
def settings():
    """Test settings with no real API keys."""
    return Settings(
        llm_provider="gemini",
        llm_model="gemini-2.5-flash",
        google_api_key="test-key",
        jina_api_key="test-key",
        log_level="DEBUG",
        log_format="console",
        enable_tracing=False,
    )


@pytest.fixture
def mock_llm():
    """Mock LLM that returns predictable responses."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Test answer about driving rules.")

    # Make it work with LCEL chains (prompt | llm | parser)
    llm.__or__ = lambda self, other: MagicMock(
        invoke=lambda inputs: "Test answer about driving rules.",
        __or__=lambda self, other: MagicMock(
            invoke=lambda inputs: "Test answer about driving rules."
        ),
    )

    return llm


@pytest.fixture
def mock_embeddings():
    """Mock embeddings that return consistent vectors."""
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1] * 768
    embeddings.embed_documents.return_value = [[0.1] * 768]
    return embeddings


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            page_content="When a school bus has its red lights flashing, you must stop.",
            metadata={"source_file": "driving-rules.pdf", "page": 0},
        ),
        Document(
            page_content="Pedestrians have the right of way at crosswalks.",
            metadata={"source_file": "driving-rules.pdf", "page": 1},
        ),
        Document(
            page_content="Emergency vehicles with sirens have priority. Pull over to the right.",
            metadata={"source_file": "driving-rules.pdf", "page": 2},
        ),
    ]


@pytest.fixture
def security_logger():
    """Fresh security logger for testing."""
    return SecurityLogger()


@pytest.fixture
def docs_with_scores(sample_documents):
    """Sample documents with relevance scores."""
    return [
        (sample_documents[0], 0.85),
        (sample_documents[1], 0.72),
        (sample_documents[2], 0.65),
    ]
