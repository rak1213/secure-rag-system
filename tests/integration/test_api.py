"""Integration tests for the FastAPI API using TestClient."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.config import Settings
from src.logger import SecurityLogger
from src.secure_rag_chain import SecureRAGResponse
from src.api.app import create_app
from src.api import dependencies


@pytest.fixture
def mock_chain():
    """Mock SecureRAGChain for API testing."""
    chain = MagicMock()
    chain.query.return_value = SecureRAGResponse(
        answer="The speed limit on highways is 110 km/h.",
        sources=["[Source: driving-rules.pdf, Page 5]"],
        was_blocked=False,
        faithfulness_score=0.95,
        retrieval_scores=[0.85, 0.72],
    )
    chain.vector_store._collection.count.return_value = 42
    return chain


@pytest.fixture
def test_settings():
    return Settings(
        google_api_key="test-key",
        jina_api_key="test-key",
        api_key="",
        log_level="WARNING",
    )


@pytest.fixture
def client(mock_chain, test_settings):
    """Test client with mocked dependencies (set after lifespan)."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        # Override deps AFTER lifespan runs so mocks aren't overwritten
        dependencies.set_rag_chain(mock_chain)
        dependencies.set_security_logger(SecurityLogger())
        dependencies.set_settings(test_settings)
        yield client

    dependencies._secure_chain = None
    dependencies._security_logger = None
    dependencies._settings = None


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["document_count"] == 42

    def test_health_shows_providers(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["llm_provider"] == "gemini"
        assert data["embedding_provider"] == "jina"


class TestQueryEndpoint:
    def test_query_returns_answer(self, client):
        response = client.post("/api/v1/query", json={"question": "What are the speed limits?"})
        assert response.status_code == 200
        data = response.json()
        assert "110 km/h" in data["answer"]
        assert data["was_blocked"] is False

    def test_query_empty_rejected(self, client):
        response = client.post("/api/v1/query", json={"question": ""})
        assert response.status_code == 422

    def test_query_too_long_rejected(self, client):
        response = client.post("/api/v1/query", json={"question": "x" * 501})
        assert response.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_returns_data(self, client):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data
        assert "queries_blocked" in data


class TestAPIKeyAuth:
    def test_no_auth_when_key_not_set(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_auth_required_when_key_set(self, mock_chain):
        """When API_KEY is configured, requests need X-API-Key header."""
        app = create_app()
        auth_settings = Settings(
            google_api_key="test", jina_api_key="test",
            api_key="secret-key",
        )

        with TestClient(app, raise_server_exceptions=False) as client:
            # Set after lifespan
            dependencies.set_rag_chain(mock_chain)
            dependencies.set_security_logger(SecurityLogger())
            dependencies.set_settings(auth_settings)

            # No key
            response = client.post("/api/v1/query", json={"question": "test"})
            assert response.status_code == 401

            # Wrong key
            response = client.post(
                "/api/v1/query",
                json={"question": "test"},
                headers={"X-API-Key": "wrong"},
            )
            assert response.status_code == 403

            # Correct key
            response = client.post(
                "/api/v1/query",
                json={"question": "test"},
                headers={"X-API-Key": "secret-key"},
            )
            assert response.status_code == 200

        dependencies._secure_chain = None
        dependencies._security_logger = None
        dependencies._settings = None
