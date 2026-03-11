"""Tests for configuration loading and validation."""

import pytest
from pydantic import ValidationError

from src.config import Settings


class TestSettings:
    def test_defaults(self):
        settings = Settings(google_api_key="test", jina_api_key="test")
        assert settings.llm_provider == "gemini"
        assert settings.llm_model == "gemini-2.5-flash"
        assert settings.llm_temperature == 0.1
        assert settings.embedding_provider == "jina"
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.num_retrieval_chunks == 4
        assert settings.confidence_threshold == 0.3
        assert settings.api_port == 8000
        assert settings.rate_limit_rpm == 60
        assert settings.timeout_seconds == 30
        assert settings.log_level == "INFO"
        assert settings.enable_tracing is False

    def test_cors_origins_list(self):
        settings = Settings(
            google_api_key="test",
            jina_api_key="test",
            cors_origins="http://localhost:3000, http://localhost:8501",
        )
        assert settings.cors_origins_list == ["http://localhost:3000", "http://localhost:8501"]

    def test_cors_origins_wildcard(self):
        settings = Settings(google_api_key="test", jina_api_key="test")
        assert settings.cors_origins_list == ["*"]

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            Settings(google_api_key="test", jina_api_key="test", llm_temperature=3.0)

    def test_port_bounds(self):
        with pytest.raises(ValidationError):
            Settings(google_api_key="test", jina_api_key="test", api_port=0)
