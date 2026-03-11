"""Centralized configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # --- LLM Configuration ---
    llm_provider: str = Field(default="gemini", description="LLM provider: gemini, openai, or anthropic")
    llm_model: str = Field(default="gemini-2.5-flash", description="Model name for the LLM")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # --- Embedding Configuration ---
    embedding_provider: str = Field(default="jina", description="Embedding provider: jina or openai")
    embedding_model: str = Field(default="jina-embeddings-v3", description="Model name for embeddings")

    # --- API Keys ---
    google_api_key: str = Field(default="", description="Google Gemini API key")
    jina_api_key: str = Field(default="", description="Jina AI API key")
    openai_api_key: str = Field(default="", description="OpenAI API key (optional)")
    anthropic_api_key: str = Field(default="", description="Anthropic API key (optional)")

    # --- RAG Tuning ---
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=2000)
    num_retrieval_chunks: int = Field(default=4, ge=1, le=20)
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- Server ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_key: str = Field(default="", description="Optional API key for authentication")
    rate_limit_rpm: int = Field(default=60, ge=1)
    cors_origins: str = Field(default="*", description="Comma-separated allowed origins")

    # --- Observability ---
    enable_tracing: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="console", description="Log format: json or console")

    # --- Execution Limits ---
    timeout_seconds: int = Field(default=30, ge=5, le=300)
    max_response_words: int = Field(default=500, ge=50, le=5000)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


def get_settings() -> Settings:
    """Create and return application settings."""
    return Settings()
