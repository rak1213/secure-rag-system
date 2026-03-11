"""Dependency injection for the FastAPI application."""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from ..config import Settings, get_settings
from ..secure_rag_chain import SecureRAGChain
from ..logger import SecurityLogger

# Global instances (initialized during app lifespan)
_secure_chain: SecureRAGChain | None = None
_security_logger: SecurityLogger | None = None
_settings: Settings | None = None

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def set_rag_chain(chain: SecureRAGChain) -> None:
    """Set the global RAG chain instance (called during startup)."""
    global _secure_chain
    _secure_chain = chain


def set_security_logger(logger: SecurityLogger) -> None:
    """Set the global security logger instance."""
    global _security_logger
    _security_logger = logger


def set_settings(settings: Settings) -> None:
    """Set the global settings instance."""
    global _settings
    _settings = settings


def get_rag_chain() -> SecureRAGChain:
    """Get the RAG chain instance."""
    if _secure_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system is not initialized yet",
        )
    return _secure_chain


def get_security_logger() -> SecurityLogger:
    """Get the security logger instance."""
    if _security_logger is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Security logger is not initialized",
        )
    return _security_logger


def get_app_settings() -> Settings:
    """Get the application settings."""
    if _settings is None:
        return get_settings()
    return _settings


def verify_api_key(
    api_key: str | None = Security(api_key_header),
    settings: Settings = Depends(get_app_settings),
) -> str | None:
    """Verify the API key if one is configured.

    If API_KEY is not set in config, authentication is disabled.
    """
    if not settings.api_key:
        return None  # No auth configured

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key
