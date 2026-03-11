"""Model-agnostic embedding factory for creating embedding models from config."""

from langchain_core.embeddings import Embeddings

from .config import Settings


def create_embeddings(settings: Settings) -> Embeddings:
    """Create an embeddings model based on the configured provider.

    Args:
        settings: Application settings with embedding config.

    Returns:
        A LangChain Embeddings instance.

    Raises:
        ValueError: If the provider is not supported or API key is missing.
    """
    provider = settings.embedding_provider.lower()

    match provider:
        case "jina":
            if not settings.jina_api_key:
                raise ValueError("JINA_API_KEY is required for Jina provider")
            from .embeddings import JinaEmbeddings
            return JinaEmbeddings(
                api_key=settings.jina_api_key,
                model=settings.embedding_model,
            )
        case "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI embedding provider")
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
            )
        case _:
            raise ValueError(
                f"Unsupported embedding provider: '{provider}'. "
                f"Supported: jina, openai"
            )
