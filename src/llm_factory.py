"""Model-agnostic LLM factory for creating chat models from config."""

from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings


def create_llm(settings: Settings) -> BaseChatModel:
    """Create a chat model based on the configured provider.

    Args:
        settings: Application settings with LLM config.

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If the provider is not supported or API key is missing.
    """
    provider = settings.llm_provider.lower()

    match provider:
        case "gemini":
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY is required for Gemini provider")
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.llm_model,
                google_api_key=settings.google_api_key,
                temperature=settings.llm_temperature,
            )
        case "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.openai_api_key,
                temperature=settings.llm_temperature,
            )
        case "anthropic":
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=settings.llm_model,
                api_key=settings.anthropic_api_key,
                temperature=settings.llm_temperature,
            )
        case _:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                f"Supported: gemini, openai, anthropic"
            )
