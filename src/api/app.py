"""FastAPI application with lifespan management."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ..config import get_settings
from ..logging_config import setup_logging, get_logger
from ..llm_factory import create_llm
from ..embedding_factory import create_embeddings
from ..vector_store import get_vector_store, index_documents, is_indexed
from ..document_loader import load_pdf_documents
from ..text_splitter import split_documents
from ..secure_rag_chain import SecureRAGChain
from ..logger import SecurityLogger
from .dependencies import set_rag_chain, set_security_logger, set_settings
from .routes import router
from .middleware import setup_middleware

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG system on startup."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level, log_format=settings.log_format)

    log.info("app.startup", llm_provider=settings.llm_provider, embedding_provider=settings.embedding_provider)

    # Optionally set up tracing
    if settings.enable_tracing:
        try:
            from ..observability import setup_tracing
            setup_tracing(settings)
            log.info("app.tracing.enabled")
        except ImportError:
            log.warning("app.tracing.unavailable", detail="Install arize-phoenix for tracing")

    # Initialize components
    llm = create_llm(settings)
    embeddings = create_embeddings(settings)
    vector_store = get_vector_store(embeddings)

    # Index documents if needed
    if not is_indexed(vector_store):
        try:
            documents = load_pdf_documents("data")
            chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
            index_documents(vector_store, chunks, embeddings)
        except (FileNotFoundError, ValueError) as e:
            log.warning("app.startup.no_documents", detail=str(e))

    # Create secure chain
    security_logger = SecurityLogger()
    secure_chain = SecureRAGChain(
        vector_store=vector_store,
        embeddings=embeddings,
        llm=llm,
        num_chunks=settings.num_retrieval_chunks,
        confidence_threshold=settings.confidence_threshold,
        timeout_seconds=settings.timeout_seconds,
        max_response_words=settings.max_response_words,
        logger=security_logger,
    )

    # Set global instances for dependency injection
    set_rag_chain(secure_chain)
    set_security_logger(security_logger)
    set_settings(settings)

    log.info("app.startup.complete")
    yield
    log.info("app.shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Secure RAG System API",
        description="Production-ready RAG system with guardrails, prompt injection defense, and evaluation",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Rate limiting
    limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.rate_limit_rpm}/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    setup_middleware(app)

    # Routes
    app.include_router(router)

    return app
