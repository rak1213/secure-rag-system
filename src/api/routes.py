"""API endpoints for the RAG system."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status

from .schemas import (
    QueryRequest, QueryResponse, HealthResponse,
    MetricsResponse, UploadResponse, ErrorResponse,
    SwitchModelRequest, SwitchModelResponse,
)
from .dependencies import (
    get_rag_chain, get_security_logger, get_app_settings,
    verify_api_key,
)
from ..secure_rag_chain import SecureRAGChain
from ..logger import SecurityLogger
from ..config import Settings
from ..document_loader import load_pdf_documents
from ..text_splitter import split_documents
from ..vector_store import index_documents
from ..logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1")


# ============================================================================
# POST /api/v1/query
# ============================================================================

@router.post(
    "/query",
    response_model=QueryResponse,
    responses={429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def query(
    request: QueryRequest,
    chain: SecureRAGChain = Depends(get_rag_chain),
    _api_key: str | None = Depends(verify_api_key),
):
    """Process a question through the secure RAG pipeline."""
    log.info("api.query", question=request.question[:80])

    response = chain.query(request.question)

    return QueryResponse(
        answer=response.answer,
        sources=response.sources,
        was_blocked=response.was_blocked,
        guardrails_triggered=response.guardrails_triggered,
        defenses_triggered=response.defenses_triggered,
        error_codes=response.error_codes,
        faithfulness_score=response.faithfulness_score,
        retrieval_scores=response.retrieval_scores,
        messages=response.messages,
    )


# ============================================================================
# POST /api/v1/documents/upload
# ============================================================================

PDF_MAGIC_BYTES = b"%PDF"
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}},
)
async def upload_document(
    file: UploadFile = File(...),
    chain: SecureRAGChain = Depends(get_rag_chain),
    _api_key: str | None = Depends(verify_api_key),
):
    """Upload a PDF document for indexing into the RAG system."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    content = await file.read()

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)} MB.",
        )

    if not content[:4].startswith(PDF_MAGIC_BYTES):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file. File does not have valid PDF magic bytes.",
        )

    # Save to a temp directory and process
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / file.filename
        tmp_path.write_bytes(content)

        documents = load_pdf_documents(tmp_dir)
        chunks = split_documents(documents)
        num_indexed = index_documents(
            chain.vector_store, chunks, chain.embeddings, force_reindex=False
        )

    log.info("api.upload.done",
             filename=file.filename,
             pages=len(documents),
             chunks=len(chunks),
             indexed=num_indexed)

    return UploadResponse(
        message=f"Successfully processed {file.filename}",
        pages_loaded=len(documents),
        chunks_created=len(chunks),
        chunks_indexed=num_indexed,
    )


# ============================================================================
# GET /api/v1/health
# ============================================================================

@router.get("/health", response_model=HealthResponse)
async def health(
    chain: SecureRAGChain = Depends(get_rag_chain),
    settings: Settings = Depends(get_app_settings),
):
    """Health and readiness check."""
    doc_count = chain.vector_store._collection.count()
    return HealthResponse(
        status="healthy",
        vector_store_ready=doc_count > 0,
        document_count=doc_count,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
    )


# ============================================================================
# GET /api/v1/metrics
# ============================================================================

@router.get("/metrics", response_model=MetricsResponse)
async def metrics(
    logger: SecurityLogger = Depends(get_security_logger),
    _api_key: str | None = Depends(verify_api_key),
):
    """Query stats, latency, and guardrail trigger counts."""
    data = logger.get_metrics()
    return MetricsResponse(**data)


# ============================================================================
# POST /api/v1/switch-model
# ============================================================================

@router.post(
    "/switch-model",
    response_model=SwitchModelResponse,
    responses={400: {"model": ErrorResponse}},
)
async def switch_model(
    request: SwitchModelRequest,
    chain: SecureRAGChain = Depends(get_rag_chain),
    settings: Settings = Depends(get_app_settings),
    _api_key: str | None = Depends(verify_api_key),
):
    """Switch the LLM model at runtime without restarting the server."""
    from ..llm_factory import create_llm
    from langchain_core.output_parsers import StrOutputParser

    provider = request.provider.lower()
    model = request.model

    # Build a temporary settings override to create the new LLM
    overrides = {"llm_provider": provider, "llm_model": model}

    # Use the provided API key, or fall back to the env key
    if request.api_key:
        match provider:
            case "gemini":
                overrides["google_api_key"] = request.api_key
            case "openai":
                overrides["openai_api_key"] = request.api_key
            case "anthropic":
                overrides["anthropic_api_key"] = request.api_key
            case _:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported provider: {provider}. Use gemini, openai, or anthropic.",
                )
    else:
        # Fall back to existing env keys
        match provider:
            case "gemini":
                overrides["google_api_key"] = settings.google_api_key
            case "openai":
                overrides["openai_api_key"] = settings.openai_api_key
            case "anthropic":
                overrides["anthropic_api_key"] = settings.anthropic_api_key

    try:
        temp_settings = Settings(**{**settings.model_dump(), **overrides})
        new_llm = create_llm(temp_settings)
    except (ValueError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create LLM: {e}",
        )

    # Hot-swap the LLM on the running chain
    chain.llm = new_llm
    chain.chain = chain.prompt | new_llm | StrOutputParser()

    log.info("api.switch_model", provider=provider, model=model)

    return SwitchModelResponse(
        message=f"Switched to {provider}/{model}",
        provider=provider,
        model=model,
    )
