"""Pydantic request/response models for the API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    question: str = Field(..., min_length=1, max_length=500, description="Question to ask the RAG system")


class SourceInfo(BaseModel):
    """Source citation information."""
    citation: str


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""
    answer: str
    sources: list[str] = []
    was_blocked: bool = False
    guardrails_triggered: list[str] = []
    defenses_triggered: list[str] = []
    error_codes: list[str] = []
    faithfulness_score: float = -1.0
    retrieval_scores: list[float] = []
    messages: list[str] = []


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str
    vector_store_ready: bool
    document_count: int
    llm_provider: str
    embedding_provider: str


class MetricsResponse(BaseModel):
    """Response body for the /metrics endpoint."""
    session_start: str
    total_queries: int
    queries_processed: int
    queries_blocked: int
    guardrails_triggered: dict[str, int]
    defenses_triggered: dict[str, int]
    avg_faithfulness: float | None
    avg_retrieval_score: float | None


class UploadResponse(BaseModel):
    """Response body for the /documents/upload endpoint."""
    message: str
    pages_loaded: int
    chunks_created: int
    chunks_indexed: int


class SwitchModelRequest(BaseModel):
    """Request body for switching the LLM model."""
    provider: str = Field(..., description="LLM provider: gemini, openai, or anthropic")
    model: str = Field(..., description="Model name (e.g. gpt-4o, gemini-2.5-flash)")
    api_key: str | None = Field(default=None, description="Optional API key. Falls back to env var if not provided.")


class SwitchModelResponse(BaseModel):
    """Response body for the /switch-model endpoint."""
    message: str
    provider: str
    model: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: str | None = None
