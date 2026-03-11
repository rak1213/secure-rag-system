"""
Guardrails Module — Input validation, output validation, and execution limits.

This module adds three layers of protection to the RAG system:
  1. INPUT GUARDRAILS  — Validate user queries BEFORE they reach the LLM
  2. OUTPUT GUARDRAILS — Validate LLM responses BEFORE returning to the user
  3. EXECUTION LIMITS  — Prevent runaway behavior (timeouts, error codes)

Error Taxonomy:
  QUERY_TOO_LONG   — Query exceeds 500 characters
  EMPTY_QUERY      — Query is empty or whitespace only
  OFF_TOPIC        — Query is not about driving/road rules
  PII_DETECTED     — Personal info found (phone, email, license plate)
  RETRIEVAL_EMPTY  — No relevant chunks found above confidence threshold
  LLM_TIMEOUT      — LLM took more than 30 seconds to respond
  POLICY_BLOCK     — General policy violation
"""

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .logging_config import get_logger

log = get_logger(__name__)


# ============================================================================
# ERROR CODES — Structured error taxonomy
# ============================================================================

class ErrorCode:
    """Standardized error codes for the guardrail system."""
    QUERY_TOO_LONG = "QUERY_TOO_LONG"
    EMPTY_QUERY = "EMPTY_QUERY"
    OFF_TOPIC = "OFF_TOPIC"
    PII_DETECTED = "PII_DETECTED"
    RETRIEVAL_EMPTY = "RETRIEVAL_EMPTY"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    POLICY_BLOCK = "POLICY_BLOCK"


@dataclass
class GuardrailResult:
    """Result from a guardrail check."""
    passed: bool
    error_code: str | None = None
    message: str = ""
    sanitized_query: str | None = None
    guardrail_name: str = ""


# ============================================================================
# INPUT GUARDRAIL 1: Query Length Limit
# ============================================================================

def check_query_length(query: str, max_chars: int = 500) -> GuardrailResult:
    """Reject queries that exceed the maximum character limit or are empty."""
    query_len = len(query)
    log.debug("guardrail.query_length.check", query_length=query_len, max_chars=max_chars)

    if not query or not query.strip():
        log.info("guardrail.query_length.blocked", reason="empty_query")
        return GuardrailResult(
            passed=False,
            error_code=ErrorCode.EMPTY_QUERY,
            message="Query is empty. Please enter a question about Nova Scotia driving rules.",
            guardrail_name="EMPTY_QUERY",
        )

    if query_len > max_chars:
        log.info("guardrail.query_length.blocked", reason="too_long", query_length=query_len, max_chars=max_chars)
        return GuardrailResult(
            passed=False,
            error_code=ErrorCode.QUERY_TOO_LONG,
            message=f"Query is too long ({query_len} characters). Maximum allowed is {max_chars} characters.",
            guardrail_name="QUERY_LENGTH_LIMIT",
        )

    log.debug("guardrail.query_length.passed")
    return GuardrailResult(passed=True, guardrail_name="QUERY_LENGTH_LIMIT")


# ============================================================================
# INPUT GUARDRAIL 2: Off-Topic Detection (LLM-based)
# ============================================================================

def check_off_topic(query: str, llm: BaseChatModel) -> GuardrailResult:
    """Use the LLM to classify whether the query is about driving/road rules."""
    log.debug("guardrail.off_topic.check")

    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query classifier. Your ONLY job is to determine if a user's question
is related to driving, traffic rules, road safety, vehicle operation, driver's licenses,
parking, pedestrians, or any transportation/road-related topic.

Respond with EXACTLY one word:
- "YES" if the question is related to driving/traffic/road topics
- "NO" if the question is about something completely unrelated

Do NOT explain your reasoning. Just respond YES or NO."""),
        ("human", "Is this question about driving/traffic/road rules? Question: {query}"),
    ])

    chain = classification_prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({"query": query})
        is_on_topic = result.strip().upper().startswith("YES")

        if is_on_topic:
            log.debug("guardrail.off_topic.passed", llm_response=result.strip())
            return GuardrailResult(passed=True, guardrail_name="OFF_TOPIC_DETECTION")
        else:
            log.info("guardrail.off_topic.blocked", llm_response=result.strip())
            return GuardrailResult(
                passed=False,
                error_code=ErrorCode.OFF_TOPIC,
                message="I can only answer questions about Nova Scotia driving rules. Your question appears to be about a different topic.",
                guardrail_name="OFF_TOPIC_DETECTION",
            )
    except Exception as e:
        log.warning("guardrail.off_topic.error", error=str(e))
        return GuardrailResult(passed=True, guardrail_name="OFF_TOPIC_DETECTION")


# ============================================================================
# INPUT GUARDRAIL 3: PII Detection
# ============================================================================

PII_PATTERNS = {
    "phone_number": re.compile(
        r'\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b'
    ),
    "email": re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ),
    "license_plate": re.compile(
        r'\b[A-Z]{2,3}[-\s]?\d{3,4}\b'
    ),
}


def check_pii(query: str) -> GuardrailResult:
    """Scan the query for personally identifiable information (PII)."""
    log.debug("guardrail.pii.check")

    found_pii = []
    sanitized = query

    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(sanitized)
        if matches:
            found_pii.append(pii_type)
            sanitized = pattern.sub(f"[REDACTED_{pii_type.upper()}]", sanitized)
            match_count = len(matches) if isinstance(matches[0], str) else len(matches)
            log.info("guardrail.pii.detected", pii_type=pii_type, count=match_count)

    if found_pii:
        pii_list = ", ".join(found_pii)
        log.info("guardrail.pii.sanitized", pii_types=pii_list)
        return GuardrailResult(
            passed=True,
            error_code=ErrorCode.PII_DETECTED,
            message=f"WARNING: Personal information detected ({pii_list}). It has been removed from your query for privacy. Processing sanitized query.",
            sanitized_query=sanitized,
            guardrail_name="PII_DETECTION",
        )

    log.debug("guardrail.pii.passed")
    return GuardrailResult(passed=True, guardrail_name="PII_DETECTION")


# ============================================================================
# OUTPUT GUARDRAIL 1: Retrieval Confidence Check
# ============================================================================

def check_retrieval_confidence(
    docs_with_scores: list[tuple],
    threshold: float = 0.3,
) -> GuardrailResult:
    """Check if retrieved chunks are relevant enough to generate an answer."""
    if not docs_with_scores:
        log.info("guardrail.retrieval_confidence.blocked", reason="no_chunks")
        return GuardrailResult(
            passed=False,
            error_code=ErrorCode.RETRIEVAL_EMPTY,
            message="I don't have enough information to answer that question. No relevant content was found in the documents.",
            guardrail_name="RETRIEVAL_CONFIDENCE",
        )

    scores = [score for _, score in docs_with_scores]
    best_score = max(scores)
    avg_score = sum(scores) / len(scores)

    log.debug("guardrail.retrieval_confidence.check",
              best_score=round(best_score, 4),
              avg_score=round(avg_score, 4),
              threshold=threshold,
              num_chunks=len(scores))

    if best_score < threshold:
        log.info("guardrail.retrieval_confidence.blocked",
                 best_score=round(best_score, 4), threshold=threshold)
        return GuardrailResult(
            passed=False,
            error_code=ErrorCode.RETRIEVAL_EMPTY,
            message=f"I don't have enough information to answer that question. The most relevant content scored {best_score:.4f}, which is below the confidence threshold of {threshold}.",
            guardrail_name="RETRIEVAL_CONFIDENCE",
        )

    log.debug("guardrail.retrieval_confidence.passed", best_score=round(best_score, 4))
    return GuardrailResult(passed=True, guardrail_name="RETRIEVAL_CONFIDENCE")


# ============================================================================
# OUTPUT GUARDRAIL 2: Response Length Limit
# ============================================================================

def check_response_length(response: str, max_words: int = 500) -> tuple[str, bool]:
    """Cap the response to a maximum word count."""
    words = response.split()
    word_count = len(words)

    log.debug("guardrail.response_length.check", word_count=word_count, max_words=max_words)

    if word_count > max_words:
        truncated = " ".join(words[:max_words]) + f"\n\n[Response truncated — exceeded maximum length of {max_words} words]"
        log.info("guardrail.response_length.truncated", word_count=word_count, max_words=max_words)
        return truncated, True

    return response, False


# ============================================================================
# EXECUTION LIMIT: Timeout Handling (cross-platform)
# ============================================================================

class TimeoutError(Exception):
    """Raised when an operation exceeds the time limit."""
    pass


def run_with_timeout(func, timeout_seconds: int = 30, *args, **kwargs):
    """Run a function with a timeout limit using ThreadPoolExecutor.

    Cross-platform and thread-safe — works on Linux, macOS, Windows, and in
    multi-threaded environments like FastAPI/uvicorn.

    Args:
        func: The function to execute.
        timeout_seconds: Maximum seconds to wait (default 30).
        *args, **kwargs: Arguments to pass to the function.

    Returns:
        The function's return value.

    Raises:
        TimeoutError: If the function exceeds the time limit.
    """
    log.debug("execution_limit.timeout.set", timeout_seconds=timeout_seconds)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            log.debug("execution_limit.timeout.ok")
            return result
        except FuturesTimeoutError:
            log.warning("execution_limit.timeout.exceeded", timeout_seconds=timeout_seconds)
            raise TimeoutError(f"Operation timed out after {timeout_seconds}s")


# ============================================================================
# COMBINED INPUT VALIDATION PIPELINE
# ============================================================================

def run_input_guardrails(
    query: str,
    llm: BaseChatModel,
    max_query_length: int = 500,
) -> tuple[bool, str, list[str], list[str], list[str]]:
    """Run ALL input guardrails in sequence on a user query.

    Returns:
        Tuple of (can_proceed, processed_query, triggered_guardrails, error_codes, messages).
    """
    log.info("guardrails.input.pipeline_start", query_preview=query[:80])

    triggered = []
    error_codes = []
    messages = []
    processed_query = query

    # --- Guardrail 1: Empty / Length Check ---
    length_result = check_query_length(query, max_query_length)
    if not length_result.passed:
        triggered.append(length_result.guardrail_name)
        error_codes.append(length_result.error_code)
        messages.append(length_result.message)
        log.info("guardrails.input.blocked", guardrail=length_result.guardrail_name)
        return False, query, triggered, error_codes, messages

    # --- Guardrail 2: PII Detection ---
    pii_result = check_pii(query)
    if pii_result.error_code == ErrorCode.PII_DETECTED:
        triggered.append(pii_result.guardrail_name)
        error_codes.append(pii_result.error_code)
        messages.append(pii_result.message)
        processed_query = pii_result.sanitized_query

    # --- Guardrail 3: Off-Topic Detection ---
    topic_result = check_off_topic(processed_query or query, llm)
    if not topic_result.passed:
        triggered.append(topic_result.guardrail_name)
        error_codes.append(topic_result.error_code)
        messages.append(topic_result.message)
        log.info("guardrails.input.blocked", guardrail=topic_result.guardrail_name)
        return False, processed_query or query, triggered, error_codes, messages

    status = "warnings" if triggered else "passed"
    log.info("guardrails.input.pipeline_done", status=status, triggered=triggered)
    return True, processed_query or query, triggered, error_codes, messages
