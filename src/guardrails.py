"""
Guardrails Module — Input validation, output validation, and execution limits.

This module adds three layers of protection to the RAG system:
  1. INPUT GUARDRAILS  — Validate user queries BEFORE they reach the LLM
  2. OUTPUT GUARDRAILS — Validate LLM responses BEFORE returning to the user
  3. EXECUTION LIMITS  — Prevent runaway behavior (timeouts, error codes)

Each guardrail logs when it triggers so you can trace exactly what happened.

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
import signal
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
    """
    Result from a guardrail check.

    Attributes:
        passed: True if the check passed (no violation), False if blocked.
        error_code: The error code if blocked, or None if passed.
        message: Human-readable explanation of what happened.
        sanitized_query: The cleaned-up query (if PII was stripped), or None.
    """
    passed: bool
    error_code: str | None = None
    message: str = ""
    sanitized_query: str | None = None
    guardrail_name: str = ""


# ============================================================================
# INPUT GUARDRAIL 1: Query Length Limit
# ============================================================================

def check_query_length(query: str, max_chars: int = 500) -> GuardrailResult:
    """
    Reject queries that exceed the maximum character limit.

    WHY: Extremely long queries can be used for prompt injection attacks
    (hiding malicious instructions in a wall of text) or can cause
    excessive token usage and slow responses.

    Args:
        query: The user's raw query string.
        max_chars: Maximum allowed characters (default 500).

    Returns:
        GuardrailResult — passed=True if query is within limit.
    """
    print(f"    [Guardrail] Checking query length: {len(query)} chars (max: {max_chars})")

    if not query or not query.strip():
        print(f"    [Guardrail] BLOCKED — Empty query detected")
        return GuardrailResult(
            passed=False,
            error_code=ErrorCode.EMPTY_QUERY,
            message="Query is empty. Please enter a question about Nova Scotia driving rules.",
            guardrail_name="EMPTY_QUERY",
        )

    if len(query) > max_chars:
        print(f"    [Guardrail] BLOCKED — Query too long ({len(query)} > {max_chars})")
        return GuardrailResult(
            passed=False,
            error_code=ErrorCode.QUERY_TOO_LONG,
            message=f"Query is too long ({len(query)} characters). Maximum allowed is {max_chars} characters.",
            guardrail_name="QUERY_LENGTH_LIMIT",
        )

    print(f"    [Guardrail] PASSED — Query length OK")
    return GuardrailResult(passed=True, guardrail_name="QUERY_LENGTH_LIMIT")


# ============================================================================
# INPUT GUARDRAIL 2: Off-Topic Detection (LLM-based)
# ============================================================================

def check_off_topic(query: str, llm: ChatGoogleGenerativeAI) -> GuardrailResult:
    """
    Use the LLM to classify whether the query is about driving/road rules.

    WHY: Our RAG system is specifically for Nova Scotia driving rules.
    Answering off-topic questions (recipes, travel, etc.) wastes resources
    and could produce hallucinated answers since the context won't be relevant.

    HOW: We send a classification prompt to Gemini asking it to respond
    with just "YES" or "NO" — is this query about driving/traffic/road rules?

    Args:
        query: The user's query string.
        llm: The Google Gemini LLM instance.

    Returns:
        GuardrailResult — passed=True if query is on-topic.
    """
    print(f"    [Guardrail] Checking if query is on-topic (driving/road rules)...")

    # Build a simple classification prompt
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
            print(f"    [Guardrail] PASSED — Query is on-topic (LLM said: {result.strip()})")
            return GuardrailResult(passed=True, guardrail_name="OFF_TOPIC_DETECTION")
        else:
            print(f"    [Guardrail] BLOCKED — Query is off-topic (LLM said: {result.strip()})")
            return GuardrailResult(
                passed=False,
                error_code=ErrorCode.OFF_TOPIC,
                message="I can only answer questions about Nova Scotia driving rules. Your question appears to be about a different topic.",
                guardrail_name="OFF_TOPIC_DETECTION",
            )
    except Exception as e:
        # If classification fails, let the query through (fail-open for classification)
        print(f"    [Guardrail] WARNING — Off-topic check failed ({e}), allowing query through")
        return GuardrailResult(passed=True, guardrail_name="OFF_TOPIC_DETECTION")


# ============================================================================
# INPUT GUARDRAIL 3: PII Detection
# ============================================================================

# Regex patterns for common PII types
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
    """
    Scan the query for personally identifiable information (PII).

    WHY: Users might accidentally include personal info like phone numbers,
    emails, or license plates in their queries. We should:
    1. Warn the user that PII was detected
    2. Strip the PII before processing (replace with [REDACTED])
    3. Still process the sanitized query if possible

    HOW: We use regex patterns to detect common PII formats:
    - Phone numbers: (902) 555-0199, 902-555-0199, +1-902-555-0199
    - Email addresses: user@example.com
    - License plates: ABC 1234, AB-1234

    Args:
        query: The user's raw query string.

    Returns:
        GuardrailResult — If PII found, passed=True but sanitized_query is set
                          and a warning message is included.
    """
    print(f"    [Guardrail] Scanning for PII (phone numbers, emails, license plates)...")

    found_pii = []
    sanitized = query

    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(sanitized)
        if matches:
            found_pii.append(pii_type)
            # Replace each match with [REDACTED]
            sanitized = pattern.sub(f"[REDACTED_{pii_type.upper()}]", sanitized)
            print(f"    [Guardrail] DETECTED — {pii_type}: {len(matches) if isinstance(matches[0], str) else len(matches)} instance(s) found and redacted")

    if found_pii:
        pii_list = ", ".join(found_pii)
        print(f"    [Guardrail] WARNING — PII detected: {pii_list}. Query sanitized.")
        return GuardrailResult(
            passed=True,  # We still process the query, but sanitized
            error_code=ErrorCode.PII_DETECTED,
            message=f"WARNING: Personal information detected ({pii_list}). It has been removed from your query for privacy. Processing sanitized query.",
            sanitized_query=sanitized,
            guardrail_name="PII_DETECTION",
        )

    print(f"    [Guardrail] PASSED — No PII detected")
    return GuardrailResult(passed=True, guardrail_name="PII_DETECTION")


# ============================================================================
# OUTPUT GUARDRAIL 1: Retrieval Confidence Check
# ============================================================================

def check_retrieval_confidence(
    docs_with_scores: list[tuple],
    threshold: float = 0.3,
) -> GuardrailResult:
    """
    Check if retrieved chunks are relevant enough to generate an answer.

    WHY: If the similarity scores are too low, the retrieved chunks are
    probably not relevant to the question. Generating an answer from
    irrelevant context leads to hallucination. Better to say "I don't know"
    than to make something up.

    HOW: We check the TOP similarity score. If even the best match is below
    our threshold, we refuse to answer.

    Note: ChromaDB returns DISTANCE scores (lower = more similar).
    LangChain's similarity_search_with_relevance_scores() converts these
    to RELEVANCE scores (higher = more similar, range 0-1).

    Args:
        docs_with_scores: List of (Document, relevance_score) tuples.
        threshold: Minimum relevance score to consider (default 0.3).

    Returns:
        GuardrailResult — passed=True if at least one chunk is above threshold.
    """
    if not docs_with_scores:
        print(f"    [Guardrail] BLOCKED — No chunks retrieved at all")
        return GuardrailResult(
            passed=False,
            error_code=ErrorCode.RETRIEVAL_EMPTY,
            message="I don't have enough information to answer that question. No relevant content was found in the documents.",
            guardrail_name="RETRIEVAL_CONFIDENCE",
        )

    # Get the best (highest) relevance score
    scores = [score for _, score in docs_with_scores]
    best_score = max(scores)
    avg_score = sum(scores) / len(scores)

    print(f"    [Guardrail] Checking retrieval confidence:")
    print(f"      Top score: {best_score:.4f} | Avg score: {avg_score:.4f} | Threshold: {threshold}")
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        preview = doc.page_content[:50].replace("\n", " ")
        status = "PASS" if score >= threshold else "FAIL"
        print(f"      Chunk {i}: score={score:.4f} [{status}] \"{preview}...\"")

    if best_score < threshold:
        print(f"    [Guardrail] BLOCKED — Best score {best_score:.4f} is below threshold {threshold}")
        return GuardrailResult(
            passed=False,
            error_code=ErrorCode.RETRIEVAL_EMPTY,
            message=f"I don't have enough information to answer that question. The most relevant content scored {best_score:.4f}, which is below the confidence threshold of {threshold}.",
            guardrail_name="RETRIEVAL_CONFIDENCE",
        )

    print(f"    [Guardrail] PASSED — Retrieval confidence OK (best: {best_score:.4f})")
    return GuardrailResult(passed=True, guardrail_name="RETRIEVAL_CONFIDENCE")


# ============================================================================
# OUTPUT GUARDRAIL 2: Response Length Limit
# ============================================================================

def check_response_length(response: str, max_words: int = 500) -> tuple[str, bool]:
    """
    Cap the response to a maximum word count.

    WHY: Extremely long responses can be a sign of the LLM going off-track,
    repeating itself, or being manipulated by a prompt injection attack.
    A reasonable cap keeps responses focused and useful.

    Args:
        response: The LLM's generated response.
        max_words: Maximum number of words allowed (default 500).

    Returns:
        Tuple of (possibly truncated response, was_truncated boolean).
    """
    words = response.split()
    word_count = len(words)

    print(f"    [Guardrail] Checking response length: {word_count} words (max: {max_words})")

    if word_count > max_words:
        truncated = " ".join(words[:max_words]) + "\n\n[Response truncated — exceeded maximum length of {max_words} words]"
        print(f"    [Guardrail] TRUNCATED — Response was {word_count} words, capped to {max_words}")
        return truncated, True

    print(f"    [Guardrail] PASSED — Response length OK")
    return response, False


# ============================================================================
# EXECUTION LIMIT: Timeout Handling
# ============================================================================

class TimeoutError(Exception):
    """Raised when an operation exceeds the time limit."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")


def run_with_timeout(func, timeout_seconds: int = 30, *args, **kwargs):
    """
    Run a function with a timeout limit.

    WHY: If the LLM API hangs or takes too long, we don't want the user
    waiting forever. A 30-second timeout gives a clear error instead.

    HOW: Uses Python's signal.alarm() on Unix/macOS to set a timer.
    If the function doesn't return in time, we raise TimeoutError.

    Args:
        func: The function to execute.
        timeout_seconds: Maximum seconds to wait (default 30).
        *args, **kwargs: Arguments to pass to the function.

    Returns:
        The function's return value.

    Raises:
        TimeoutError: If the function exceeds the time limit.
    """
    print(f"    [Execution Limit] Setting {timeout_seconds}s timeout for LLM call...")

    # Store the old handler so we can restore it
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    try:
        signal.alarm(timeout_seconds)  # Start the timer
        result = func(*args, **kwargs)
        signal.alarm(0)  # Cancel the timer (function finished in time)
        print(f"    [Execution Limit] LLM responded within timeout")
        return result
    except TimeoutError:
        print(f"    [Execution Limit] TIMEOUT — LLM did not respond in {timeout_seconds}s")
        raise
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # Restore old handler


# ============================================================================
# COMBINED INPUT VALIDATION PIPELINE
# ============================================================================

def run_input_guardrails(
    query: str,
    llm: ChatGoogleGenerativeAI,
    max_query_length: int = 500,
) -> tuple[bool, str, list[str], list[str], list[str]]:
    """
    Run ALL input guardrails in sequence on a user query.

    Pipeline order:
      1. Check if query is empty
      2. Check query length
      3. Check for PII (and sanitize if found)
      4. Check if query is off-topic

    Args:
        query: The raw user query.
        llm: Google Gemini LLM for off-topic classification.
        max_query_length: Maximum query length in characters.

    Returns:
        Tuple of:
          - can_proceed (bool): Whether the query should be processed
          - processed_query (str): The query to use (possibly sanitized)
          - triggered_guardrails (list): Names of guardrails that triggered
          - error_codes (list): Error codes for any violations
          - messages (list): Messages to show the user
    """
    # Type note: returns 5-element tuple
    print(f"\n  === INPUT GUARDRAILS PIPELINE ===")
    print(f"  Processing query: \"{query[:80]}{'...' if len(query) > 80 else ''}\"")

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
        print(f"  === INPUT GUARDRAILS: BLOCKED by {length_result.guardrail_name} ===\n")
        return False, query, triggered, error_codes, messages

    # --- Guardrail 2: PII Detection ---
    pii_result = check_pii(query)
    if pii_result.error_code == ErrorCode.PII_DETECTED:
        triggered.append(pii_result.guardrail_name)
        error_codes.append(pii_result.error_code)
        messages.append(pii_result.message)
        processed_query = pii_result.sanitized_query  # Use the sanitized version
        # Note: We DON'T block — we just sanitize and warn

    # --- Guardrail 3: Off-Topic Detection ---
    topic_result = check_off_topic(processed_query or query, llm)
    if not topic_result.passed:
        triggered.append(topic_result.guardrail_name)
        error_codes.append(topic_result.error_code)
        messages.append(topic_result.message)
        print(f"  === INPUT GUARDRAILS: BLOCKED by {topic_result.guardrail_name} ===\n")
        return False, processed_query or query, triggered, error_codes, messages

    status = "WARNINGS" if triggered else "ALL PASSED"
    print(f"  === INPUT GUARDRAILS: {status} ===\n")
    return True, processed_query or query, triggered, error_codes, messages
