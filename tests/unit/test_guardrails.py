"""Tests for guardrails module — each guardrail function in isolation."""

import pytest

from src.guardrails import (
    check_query_length,
    check_pii,
    check_retrieval_confidence,
    check_response_length,
    run_with_timeout,
    ErrorCode,
    TimeoutError,
)


class TestCheckQueryLength:
    def test_normal_query_passes(self):
        result = check_query_length("What are the speed limits?")
        assert result.passed is True

    def test_empty_query_blocked(self):
        result = check_query_length("")
        assert result.passed is False
        assert result.error_code == ErrorCode.EMPTY_QUERY

    def test_whitespace_query_blocked(self):
        result = check_query_length("   ")
        assert result.passed is False
        assert result.error_code == ErrorCode.EMPTY_QUERY

    def test_too_long_query_blocked(self):
        result = check_query_length("x" * 501)
        assert result.passed is False
        assert result.error_code == ErrorCode.QUERY_TOO_LONG

    def test_exactly_at_limit_passes(self):
        result = check_query_length("x" * 500)
        assert result.passed is True

    def test_custom_limit(self):
        result = check_query_length("x" * 101, max_chars=100)
        assert result.passed is False


class TestCheckPII:
    def test_no_pii_passes(self):
        result = check_pii("What are the speed limits on highways?")
        assert result.passed is True
        assert result.sanitized_query is None

    def test_phone_number_detected(self):
        result = check_pii("My number is 902-555-0199, can I park here?")
        assert result.passed is True  # PII doesn't block, just sanitizes
        assert result.error_code == ErrorCode.PII_DETECTED
        assert "REDACTED" in result.sanitized_query

    def test_email_detected(self):
        result = check_pii("Contact me at driver@example.com about parking")
        assert result.error_code == ErrorCode.PII_DETECTED
        assert "REDACTED" in result.sanitized_query

    def test_license_plate_detected(self):
        result = check_pii("My plate is ABC 1234, where can I park?")
        assert result.error_code == ErrorCode.PII_DETECTED


class TestCheckRetrievalConfidence:
    def test_high_scores_pass(self, docs_with_scores):
        result = check_retrieval_confidence(docs_with_scores, threshold=0.3)
        assert result.passed is True

    def test_low_scores_blocked(self, sample_documents):
        low_score_docs = [(sample_documents[0], 0.1), (sample_documents[1], 0.05)]
        result = check_retrieval_confidence(low_score_docs, threshold=0.3)
        assert result.passed is False
        assert result.error_code == ErrorCode.RETRIEVAL_EMPTY

    def test_empty_docs_blocked(self):
        result = check_retrieval_confidence([], threshold=0.3)
        assert result.passed is False

    def test_custom_threshold(self, docs_with_scores):
        result = check_retrieval_confidence(docs_with_scores, threshold=0.9)
        assert result.passed is False


class TestCheckResponseLength:
    def test_short_response_passes(self):
        response, truncated = check_response_length("Short answer.", max_words=500)
        assert truncated is False
        assert response == "Short answer."

    def test_long_response_truncated(self):
        long_response = " ".join(["word"] * 600)
        response, truncated = check_response_length(long_response, max_words=500)
        assert truncated is True
        assert "truncated" in response.lower()


class TestRunWithTimeout:
    def test_fast_function_succeeds(self):
        result = run_with_timeout(lambda: 42, timeout_seconds=5)
        assert result == 42

    def test_slow_function_times_out(self):
        import time

        def slow():
            time.sleep(10)
            return "done"

        with pytest.raises(TimeoutError):
            run_with_timeout(slow, timeout_seconds=1)
