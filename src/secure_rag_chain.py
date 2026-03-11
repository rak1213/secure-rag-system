"""
Secure RAG Chain — Enhanced RAG pipeline with all guardrails and defenses.

  Layer 1: INPUT GUARDRAILS   — Validate queries before processing
  Layer 2: PROMPT DEFENSES    — Protect against prompt injection attacks
  Layer 3: OUTPUT GUARDRAILS  — Validate responses before returning
"""

from dataclasses import dataclass, field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from .retriever import format_docs_with_sources
from .guardrails import (
    run_input_guardrails,
    check_retrieval_confidence,
    check_response_length,
    run_with_timeout,
    ErrorCode,
    TimeoutError,
)
from .prompt_defense import (
    get_hardened_system_prompt,
    run_prompt_defenses,
    wrap_context_with_delimiters,
    validate_output,
)
from .evaluation import (
    check_faithfulness,
    evaluate_retrieval_relevance,
)
from .logger import SecurityLogger
from .logging_config import get_logger

log = get_logger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SecureRAGResponse:
    """Response from the secure RAG system with security metadata."""
    answer: str
    sources: list[str] = field(default_factory=list)
    source_documents: list[Document] = field(default_factory=list)
    guardrails_triggered: list[str] = field(default_factory=list)
    defenses_triggered: list[str] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)
    was_blocked: bool = False
    faithfulness_score: float = -1.0
    retrieval_scores: list[float] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def format_full_response(self) -> str:
        """Format the complete response with all metadata."""
        parts = []

        if self.messages:
            for msg in self.messages:
                parts.append(f"[!] {msg}")
            parts.append("")

        parts.append(self.answer)

        if self.sources:
            parts.append("\n--- Sources ---")
            for source in self.sources:
                parts.append(f"  {source}")

        return "\n".join(parts)


# ============================================================================
# SECURE RAG CHAIN
# ============================================================================

class SecureRAGChain:
    """Production-ready RAG chain with guardrails, prompt injection defense,
    and evaluation built in."""

    def __init__(
        self,
        vector_store: Chroma,
        embeddings: Embeddings,
        llm: BaseChatModel,
        num_chunks: int = 4,
        confidence_threshold: float = 0.3,
        timeout_seconds: int = 30,
        max_response_words: int = 500,
        logger: SecurityLogger | None = None,
    ):
        """Initialize the Secure RAG chain.

        Args:
            vector_store: ChromaDB vector store with indexed documents.
            embeddings: Embeddings model.
            llm: Language model instance (any BaseChatModel).
            num_chunks: Number of chunks to retrieve per query.
            confidence_threshold: Minimum relevance score for retrieval.
            timeout_seconds: Max seconds to wait for LLM response.
            max_response_words: Maximum words in the response.
            logger: SecurityLogger instance for tracking events.
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.num_chunks = num_chunks
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds
        self.max_response_words = max_response_words
        self.logger = logger or SecurityLogger()
        self.llm = llm

        hardened_prompt = get_hardened_system_prompt()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", hardened_prompt),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

        log.info("secure_rag.init",
                 num_chunks=num_chunks,
                 confidence_threshold=confidence_threshold,
                 timeout_seconds=timeout_seconds,
                 max_response_words=max_response_words)

    def query(self, question: str) -> SecureRAGResponse:
        """Process a question through the SECURE RAG pipeline (8 steps)."""
        log.info("secure_rag.query.start", question=question[:80])

        all_guardrails = []
        all_defenses = []
        all_error_codes = []
        all_messages = []

        # STEP 1: INPUT GUARDRAILS
        log.info("secure_rag.step", step="1/8", name="input_guardrails")
        can_proceed, processed_query, triggered, error_codes, messages = run_input_guardrails(
            question, self.llm
        )
        all_guardrails.extend(triggered)
        all_error_codes.extend(error_codes)
        all_messages.extend(messages)

        if not can_proceed:
            answer = messages[-1] if messages else "Your query was blocked by our safety system."
            response = SecureRAGResponse(
                answer=answer,
                guardrails_triggered=all_guardrails,
                error_codes=all_error_codes,
                was_blocked=True,
                messages=all_messages,
            )
            self._log_response(question, response)
            log.info("secure_rag.query.blocked", step="input_guardrails")
            return response

        # STEP 2: PROMPT INJECTION DEFENSES
        log.info("secure_rag.step", step="2/8", name="prompt_defenses")
        defense_ok, refusal_msg, defenses = run_prompt_defenses(processed_query)
        all_defenses.extend(defenses)

        if not defense_ok:
            all_error_codes.append(ErrorCode.POLICY_BLOCK)
            response = SecureRAGResponse(
                answer=refusal_msg,
                guardrails_triggered=all_guardrails,
                defenses_triggered=all_defenses,
                error_codes=all_error_codes,
                was_blocked=True,
                messages=all_messages,
            )
            self._log_response(question, response)
            log.info("secure_rag.query.blocked", step="prompt_defenses")
            return response

        # STEP 3: EMBED & RETRIEVE
        log.info("secure_rag.step", step="3/8", name="embed_retrieve")
        docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(
            processed_query, k=self.num_chunks
        )

        scores = [score for _, score in docs_with_scores]
        docs = [doc for doc, _ in docs_with_scores]

        log.info("secure_rag.retrieved", num_chunks=len(docs),
                 top_score=round(max(scores), 4) if scores else 0)

        # STEP 4: RETRIEVAL CONFIDENCE CHECK
        log.info("secure_rag.step", step="4/8", name="retrieval_confidence")
        confidence_result = check_retrieval_confidence(
            docs_with_scores, threshold=self.confidence_threshold
        )

        if not confidence_result.passed:
            all_guardrails.append(confidence_result.guardrail_name)
            all_error_codes.append(confidence_result.error_code)
            response = SecureRAGResponse(
                answer=confidence_result.message,
                guardrails_triggered=all_guardrails,
                defenses_triggered=all_defenses,
                error_codes=all_error_codes,
                was_blocked=True,
                retrieval_scores=scores,
                messages=all_messages,
            )
            self._log_response(question, response)
            log.info("secure_rag.query.blocked", step="retrieval_confidence")
            return response

        # STEP 5: CONTEXT WRAPPING
        log.info("secure_rag.step", step="5/8", name="context_wrapping")
        context, citations = format_docs_with_sources(docs)
        wrapped_context = wrap_context_with_delimiters(context)

        # STEP 6: LLM GENERATION
        log.info("secure_rag.step", step="6/8", name="llm_generation")
        try:
            def generate():
                return self.chain.invoke({
                    "context": wrapped_context,
                    "question": processed_query,
                })

            answer = run_with_timeout(generate, self.timeout_seconds)
            log.info("secure_rag.llm_response", answer_chars=len(answer))

        except TimeoutError:
            all_error_codes.append(ErrorCode.LLM_TIMEOUT)
            all_guardrails.append("LLM_TIMEOUT")
            response = SecureRAGResponse(
                answer="The system took too long to respond. Please try again with a simpler question.",
                guardrails_triggered=all_guardrails,
                defenses_triggered=all_defenses,
                error_codes=all_error_codes,
                was_blocked=True,
                retrieval_scores=scores,
                messages=all_messages,
            )
            self._log_response(question, response)
            log.warning("secure_rag.query.timeout", timeout_seconds=self.timeout_seconds)
            return response

        except Exception as e:
            all_error_codes.append(ErrorCode.POLICY_BLOCK)
            response = SecureRAGResponse(
                answer=f"An error occurred while generating the answer: {str(e)}",
                guardrails_triggered=all_guardrails,
                defenses_triggered=all_defenses,
                error_codes=all_error_codes,
                was_blocked=True,
                retrieval_scores=scores,
                messages=all_messages,
            )
            self._log_response(question, response)
            log.error("secure_rag.query.error", error=str(e))
            return response

        # STEP 7: OUTPUT VALIDATION
        log.info("secure_rag.step", step="7/8", name="output_validation")
        output_safe, output_reason = validate_output(answer)
        if not output_safe:
            all_defenses.append("OUTPUT_VALIDATION")
            all_error_codes.append(ErrorCode.POLICY_BLOCK)
            answer = (
                "I can only answer questions about Nova Scotia driving rules. "
                "Please rephrase your question."
            )
            all_messages.append(f"Response was filtered: {output_reason}")
            log.warning("secure_rag.output_filtered", reason=output_reason)

        answer, was_truncated = check_response_length(answer, self.max_response_words)
        if was_truncated:
            all_guardrails.append("RESPONSE_LENGTH_LIMIT")

        # STEP 8: FAITHFULNESS EVALUATION
        log.info("secure_rag.step", step="8/8", name="faithfulness_eval")
        faithfulness_score = check_faithfulness(answer, context, self.llm)
        evaluate_retrieval_relevance(scores, self.confidence_threshold)

        # BUILD FINAL RESPONSE
        response = SecureRAGResponse(
            answer=answer,
            sources=citations,
            source_documents=docs,
            guardrails_triggered=all_guardrails,
            defenses_triggered=all_defenses,
            error_codes=all_error_codes,
            was_blocked=False,
            faithfulness_score=faithfulness_score,
            retrieval_scores=scores,
            messages=all_messages,
        )

        self._log_response(question, response)

        log.info("secure_rag.query.done",
                 answer_chars=len(answer),
                 faithfulness=faithfulness_score,
                 guardrails=all_guardrails or None,
                 defenses=all_defenses or None)

        return response

    def _log_response(self, query: str, response: SecureRAGResponse) -> None:
        """Log the query and response to the security logger."""
        self.logger.log_query(
            query=query,
            guardrails_triggered=response.guardrails_triggered,
            error_codes=response.error_codes,
            defenses_triggered=response.defenses_triggered,
            was_blocked=response.was_blocked,
            faithfulness_score=response.faithfulness_score,
            retrieval_scores=response.retrieval_scores,
            answer_length_words=len(response.answer.split()),
        )

    def batch_query(self, questions: list[str]) -> list[SecureRAGResponse]:
        """Process multiple questions through the secure pipeline."""
        return [self.query(q) for q in questions]
