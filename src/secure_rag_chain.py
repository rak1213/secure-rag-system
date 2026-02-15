"""
Secure RAG Chain — Enhanced RAG pipeline with all guardrails and defenses.

This module wraps the base RAG pipeline with three security layers:

  Layer 1: INPUT GUARDRAILS   → Validate queries before processing
  Layer 2: PROMPT DEFENSES    → Protect against prompt injection attacks
  Layer 3: OUTPUT GUARDRAILS  → Validate responses before returning

The flow for each query is:
  ┌──────────────────────────────────────────────────────────────┐
  │ User Query                                                    │
  │   ↓                                                           │
  │ [INPUT GUARDRAILS] — length, PII, off-topic checks           │
  │   ↓                                                           │
  │ [PROMPT DEFENSES] — injection detection, jailbreak check     │
  │   ↓                                                           │
  │ [RETRIEVAL] — similarity search WITH relevance scores        │
  │   ↓                                                           │
  │ [RETRIEVAL CONFIDENCE] — check if chunks are relevant enough │
  │   ↓                                                           │
  │ [CONTEXT WRAPPING] — wrap context in safety delimiters       │
  │   ↓                                                           │
  │ [LLM GENERATION] — hardened system prompt + timeout          │
  │   ↓                                                           │
  │ [OUTPUT VALIDATION] — check for leaked prompts, length cap   │
  │   ↓                                                           │
  │ [EVALUATION] — faithfulness scoring                          │
  │   ↓                                                           │
  │ Secure Response                                               │
  └──────────────────────────────────────────────────────────────┘
"""

import os
from dataclasses import dataclass, field
from langchain_google_genai import ChatGoogleGenerativeAI
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


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SecureRAGResponse:
    """
    Response from the secure RAG system.

    Extends the original RAGResponse with security and evaluation metadata.

    Attributes:
        answer: The generated answer text (may be a refusal message).
        sources: List of source citations.
        source_documents: Retrieved Document objects.
        guardrails_triggered: Which guardrails activated for this query.
        defenses_triggered: Which prompt defenses activated.
        error_codes: Any error codes generated.
        was_blocked: Whether the query was blocked before reaching the LLM.
        faithfulness_score: LLM faithfulness evaluation score (0-1).
        retrieval_scores: Similarity scores from chunk retrieval.
        messages: Warning/info messages for the user.
    """
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

        # Show any warnings/messages first
        if self.messages:
            for msg in self.messages:
                parts.append(f"[!] {msg}")
            parts.append("")

        # The answer
        parts.append(self.answer)

        # Sources
        if self.sources:
            parts.append("\n--- Sources ---")
            for source in self.sources:
                parts.append(f"  {source}")

        return "\n".join(parts)


# ============================================================================
# SECURE RAG CHAIN
# ============================================================================

class SecureRAGChain:
    """
    Production-ready RAG chain with guardrails, prompt injection defense,
    and evaluation built in.

    Wraps the core components (Jina embeddings, ChromaDB, Google Gemini)
    with three security layers.
    """

    def __init__(
        self,
        vector_store: Chroma,
        embeddings: Embeddings,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        num_chunks: int = 4,
        confidence_threshold: float = 0.3,
        timeout_seconds: int = 30,
        max_response_words: int = 500,
        logger: SecurityLogger | None = None,
    ):
        """
        Initialize the Secure RAG chain.

        Args:
            vector_store: ChromaDB vector store with indexed documents.
            embeddings: Jina AI embeddings model.
            api_key: Google API key (reads from env if not provided).
            model_name: Gemini model name.
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

        # Initialize LLM
        google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=0.1,
        )

        # Build the LCEL chain with HARDENED system prompt
        hardened_prompt = get_hardened_system_prompt()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", hardened_prompt),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

        print(f"\n  SecureRAGChain initialized:")
        print(f"    Model: {model_name}")
        print(f"    Chunks per query: {num_chunks}")
        print(f"    Confidence threshold: {confidence_threshold}")
        print(f"    Timeout: {timeout_seconds}s")
        print(f"    Max response: {max_response_words} words")
        print(f"    Defenses: ALL 5 active")
        print(f"    Guardrails: Input + Output + Execution limits")

    def query(self, question: str) -> SecureRAGResponse:
        """
        Process a question through the SECURE RAG pipeline.

        This is the main method. Each step is logged in detail so you
        can trace exactly what happened at every security checkpoint.

        Steps:
          1. Input Guardrails (length, PII, off-topic)
          2. Prompt Injection Defenses (jailbreak, sanitization)
          3. Embed & Retrieve (with relevance scores)
          4. Retrieval Confidence Check
          5. Context Wrapping (instruction-data separation)
          6. LLM Generation (hardened prompt + timeout)
          7. Output Validation (leak check + length cap)
          8. Faithfulness Evaluation

        Args:
            question: The user's question.

        Returns:
            SecureRAGResponse with answer and all security metadata.
        """
        print(f"\n{'='*70}")
        print(f"  SECURE RAG QUERY")
        print(f"  Question: \"{question[:80]}{'...' if len(question) > 80 else ''}\"")
        print(f"{'='*70}")

        all_guardrails = []
        all_defenses = []
        all_error_codes = []
        all_messages = []

        # ==================================================================
        # STEP 1: INPUT GUARDRAILS
        # ==================================================================
        print(f"\n  STEP 1/8: INPUT GUARDRAILS")
        can_proceed, processed_query, triggered, error_codes, messages = run_input_guardrails(
            question, self.llm
        )
        all_guardrails.extend(triggered)
        all_error_codes.extend(error_codes)
        all_messages.extend(messages)

        if not can_proceed:
            # Query was blocked by input guardrails
            answer = messages[-1] if messages else "Your query was blocked by our safety system."
            response = SecureRAGResponse(
                answer=answer,
                guardrails_triggered=all_guardrails,
                error_codes=all_error_codes,
                was_blocked=True,
                messages=all_messages,
            )
            self._log_response(question, response)
            print(f"\n  RESULT: BLOCKED by input guardrails")
            print(f"{'='*70}\n")
            return response

        # ==================================================================
        # STEP 2: PROMPT INJECTION DEFENSES
        # ==================================================================
        print(f"\n  STEP 2/8: PROMPT INJECTION DEFENSES")
        defense_ok, refusal_msg, defenses = run_prompt_defenses(processed_query)
        all_defenses.extend(defenses)

        if not defense_ok:
            # Prompt injection detected — return standardized refusal
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
            print(f"\n  RESULT: BLOCKED by prompt injection defense")
            print(f"{'='*70}\n")
            return response

        # ==================================================================
        # STEP 3: EMBED & RETRIEVE (with relevance scores)
        # ==================================================================
        print(f"\n  STEP 3/8: EMBED & RETRIEVE WITH SCORES")
        print(f"    Embedding query and searching vector store...")

        # query_vector = self.embeddings.embed_query(processed_query)
        # print(f"    Query embedded: {len(query_vector)} dimensions")

        # Use similarity_search_with_relevance_scores for confidence checking
        docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(
            processed_query, k=self.num_chunks
        )

        scores = [score for _, score in docs_with_scores]
        docs = [doc for doc, _ in docs_with_scores]

        print(f"    Retrieved {len(docs)} chunks with scores:")
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"      Chunk {i}: score={score:.4f} | \"{preview}...\"")

        # ==================================================================
        # STEP 4: RETRIEVAL CONFIDENCE CHECK
        # ==================================================================
        print(f"\n  STEP 4/8: RETRIEVAL CONFIDENCE CHECK")
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
            print(f"\n  RESULT: BLOCKED by retrieval confidence check")
            print(f"{'='*70}\n")
            return response

        # ==================================================================
        # STEP 5: CONTEXT WRAPPING (Defense 3 - Instruction-Data Separation)
        # ==================================================================
        print(f"\n  STEP 5/8: CONTEXT WRAPPING (instruction-data separation)")
        context, citations = format_docs_with_sources(docs)
        wrapped_context = wrap_context_with_delimiters(context)
        print(f"    Context: {len(wrapped_context)} chars from {len(docs)} chunks")

        # ==================================================================
        # STEP 6: LLM GENERATION (with hardened prompt + timeout)
        # ==================================================================
        print(f"\n  STEP 6/8: LLM GENERATION (hardened prompt + timeout)")
        print(f"    Sending to Gemini with hardened system prompt...")
        print(f"    Timeout: {self.timeout_seconds}s")

        try:
            def generate():
                return self.chain.invoke({
                    "context": wrapped_context,
                    "question": processed_query,
                })

            answer = run_with_timeout(generate, self.timeout_seconds)
            print(f"    LLM response received: {len(answer)} chars")

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
            print(f"\n  RESULT: TIMEOUT — LLM exceeded {self.timeout_seconds}s")
            print(f"{'='*70}\n")
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
            print(f"\n  RESULT: ERROR — {e}")
            print(f"{'='*70}\n")
            return response

        # ==================================================================
        # STEP 7: OUTPUT VALIDATION (Defense 4 + length cap)
        # ==================================================================
        print(f"\n  STEP 7/8: OUTPUT VALIDATION")

        # Defense 4: Check for leaked system prompt or suspicious content
        output_safe, output_reason = validate_output(answer)
        if not output_safe:
            all_defenses.append("OUTPUT_VALIDATION")
            all_error_codes.append(ErrorCode.POLICY_BLOCK)
            answer = (
                "I can only answer questions about Nova Scotia driving rules. "
                "Please rephrase your question."
            )
            all_messages.append(f"Response was filtered: {output_reason}")
            print(f"    Output REPLACED due to: {output_reason}")

        # Response length cap
        answer, was_truncated = check_response_length(answer, self.max_response_words)
        if was_truncated:
            all_guardrails.append("RESPONSE_LENGTH_LIMIT")

        # ==================================================================
        # STEP 8: FAITHFULNESS EVALUATION
        # ==================================================================
        print(f"\n  STEP 8/8: FAITHFULNESS EVALUATION")
        faithfulness_score = check_faithfulness(answer, context, self.llm)

        # Retrieval relevance evaluation
        relevance_stats = evaluate_retrieval_relevance(scores, self.confidence_threshold)

        # ==================================================================
        # BUILD FINAL RESPONSE
        # ==================================================================
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

        print(f"\n  RESULT: SUCCESS")
        print(f"    Answer: {len(answer)} chars")
        print(f"    Faithfulness: {faithfulness_score:.2f}")
        print(f"    Guardrails triggered: {all_guardrails or 'NONE'}")
        print(f"    Defenses triggered: {all_defenses or 'NONE'}")
        print(f"{'='*70}\n")

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
        """
        Process multiple questions through the secure pipeline.

        Args:
            questions: List of questions to answer.

        Returns:
            List of SecureRAGResponse objects.
        """
        return [self.query(q) for q in questions]
