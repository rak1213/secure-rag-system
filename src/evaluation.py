"""
Evaluation Module — Measure answer quality with 3 evaluation metrics.

  Metric 1: FAITHFULNESS CHECK — LLM-based scoring of answer vs context
  Metric 2: RETRIEVAL RELEVANCE — Similarity score analysis
  Metric 3: REFUSAL ACCURACY — Correct answer vs refuse decisions
"""

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .logging_config import get_logger

log = get_logger(__name__)


# ============================================================================
# METRIC 1: Faithfulness Check
# ============================================================================

FAITHFULNESS_PROMPT = """You are an evaluation judge. Your job is to determine if an ANSWER is
faithfully supported by the RETRIEVED CONTEXT.

Rules:
1. Check each claim in the answer against the context.
2. If ALL claims in the answer are supported by the context, score = 1.0
3. If SOME claims are supported but others are not in the context, score = 0.5
4. If the answer contains significant information NOT in the context, score = 0.0
5. If the answer is a refusal ("I don't have information" etc.), score = 1.0 (correct behavior)

RETRIEVED CONTEXT:
{context}

ANSWER TO EVALUATE:
{answer}

Respond with ONLY a number between 0.0 and 1.0 (e.g., "0.8" or "1.0").
Do NOT explain your reasoning. Just output the score."""


def check_faithfulness(
    answer: str,
    context: str,
    llm: BaseChatModel,
) -> float:
    """Use the LLM to evaluate whether the answer is faithful to the context.

    Returns:
        Float score between 0.0 and 1.0 (1.0 = fully faithful).
    """
    log.debug("eval.faithfulness.check")

    prompt = ChatPromptTemplate.from_messages([
        ("system", FAITHFULNESS_PROMPT),
        ("human", "Score the faithfulness of this answer."),
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "context": context,
            "answer": answer,
        })

        score_text = result.strip()
        match = re.search(r'(\d+\.?\d*)', score_text)
        if match:
            score = float(match.group(1))
            score = max(0.0, min(1.0, score))
        else:
            log.warning("eval.faithfulness.parse_error", raw_response=score_text)
            score = 0.5

        level = "high" if score >= 0.8 else ("medium" if score >= 0.5 else "low")
        log.info("eval.faithfulness.scored", score=score, level=level)
        return score

    except Exception as e:
        log.error("eval.faithfulness.error", error=str(e))
        return -1.0


# ============================================================================
# METRIC 2: Retrieval Relevance
# ============================================================================

def evaluate_retrieval_relevance(
    scores: list[float],
    threshold: float = 0.3,
) -> dict:
    """Analyze the similarity scores of retrieved chunks.

    Returns:
        Dictionary with relevance statistics.
    """
    if not scores:
        log.debug("eval.retrieval_relevance.no_scores")
        return {
            "num_chunks": 0, "top_score": 0.0, "avg_score": 0.0,
            "min_score": 0.0, "above_threshold": 0, "below_threshold": 0,
            "flagged": True,
        }

    top_score = max(scores)
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    above = sum(1 for s in scores if s >= threshold)
    below = sum(1 for s in scores if s < threshold)
    flagged = top_score < threshold

    result = {
        "num_chunks": len(scores),
        "top_score": round(top_score, 4),
        "avg_score": round(avg_score, 4),
        "min_score": round(min_score, 4),
        "above_threshold": above,
        "below_threshold": below,
        "flagged": flagged,
    }

    log.info("eval.retrieval_relevance",
             top_score=result["top_score"],
             avg_score=result["avg_score"],
             above_threshold=above,
             flagged=flagged)

    return result


# ============================================================================
# METRIC 3: Refusal Accuracy
# ============================================================================

ANSWERABLE_QUESTIONS = [
    "What are the rules for passing a school bus?",
    "When must you yield to pedestrians?",
    "What should you do when approached by an emergency vehicle?",
]

UNANSWERABLE_QUESTIONS = [
    "What is the recipe for chocolate cake?",
    "What is the capital of Australia?",
    "How do I fix a Python syntax error?",
]


def evaluate_refusal_accuracy(results: list[dict]) -> dict:
    """Evaluate whether the system correctly answered answerable questions
    and correctly refused unanswerable ones.

    Returns:
        Dictionary with accuracy statistics.
    """
    log.info("eval.refusal_accuracy.start", total_results=len(results))

    correct = 0
    total = len(results)
    details = []

    for r in results:
        query = r["query"]
        answer = r["answer"]
        should_answer = r["should_answer"]
        error_code = r.get("error_code")

        refusal_indicators = [
            "I can only answer questions about",
            "I don't have enough information",
            "I could not find information",
            "I cannot comply",
            "off-topic",
            "not about driving",
            "query is empty",
            "too long to respond",
        ]
        actually_refused = (
            error_code in ["OFF_TOPIC", "RETRIEVAL_EMPTY", "POLICY_BLOCK", "EMPTY_QUERY", "QUERY_TOO_LONG", "LLM_TIMEOUT"]
            or any(indicator.lower() in answer.lower() for indicator in refusal_indicators)
        )

        actually_answered = not actually_refused

        if should_answer and actually_answered:
            status = "CORRECT (answered answerable)"
            correct += 1
        elif not should_answer and actually_refused:
            status = "CORRECT (refused unanswerable)"
            correct += 1
        elif should_answer and actually_refused:
            status = "WRONG (refused answerable)"
        else:
            status = "WRONG (answered unanswerable)"

        details.append({
            "query": query[:60],
            "should_answer": should_answer,
            "actually_answered": actually_answered,
            "status": status,
        })

        log.debug("eval.refusal_accuracy.result",
                   query_preview=query[:50], status=status)

    accuracy = correct / total if total > 0 else 0.0
    log.info("eval.refusal_accuracy.done",
             correct=correct, total=total, accuracy=round(accuracy, 4))

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "details": details,
    }
