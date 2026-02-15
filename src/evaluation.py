"""
Evaluation Module — Measure answer quality with 3 evaluation metrics.

This module implements ALL THREE evaluation signals:

  Metric 1: FAITHFULNESS CHECK
    → Use the LLM to score whether the answer is supported by retrieved chunks.
    → Scores 0.0 to 1.0 (1.0 = fully faithful, 0.0 = completely hallucinated)

  Metric 2: RETRIEVAL RELEVANCE
    → Log similarity scores of retrieved chunks for each query.
    → Report average relevance and flag low-scoring queries.

  Metric 3: REFUSAL ACCURACY
    → Test with answerable and unanswerable questions.
    → Report whether the system correctly answered vs. correctly refused.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
    llm: ChatGoogleGenerativeAI,
) -> float:
    """
    Use the LLM to evaluate whether the answer is faithful to the context.

    WHY: A RAG system should only generate answers grounded in the retrieved
    documents. If the answer contains information NOT in the context, it's
    hallucinating — making things up. This is dangerous in a driving rules
    system where wrong info could lead to accidents.

    HOW: We send the answer and context to Gemini with a dedicated evaluation
    prompt. The LLM acts as a "judge" and scores the faithfulness from 0 to 1.
    This is a separate LLM call from the answer generation — the LLM is
    evaluating its own output, which provides a useful (if imperfect) signal.

    Args:
        answer: The generated answer to evaluate.
        context: The retrieved context that was used to generate the answer.
        llm: Google Gemini LLM instance.

    Returns:
        Float score between 0.0 and 1.0 (1.0 = fully faithful).
    """
    print(f"    [Eval: Faithfulness] Asking LLM to score answer against context...")

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

        # Parse the score from the response
        score_text = result.strip()
        # Extract the first float-like number from the response
        import re
        match = re.search(r'(\d+\.?\d*)', score_text)
        if match:
            score = float(match.group(1))
            score = max(0.0, min(1.0, score))  # Clamp to 0-1
        else:
            print(f"    [Eval: Faithfulness] Could not parse score from: '{score_text}', defaulting to 0.5")
            score = 0.5

        print(f"    [Eval: Faithfulness] Score: {score:.2f}")
        if score >= 0.8:
            print(f"      Verdict: HIGH faithfulness — answer is well-grounded in context")
        elif score >= 0.5:
            print(f"      Verdict: MEDIUM faithfulness — some claims may not be in context")
        else:
            print(f"      Verdict: LOW faithfulness — answer may contain hallucinations")

        return score

    except Exception as e:
        print(f"    [Eval: Faithfulness] ERROR — Could not evaluate: {e}")
        return -1.0  # Indicates evaluation failure


# ============================================================================
# METRIC 2: Retrieval Relevance
# ============================================================================

def evaluate_retrieval_relevance(
    scores: list[float],
    threshold: float = 0.3,
) -> dict:
    """
    Analyze the similarity scores of retrieved chunks.

    WHY: If the retrieved chunks have low similarity scores, the answer
    is likely to be poor. Tracking relevance scores helps us:
    1. Identify queries where the system struggles
    2. Tune the retrieval threshold
    3. Detect potential gaps in the document collection

    HOW: We compute basic statistics on the retrieval scores and flag
    any query where the top chunk scores below the threshold.

    Args:
        scores: List of similarity/relevance scores from retrieval.
        threshold: Minimum acceptable score (default 0.3).

    Returns:
        Dictionary with relevance statistics.
    """
    print(f"    [Eval: Retrieval Relevance] Analyzing {len(scores)} chunk scores...")

    if not scores:
        print(f"    [Eval: Retrieval Relevance] No scores to analyze")
        return {
            "num_chunks": 0,
            "top_score": 0.0,
            "avg_score": 0.0,
            "min_score": 0.0,
            "above_threshold": 0,
            "below_threshold": 0,
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

    print(f"      Top score: {top_score:.4f}")
    print(f"      Avg score: {avg_score:.4f}")
    print(f"      Min score: {min_score:.4f}")
    print(f"      Chunks above threshold ({threshold}): {above}/{len(scores)}")

    if flagged:
        print(f"      FLAG: Top score is below threshold — retrieval quality is poor")
    else:
        print(f"      Retrieval quality: GOOD")

    return result


# ============================================================================
# METRIC 3: Refusal Accuracy
# ============================================================================

# Define test cases for refusal accuracy evaluation
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
    """
    Evaluate whether the system correctly answered answerable questions
    and correctly refused unanswerable ones.

    WHY: A good RAG system should:
    - ANSWER questions that are in its domain and covered by the documents
    - REFUSE questions that are off-topic or not covered by the documents
    Getting either wrong is a failure:
    - Answering an unanswerable question = hallucination
    - Refusing an answerable question = too restrictive

    HOW: We check each result to see if it was an answer or a refusal,
    then compare against whether it should have been answered or refused.

    Args:
        results: List of dicts with keys:
            - "query": The question asked
            - "answer": The system's response
            - "should_answer": True if the question should be answerable
            - "error_code": Any error code that was returned

    Returns:
        Dictionary with accuracy statistics.
    """
    print(f"\n  === REFUSAL ACCURACY EVALUATION ===")

    correct = 0
    total = len(results)
    details = []

    for r in results:
        query = r["query"]
        answer = r["answer"]
        should_answer = r["should_answer"]
        error_code = r.get("error_code")

        # Determine if the system actually answered or refused
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

        print(f"    Q: \"{query[:50]}...\"")
        print(f"      Expected: {'ANSWER' if should_answer else 'REFUSE'} | Got: {'ANSWER' if actually_answered else 'REFUSE'} | {status}")

    accuracy = correct / total if total > 0 else 0.0

    print(f"\n    Refusal Accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"  === END REFUSAL ACCURACY EVALUATION ===\n")

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "details": details,
    }
