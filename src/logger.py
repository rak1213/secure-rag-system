"""
Logging Dashboard Module — Track and summarize system activity.

This module provides a simple logging system that records:
  - Total queries processed
  - Guardrails triggered (count by type)
  - Prompt injection attempts blocked
  - Faithfulness scores for evaluation
  - Retrieval relevance scores

At the end of a run, it prints a clean summary dashboard.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class QueryLog:
    """Log entry for a single query processed by the system."""
    query: str
    timestamp: str
    guardrails_triggered: list[str] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)
    defenses_triggered: list[str] = field(default_factory=list)
    was_blocked: bool = False
    faithfulness_score: float = -1.0  # -1 means not evaluated
    retrieval_scores: list[float] = field(default_factory=list)
    answer_length_words: int = 0


class SecurityLogger:
    """
    Tracks all security events and query results for dashboard reporting.

    Usage:
        logger = SecurityLogger()
        logger.log_query(...)       # Log each query
        logger.print_dashboard()    # Print summary at the end
        logger.get_dashboard_text() # Get summary as string
    """

    def __init__(self):
        """Initialize the logger with empty tracking lists."""
        self.logs: list[QueryLog] = []
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log_query(
        self,
        query: str,
        guardrails_triggered: list[str] = None,
        error_codes: list[str] = None,
        defenses_triggered: list[str] = None,
        was_blocked: bool = False,
        faithfulness_score: float = -1.0,
        retrieval_scores: list[float] = None,
        answer_length_words: int = 0,
    ) -> None:
        """
        Record a query and its processing results.

        Args:
            query: The user's query.
            guardrails_triggered: List of guardrail names that activated.
            error_codes: List of error codes generated.
            defenses_triggered: List of defense names that activated.
            was_blocked: Whether the query was blocked entirely.
            faithfulness_score: LLM faithfulness evaluation score (0-1).
            retrieval_scores: Similarity scores from retrieval.
            answer_length_words: Word count of the generated answer.
        """
        log = QueryLog(
            query=query[:100],
            timestamp=datetime.now().strftime("%H:%M:%S"),
            guardrails_triggered=guardrails_triggered or [],
            error_codes=error_codes or [],
            defenses_triggered=defenses_triggered or [],
            was_blocked=was_blocked,
            faithfulness_score=faithfulness_score,
            retrieval_scores=retrieval_scores or [],
            answer_length_words=answer_length_words,
        )
        self.logs.append(log)

    def get_dashboard_text(self) -> str:
        """
        Generate the dashboard summary as a formatted string.

        Returns:
            Multi-line string with the complete dashboard.
        """
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  SECURITY LOGGING DASHBOARD")
        lines.append(f"  Session started: {self.start_time}")
        lines.append("=" * 70)

        # --- Total queries ---
        total = len(self.logs)
        blocked = sum(1 for log in self.logs if log.was_blocked)
        processed = total - blocked
        lines.append(f"\n  QUERY SUMMARY")
        lines.append(f"  {'Total queries:':<35} {total}")
        lines.append(f"  {'Successfully processed:':<35} {processed}")
        lines.append(f"  {'Blocked by guardrails/defenses:':<35} {blocked}")

        # --- Guardrails triggered (count by type) ---
        guardrail_counts = {}
        for log in self.logs:
            for g in log.guardrails_triggered:
                guardrail_counts[g] = guardrail_counts.get(g, 0) + 1

        lines.append(f"\n  GUARDRAILS TRIGGERED (by type)")
        if guardrail_counts:
            for name, count in sorted(guardrail_counts.items()):
                lines.append(f"  {'  ' + name + ':':<35} {count} time(s)")
        else:
            lines.append(f"  {'  (none triggered)'}")

        # --- Injection attempts blocked ---
        defense_counts = {}
        for log in self.logs:
            for d in log.defenses_triggered:
                defense_counts[d] = defense_counts.get(d, 0) + 1

        injection_blocked = sum(1 for log in self.logs if log.defenses_triggered)
        lines.append(f"\n  PROMPT INJECTION DEFENSE")
        lines.append(f"  {'Total injection attempts blocked:':<35} {injection_blocked}")
        if defense_counts:
            for name, count in sorted(defense_counts.items()):
                lines.append(f"  {'  ' + name + ':':<35} {count} time(s)")

        # --- Faithfulness scores ---
        faith_scores = [log.faithfulness_score for log in self.logs if log.faithfulness_score >= 0]
        lines.append(f"\n  EVALUATION METRICS")
        if faith_scores:
            avg_faith = sum(faith_scores) / len(faith_scores)
            lines.append(f"  {'Faithfulness scores evaluated:':<35} {len(faith_scores)}")
            lines.append(f"  {'Average faithfulness score:':<35} {avg_faith:.2f}")
            lines.append(f"  {'Min faithfulness score:':<35} {min(faith_scores):.2f}")
            lines.append(f"  {'Max faithfulness score:':<35} {max(faith_scores):.2f}")
        else:
            lines.append(f"  {'Faithfulness scores evaluated:':<35} 0")

        # --- Retrieval relevance ---
        all_retrieval_scores = []
        for log in self.logs:
            all_retrieval_scores.extend(log.retrieval_scores)

        if all_retrieval_scores:
            avg_retrieval = sum(all_retrieval_scores) / len(all_retrieval_scores)
            lines.append(f"  {'Avg retrieval relevance score:':<35} {avg_retrieval:.4f}")
        else:
            lines.append(f"  {'Avg retrieval relevance score:':<35} N/A")

        # --- Error code breakdown ---
        error_counts = {}
        for log in self.logs:
            for e in log.error_codes:
                error_counts[e] = error_counts.get(e, 0) + 1

        lines.append(f"\n  ERROR CODES ISSUED")
        if error_counts:
            for code, count in sorted(error_counts.items()):
                lines.append(f"  {'  ' + code + ':':<35} {count} time(s)")
        else:
            lines.append(f"  {'  (none)'}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("  END OF DASHBOARD")
        lines.append("=" * 70)
        lines.append("")

        return "\n".join(lines)

    def print_dashboard(self) -> None:
        """Print the dashboard summary to the console."""
        print(self.get_dashboard_text())
