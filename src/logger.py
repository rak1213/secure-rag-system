"""Logging Dashboard Module — Track and summarize system activity."""

from dataclasses import dataclass, field
from datetime import datetime

from .logging_config import get_logger

log = get_logger(__name__)


@dataclass
class QueryLog:
    """Log entry for a single query processed by the system."""
    query: str
    timestamp: str
    guardrails_triggered: list[str] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)
    defenses_triggered: list[str] = field(default_factory=list)
    was_blocked: bool = False
    faithfulness_score: float = -1.0
    retrieval_scores: list[float] = field(default_factory=list)
    answer_length_words: int = 0


class SecurityLogger:
    """Tracks all security events and query results for dashboard reporting."""

    def __init__(self):
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
        """Record a query and its processing results."""
        entry = QueryLog(
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
        self.logs.append(entry)
        log.debug("security_logger.query_logged", was_blocked=was_blocked)

    def get_metrics(self) -> dict:
        """Get metrics as a dictionary (used by the API)."""
        total = len(self.logs)
        blocked = sum(1 for entry in self.logs if entry.was_blocked)
        faith_scores = [entry.faithfulness_score for entry in self.logs if entry.faithfulness_score >= 0]
        all_retrieval_scores = []
        for entry in self.logs:
            all_retrieval_scores.extend(entry.retrieval_scores)

        guardrail_counts = {}
        for entry in self.logs:
            for g in entry.guardrails_triggered:
                guardrail_counts[g] = guardrail_counts.get(g, 0) + 1

        defense_counts = {}
        for entry in self.logs:
            for d in entry.defenses_triggered:
                defense_counts[d] = defense_counts.get(d, 0) + 1

        return {
            "session_start": self.start_time,
            "total_queries": total,
            "queries_processed": total - blocked,
            "queries_blocked": blocked,
            "guardrails_triggered": guardrail_counts,
            "defenses_triggered": defense_counts,
            "avg_faithfulness": round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None,
            "avg_retrieval_score": round(sum(all_retrieval_scores) / len(all_retrieval_scores), 4) if all_retrieval_scores else None,
        }

    def get_dashboard_text(self) -> str:
        """Generate the dashboard summary as a formatted string."""
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  SECURITY LOGGING DASHBOARD")
        lines.append(f"  Session started: {self.start_time}")
        lines.append("=" * 70)

        total = len(self.logs)
        blocked = sum(1 for entry in self.logs if entry.was_blocked)
        processed = total - blocked
        lines.append("\n  QUERY SUMMARY")
        lines.append(f"  {'Total queries:':<35} {total}")
        lines.append(f"  {'Successfully processed:':<35} {processed}")
        lines.append(f"  {'Blocked by guardrails/defenses:':<35} {blocked}")

        guardrail_counts = {}
        for entry in self.logs:
            for g in entry.guardrails_triggered:
                guardrail_counts[g] = guardrail_counts.get(g, 0) + 1

        lines.append("\n  GUARDRAILS TRIGGERED (by type)")
        if guardrail_counts:
            for name, count in sorted(guardrail_counts.items()):
                lines.append(f"  {'  ' + name + ':':<35} {count} time(s)")
        else:
            lines.append(f"  {'  (none triggered)'}")

        defense_counts = {}
        for entry in self.logs:
            for d in entry.defenses_triggered:
                defense_counts[d] = defense_counts.get(d, 0) + 1

        injection_blocked = sum(1 for entry in self.logs if entry.defenses_triggered)
        lines.append("\n  PROMPT INJECTION DEFENSE")
        lines.append(f"  {'Total injection attempts blocked:':<35} {injection_blocked}")
        if defense_counts:
            for name, count in sorted(defense_counts.items()):
                lines.append(f"  {'  ' + name + ':':<35} {count} time(s)")

        faith_scores = [entry.faithfulness_score for entry in self.logs if entry.faithfulness_score >= 0]
        lines.append("\n  EVALUATION METRICS")
        if faith_scores:
            avg_faith = sum(faith_scores) / len(faith_scores)
            lines.append(f"  {'Faithfulness scores evaluated:':<35} {len(faith_scores)}")
            lines.append(f"  {'Average faithfulness score:':<35} {avg_faith:.2f}")
            lines.append(f"  {'Min faithfulness score:':<35} {min(faith_scores):.2f}")
            lines.append(f"  {'Max faithfulness score:':<35} {max(faith_scores):.2f}")
        else:
            lines.append(f"  {'Faithfulness scores evaluated:':<35} 0")

        all_retrieval_scores = []
        for entry in self.logs:
            all_retrieval_scores.extend(entry.retrieval_scores)

        if all_retrieval_scores:
            avg_retrieval = sum(all_retrieval_scores) / len(all_retrieval_scores)
            lines.append(f"  {'Avg retrieval relevance score:':<35} {avg_retrieval:.4f}")
        else:
            lines.append(f"  {'Avg retrieval relevance score:':<35} N/A")

        error_counts = {}
        for entry in self.logs:
            for e in entry.error_codes:
                error_counts[e] = error_counts.get(e, 0) + 1

        lines.append("\n  ERROR CODES ISSUED")
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
