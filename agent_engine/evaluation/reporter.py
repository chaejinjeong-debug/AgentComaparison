"""Report generation for evaluation results.

Provides functionality to generate and export evaluation reports.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_engine.evaluation.framework import EvaluationSummary


@dataclass
class EvaluationReport:
    """Evaluation report container.

    Attributes:
        title: Report title
        generated_at: Report generation time
        summary: Evaluation summary
        config: Evaluation configuration used
    """

    title: str = "Agent Evaluation Report"
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    summary: EvaluationSummary | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "config": self.config,
            "summary": self.summary.to_dict() if self.summary else None,
        }


class EvaluationReporter:
    """Reporter for generating evaluation reports.

    Supports multiple output formats: JSON, Markdown, HTML.
    """

    def __init__(self, report: EvaluationReport) -> None:
        """Initialize the reporter.

        Args:
            report: EvaluationReport to generate output from
        """
        self.report = report

    def to_json(self, indent: int = 2) -> str:
        """Generate JSON report.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(self.report.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Generate Markdown report.

        Returns:
            Markdown string
        """
        lines = []
        lines.append(f"# {self.report.title}")
        lines.append("")
        lines.append(f"**Generated:** {self.report.generated_at.isoformat()}")
        lines.append("")

        if self.report.summary:
            summary = self.report.summary
            quality = summary.quality_metrics
            perf = summary.performance_metrics

            # Summary section
            lines.append("## Summary")
            lines.append("")
            status = "PASSED" if summary.passed_threshold else "FAILED"
            lines.append(f"**Status:** {status}")
            lines.append(f"**Threshold:** {summary.threshold:.0%}")
            lines.append(f"**Total Tests:** {summary.total_tests}")
            lines.append("")

            # Quality metrics
            lines.append("## Quality Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Accuracy | {quality.accuracy:.2%} |")
            lines.append(f"| Passed | {quality.passed} |")
            lines.append(f"| Failed | {quality.failed} |")
            lines.append(f"| Avg Score | {quality.average_score:.2f} |")
            lines.append("")

            # Performance metrics
            lines.append("## Performance Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| P50 Latency | {perf.p50_ms:.0f}ms |")
            lines.append(f"| P90 Latency | {perf.p90_ms:.0f}ms |")
            lines.append(f"| P99 Latency | {perf.p99_ms:.0f}ms |")
            lines.append(f"| Error Rate | {perf.error_rate:.2%} |")
            lines.append("")

            # SLA check
            meets_sla, violations = perf.meets_sla()
            lines.append("## SLA Compliance")
            lines.append("")
            if meets_sla:
                lines.append("All SLA requirements met.")
            else:
                lines.append("**SLA Violations:**")
                for v in violations:
                    lines.append(f"- {v}")
            lines.append("")

            # Results by category
            lines.append("## Results by Category")
            lines.append("")
            by_category = summary._results_by_category()
            if by_category:
                lines.append("| Category | Passed | Failed |")
                lines.append("|----------|--------|--------|")
                for cat, counts in by_category.items():
                    lines.append(f"| {cat} | {counts['passed']} | {counts['failed']} |")
            lines.append("")

            # Failed tests
            failed_results = [r for r in summary.results if not r.passed]
            if failed_results:
                lines.append("## Failed Tests")
                lines.append("")
                for result in failed_results[:10]:  # Limit to 10
                    lines.append(f"### {result.test_case_id}")
                    lines.append(f"- **Category:** {result.category}")
                    lines.append(f"- **Reason:** {result.reason}")
                    if result.error:
                        lines.append(f"- **Error:** {result.error}")
                    lines.append("")

        return "\n".join(lines)

    def save_json(self, path: str | Path) -> None:
        """Save report as JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    def save_markdown(self, path: str | Path) -> None:
        """Save report as Markdown file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")

    def print_summary(self) -> None:
        """Print summary to console."""
        if not self.report.summary:
            print("No evaluation summary available")
            return

        summary = self.report.summary
        quality = summary.quality_metrics
        perf = summary.performance_metrics

        print("=" * 60)
        print(self.report.title)
        print("=" * 60)
        print()

        status = "PASSED" if summary.passed_threshold else "FAILED"
        print(f"Status: {status}")
        print(f"Threshold: {summary.threshold:.0%}")
        print()

        print("Quality Metrics:")
        print(f"  - Accuracy: {quality.accuracy:.2%}")
        print(f"  - Passed: {quality.passed}/{quality.total_tests}")
        print(f"  - Failed: {quality.failed}/{quality.total_tests}")
        print()

        print("Performance Metrics:")
        print(f"  - P50 Latency: {perf.p50_ms:.0f}ms")
        print(f"  - P90 Latency: {perf.p90_ms:.0f}ms")
        print(f"  - P99 Latency: {perf.p99_ms:.0f}ms")
        print(f"  - Error Rate: {perf.error_rate:.2%}")
        print()

        meets_sla, violations = perf.meets_sla()
        if violations:
            print("SLA Violations:")
            for v in violations:
                print(f"  - {v}")
        else:
            print("SLA: All requirements met")

        print("=" * 60)
