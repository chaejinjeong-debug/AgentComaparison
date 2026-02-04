"""Metrics models for agent evaluation.

Defines quality and performance metrics for measuring agent behavior.
"""

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityMetrics:
    """Quality metrics for agent responses.

    Attributes:
        total_tests: Total number of test cases
        passed: Number of passed tests
        failed: Number of failed tests
        accuracy: Pass rate (0.0 - 1.0)
        scores: Individual test scores
        failure_reasons: Reasons for failures
    """

    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    accuracy: float = 0.0
    scores: list[float] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)

    def add_result(self, passed: bool, score: float = 1.0, reason: str = "") -> None:
        """Add a test result.

        Args:
            passed: Whether the test passed
            score: Score for this test (0.0 - 1.0)
            reason: Failure reason if not passed
        """
        self.total_tests += 1
        self.scores.append(score)

        if passed:
            self.passed += 1
        else:
            self.failed += 1
            if reason:
                self.failure_reasons.append(reason)

        self._recalculate_accuracy()

    def _recalculate_accuracy(self) -> None:
        """Recalculate accuracy based on results."""
        if self.total_tests > 0:
            self.accuracy = self.passed / self.total_tests

    @property
    def average_score(self) -> float:
        """Calculate average score across all tests."""
        if not self.scores:
            return 0.0
        return statistics.mean(self.scores)

    def meets_threshold(self, threshold: float = 0.85) -> bool:
        """Check if quality meets the threshold.

        Args:
            threshold: Minimum accuracy threshold

        Returns:
            True if accuracy meets or exceeds threshold
        """
        return self.accuracy >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "accuracy": round(self.accuracy, 4),
            "average_score": round(self.average_score, 4),
            "failure_count": len(self.failure_reasons),
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for agent responses.

    Attributes:
        latencies_ms: List of response latencies in milliseconds
        request_count: Total number of requests
        error_count: Number of errors
        timeout_count: Number of timeouts
    """

    latencies_ms: list[float] = field(default_factory=list)
    request_count: int = 0
    error_count: int = 0
    timeout_count: int = 0

    def add_latency(self, latency_ms: float) -> None:
        """Add a latency measurement.

        Args:
            latency_ms: Response latency in milliseconds
        """
        self.latencies_ms.append(latency_ms)
        self.request_count += 1

    def add_error(self) -> None:
        """Record an error."""
        self.error_count += 1
        self.request_count += 1

    def add_timeout(self) -> None:
        """Record a timeout."""
        self.timeout_count += 1
        self.request_count += 1

    @property
    def p50_ms(self) -> float:
        """Calculate P50 (median) latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)

    @property
    def p90_ms(self) -> float:
        """Calculate P90 latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        index = int(len(sorted_latencies) * 0.90)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    @property
    def p99_ms(self) -> float:
        """Calculate P99 latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        index = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    @property
    def mean_ms(self) -> float:
        """Calculate mean latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)

    @property
    def min_ms(self) -> float:
        """Get minimum latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        return min(self.latencies_ms)

    @property
    def max_ms(self) -> float:
        """Get maximum latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        return max(self.latencies_ms)

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count + self.timeout_count) / self.request_count

    @property
    def throughput(self) -> float:
        """Calculate throughput (successful requests per second).

        Note: This requires total_time to be set externally.
        Returns the request count for now.
        """
        return self.request_count - self.error_count - self.timeout_count

    def meets_sla(
        self,
        p50_threshold_ms: float = 2000.0,
        p99_threshold_ms: float = 10000.0,
        error_rate_threshold: float = 0.05,
    ) -> tuple[bool, list[str]]:
        """Check if performance meets SLA requirements.

        Args:
            p50_threshold_ms: Maximum P50 latency (default: 2s)
            p99_threshold_ms: Maximum P99 latency (default: 10s)
            error_rate_threshold: Maximum error rate (default: 5%)

        Returns:
            Tuple of (meets_sla, list of violations)
        """
        violations = []

        if self.p50_ms > p50_threshold_ms:
            violations.append(f"P50 latency {self.p50_ms:.0f}ms > {p50_threshold_ms:.0f}ms")

        if self.p99_ms > p99_threshold_ms:
            violations.append(f"P99 latency {self.p99_ms:.0f}ms > {p99_threshold_ms:.0f}ms")

        if self.error_rate > error_rate_threshold:
            violations.append(f"Error rate {self.error_rate:.2%} > {error_rate_threshold:.2%}")

        return len(violations) == 0, violations

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "timeout_count": self.timeout_count,
            "error_rate": round(self.error_rate, 4),
            "latency": {
                "min_ms": round(self.min_ms, 2),
                "mean_ms": round(self.mean_ms, 2),
                "p50_ms": round(self.p50_ms, 2),
                "p90_ms": round(self.p90_ms, 2),
                "p99_ms": round(self.p99_ms, 2),
                "max_ms": round(self.max_ms, 2),
            },
        }
