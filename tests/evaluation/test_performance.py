"""Tests for performance metrics."""

import pytest

from agent_engine.evaluation import PerformanceMetrics, QualityMetrics


class TestQualityMetrics:
    """Tests for QualityMetrics."""

    def test_empty_metrics(self) -> None:
        """Test empty metrics initialization."""
        metrics = QualityMetrics()
        assert metrics.total_tests == 0
        assert metrics.passed == 0
        assert metrics.failed == 0
        assert metrics.accuracy == 0.0
        assert metrics.average_score == 0.0

    def test_add_passed_result(self) -> None:
        """Test adding a passed result."""
        metrics = QualityMetrics()
        metrics.add_result(passed=True, score=1.0)

        assert metrics.total_tests == 1
        assert metrics.passed == 1
        assert metrics.failed == 0
        assert metrics.accuracy == 1.0

    def test_add_failed_result(self) -> None:
        """Test adding a failed result."""
        metrics = QualityMetrics()
        metrics.add_result(passed=False, score=0.0, reason="Test failed")

        assert metrics.total_tests == 1
        assert metrics.passed == 0
        assert metrics.failed == 1
        assert metrics.accuracy == 0.0
        assert "Test failed" in metrics.failure_reasons

    def test_accuracy_calculation(self) -> None:
        """Test accuracy calculation with mixed results."""
        metrics = QualityMetrics()
        metrics.add_result(passed=True, score=1.0)
        metrics.add_result(passed=True, score=0.8)
        metrics.add_result(passed=False, score=0.2)
        metrics.add_result(passed=True, score=0.9)

        assert metrics.total_tests == 4
        assert metrics.passed == 3
        assert metrics.failed == 1
        assert metrics.accuracy == 0.75

    def test_average_score(self) -> None:
        """Test average score calculation."""
        metrics = QualityMetrics()
        metrics.add_result(passed=True, score=1.0)
        metrics.add_result(passed=True, score=0.5)

        assert metrics.average_score == 0.75

    def test_meets_threshold(self) -> None:
        """Test threshold check."""
        metrics = QualityMetrics()
        metrics.add_result(passed=True, score=1.0)
        metrics.add_result(passed=True, score=1.0)
        metrics.add_result(passed=False, score=0.0)

        # 66.7% accuracy
        assert metrics.meets_threshold(0.5) is True
        assert metrics.meets_threshold(0.7) is False
        assert metrics.meets_threshold(0.85) is False

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        metrics = QualityMetrics()
        metrics.add_result(passed=True, score=1.0)

        result = metrics.to_dict()
        assert result["total_tests"] == 1
        assert result["passed"] == 1
        assert result["accuracy"] == 1.0


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""

    def test_empty_metrics(self) -> None:
        """Test empty metrics initialization."""
        metrics = PerformanceMetrics()
        assert metrics.request_count == 0
        assert metrics.p50_ms == 0.0
        assert metrics.p99_ms == 0.0
        assert metrics.error_rate == 0.0

    def test_add_latency(self) -> None:
        """Test adding latency measurements."""
        metrics = PerformanceMetrics()
        metrics.add_latency(100.0)
        metrics.add_latency(200.0)
        metrics.add_latency(300.0)

        assert metrics.request_count == 3
        assert metrics.min_ms == 100.0
        assert metrics.max_ms == 300.0
        assert metrics.mean_ms == 200.0

    def test_percentiles(self) -> None:
        """Test percentile calculations."""
        metrics = PerformanceMetrics()

        # Add 100 latencies from 1 to 100
        for i in range(1, 101):
            metrics.add_latency(float(i))

        assert metrics.p50_ms == 50.5  # Median of 1-100
        assert metrics.p90_ms == 91.0  # Index 90 (0-indexed) = 91
        assert metrics.p99_ms == 100.0  # Index 99 (0-indexed) = 100

    def test_error_rate(self) -> None:
        """Test error rate calculation."""
        metrics = PerformanceMetrics()
        metrics.add_latency(100.0)
        metrics.add_latency(100.0)
        metrics.add_error()
        metrics.add_timeout()

        assert metrics.request_count == 4
        assert metrics.error_count == 1
        assert metrics.timeout_count == 1
        assert metrics.error_rate == 0.5

    def test_meets_sla_pass(self) -> None:
        """Test SLA compliance when all requirements met."""
        metrics = PerformanceMetrics()

        # Add low latencies
        for _ in range(100):
            metrics.add_latency(500.0)

        meets_sla, violations = metrics.meets_sla(
            p50_threshold_ms=2000.0,
            p99_threshold_ms=10000.0,
            error_rate_threshold=0.05,
        )

        assert meets_sla is True
        assert len(violations) == 0

    def test_meets_sla_fail_p50(self) -> None:
        """Test SLA compliance failure on P50."""
        metrics = PerformanceMetrics()

        # Add high latencies
        for _ in range(100):
            metrics.add_latency(3000.0)

        meets_sla, violations = metrics.meets_sla(
            p50_threshold_ms=2000.0,
            p99_threshold_ms=10000.0,
        )

        assert meets_sla is False
        assert any("P50" in v for v in violations)

    def test_meets_sla_fail_error_rate(self) -> None:
        """Test SLA compliance failure on error rate."""
        metrics = PerformanceMetrics()

        # 10% error rate
        for _ in range(90):
            metrics.add_latency(100.0)
        for _ in range(10):
            metrics.add_error()

        meets_sla, violations = metrics.meets_sla(
            error_rate_threshold=0.05,
        )

        assert meets_sla is False
        assert any("Error rate" in v for v in violations)

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        metrics = PerformanceMetrics()
        metrics.add_latency(100.0)
        metrics.add_latency(200.0)
        metrics.add_error()

        result = metrics.to_dict()
        assert result["request_count"] == 3
        assert result["error_count"] == 1
        assert "latency" in result
        assert result["latency"]["min_ms"] == 100.0
        assert result["latency"]["max_ms"] == 200.0

    def test_throughput(self) -> None:
        """Test throughput calculation."""
        metrics = PerformanceMetrics()
        metrics.add_latency(100.0)
        metrics.add_latency(100.0)
        metrics.add_error()

        # Throughput is successful requests
        assert metrics.throughput == 2
