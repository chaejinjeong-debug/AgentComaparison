"""Evaluation framework for agent quality and performance testing.

Provides the main evaluation orchestration logic.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from agent_engine.evaluation.metrics import PerformanceMetrics, QualityMetrics
from agent_engine.evaluation.test_cases import TestCase

logger = structlog.get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result of a single test case evaluation.

    Attributes:
        test_case_id: ID of the test case
        category: Test category
        passed: Whether the test passed
        score: Score for this test
        reason: Pass/fail reason
        response: Agent response
        latency_ms: Response latency in milliseconds
        error: Error message if any
    """

    test_case_id: str
    category: str
    passed: bool
    score: float
    reason: str
    response: str | None = None
    latency_ms: float | None = None
    error: str | None = None


@dataclass
class EvaluationSummary:
    """Summary of evaluation run.

    Attributes:
        started_at: Evaluation start time
        completed_at: Evaluation completion time
        total_tests: Total number of tests
        quality_metrics: Quality metrics
        performance_metrics: Performance metrics
        results: Individual test results
        passed_threshold: Whether quality threshold was met
    """

    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    total_tests: int = 0
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    results: list[EvaluationResult] = field(default_factory=list)
    passed_threshold: bool = False
    threshold: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_tests": self.total_tests,
            "passed_threshold": self.passed_threshold,
            "threshold": self.threshold,
            "quality": self.quality_metrics.to_dict(),
            "performance": self.performance_metrics.to_dict(),
            "results_summary": {
                "by_category": self._results_by_category(),
            },
        }

    def _results_by_category(self) -> dict[str, dict[str, int]]:
        """Group results by category."""
        categories: dict[str, dict[str, int]] = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {"passed": 0, "failed": 0}
            if result.passed:
                categories[result.category]["passed"] += 1
            else:
                categories[result.category]["failed"] += 1
        return categories


class EvaluationFramework:
    """Framework for evaluating agent quality and performance.

    This class orchestrates the evaluation process, running test cases
    against an agent and collecting metrics.

    Attributes:
        agent_query_func: Function to call the agent
        threshold: Quality threshold (default: 0.85)
        timeout_seconds: Default timeout per test
    """

    def __init__(
        self,
        agent_query_func: Callable[[str], str] | None = None,
        async_query_func: Callable[[str], Any] | None = None,
        threshold: float = 0.85,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize the evaluation framework.

        Args:
            agent_query_func: Synchronous function to query the agent
            async_query_func: Async function to query the agent
            threshold: Quality threshold (0.0 - 1.0)
            timeout_seconds: Default timeout per test
        """
        self.agent_query_func = agent_query_func
        self.async_query_func = async_query_func
        self.threshold = threshold
        self.timeout_seconds = timeout_seconds

    def evaluate(
        self,
        test_cases: list[TestCase],
        categories: list[str] | None = None,
    ) -> EvaluationSummary:
        """Run synchronous evaluation on test cases.

        Args:
            test_cases: List of test cases to evaluate
            categories: Filter by categories (optional)

        Returns:
            EvaluationSummary with results
        """
        if not self.agent_query_func:
            raise ValueError("agent_query_func must be provided for sync evaluation")

        # Filter by category if specified
        if categories:
            test_cases = [tc for tc in test_cases if tc.category in categories]

        summary = EvaluationSummary(threshold=self.threshold)
        summary.total_tests = len(test_cases)

        for test_case in test_cases:
            result = self._evaluate_single(test_case)
            summary.results.append(result)

            # Update metrics
            summary.quality_metrics.add_result(
                passed=result.passed,
                score=result.score,
                reason=result.reason if not result.passed else "",
            )

            if result.latency_ms is not None:
                summary.performance_metrics.add_latency(result.latency_ms)
            elif result.error:
                summary.performance_metrics.add_error()

        summary.completed_at = datetime.now(UTC)
        summary.passed_threshold = summary.quality_metrics.meets_threshold(self.threshold)

        return summary

    async def evaluate_async(
        self,
        test_cases: list[TestCase],
        categories: list[str] | None = None,
        concurrency: int = 5,
    ) -> EvaluationSummary:
        """Run async evaluation on test cases.

        Args:
            test_cases: List of test cases to evaluate
            categories: Filter by categories (optional)
            concurrency: Maximum concurrent evaluations

        Returns:
            EvaluationSummary with results
        """
        if not self.async_query_func:
            raise ValueError("async_query_func must be provided for async evaluation")

        # Filter by category if specified
        if categories:
            test_cases = [tc for tc in test_cases if tc.category in categories]

        summary = EvaluationSummary(threshold=self.threshold)
        summary.total_tests = len(test_cases)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)

        async def evaluate_with_semaphore(tc: TestCase) -> EvaluationResult:
            async with semaphore:
                return await self._evaluate_single_async(tc)

        # Run evaluations concurrently
        results = await asyncio.gather(
            *[evaluate_with_semaphore(tc) for tc in test_cases],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                # Handle exceptions
                error_result = EvaluationResult(
                    test_case_id="unknown",
                    category="unknown",
                    passed=False,
                    score=0.0,
                    reason=str(result),
                    error=str(result),
                )
                summary.results.append(error_result)
                summary.quality_metrics.add_result(False, 0.0, str(result))
                summary.performance_metrics.add_error()
            else:
                summary.results.append(result)
                summary.quality_metrics.add_result(
                    passed=result.passed,
                    score=result.score,
                    reason=result.reason if not result.passed else "",
                )
                if result.latency_ms is not None:
                    summary.performance_metrics.add_latency(result.latency_ms)
                elif result.error:
                    summary.performance_metrics.add_error()

        summary.completed_at = datetime.now(UTC)
        summary.passed_threshold = summary.quality_metrics.meets_threshold(self.threshold)

        return summary

    def _evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case synchronously.

        Args:
            test_case: Test case to evaluate

        Returns:
            EvaluationResult
        """
        logger.debug("evaluating_test_case", test_case_id=test_case.id)

        try:
            start_time = time.perf_counter()
            response = self.agent_query_func(test_case.input)  # type: ignore
            latency_ms = (time.perf_counter() - start_time) * 1000

            passed, score, reason = test_case.validate_response(response)

            return EvaluationResult(
                test_case_id=test_case.id,
                category=test_case.category,
                passed=passed,
                score=score,
                reason=reason,
                response=response,
                latency_ms=latency_ms,
            )

        except TimeoutError:
            return EvaluationResult(
                test_case_id=test_case.id,
                category=test_case.category,
                passed=False,
                score=0.0,
                reason="Timeout",
                error="Request timed out",
            )

        except Exception as e:
            logger.exception("evaluation_error", test_case_id=test_case.id, error=str(e))
            return EvaluationResult(
                test_case_id=test_case.id,
                category=test_case.category,
                passed=False,
                score=0.0,
                reason=f"Error: {e}",
                error=str(e),
            )

    async def _evaluate_single_async(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case asynchronously.

        Args:
            test_case: Test case to evaluate

        Returns:
            EvaluationResult
        """
        logger.debug("evaluating_test_case_async", test_case_id=test_case.id)

        try:
            start_time = time.perf_counter()
            response = await asyncio.wait_for(
                self.async_query_func(test_case.input),  # type: ignore
                timeout=test_case.timeout_seconds or self.timeout_seconds,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000

            passed, score, reason = test_case.validate_response(str(response))

            return EvaluationResult(
                test_case_id=test_case.id,
                category=test_case.category,
                passed=passed,
                score=score,
                reason=reason,
                response=str(response),
                latency_ms=latency_ms,
            )

        except TimeoutError:
            return EvaluationResult(
                test_case_id=test_case.id,
                category=test_case.category,
                passed=False,
                score=0.0,
                reason="Timeout",
                error="Request timed out",
            )

        except Exception as e:
            logger.exception("evaluation_error_async", test_case_id=test_case.id, error=str(e))
            return EvaluationResult(
                test_case_id=test_case.id,
                category=test_case.category,
                passed=False,
                score=0.0,
                reason=f"Error: {e}",
                error=str(e),
            )
