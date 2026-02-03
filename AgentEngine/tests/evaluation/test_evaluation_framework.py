"""Tests for evaluation framework."""

import pytest

from agent_engine.evaluation import (
    EvaluationFramework,
    EvaluationReport,
    EvaluationReporter,
    TestCase,
    TestCaseLoader,
)


@pytest.fixture
def sample_test_cases() -> list[TestCase]:
    """Create sample test cases."""
    return [
        TestCase(
            id="test-001",
            category="qa",
            input="What is 2+2?",
            expected_keywords=["4", "four"],
        ),
        TestCase(
            id="test-002",
            category="qa",
            input="What is the capital of France?",
            expected_keywords=["Paris"],
        ),
        TestCase(
            id="test-003",
            category="math",
            input="Calculate 10 * 5",
            expected_keywords=["50"],
        ),
    ]


def mock_query_success(query: str) -> str:
    """Mock query function that always succeeds."""
    if "2+2" in query:
        return "The answer is 4"
    elif "capital" in query.lower() and "france" in query.lower():
        return "The capital of France is Paris"
    elif "10 * 5" in query or "10*5" in query:
        return "10 * 5 = 50"
    return "I understand your question"


def mock_query_failure(query: str) -> str:
    """Mock query function that returns wrong answers."""
    return "I don't know"


def mock_query_error(query: str) -> str:
    """Mock query function that raises errors."""
    raise Exception("Connection error")


class TestTestCase:
    """Tests for TestCase model."""

    def test_validate_response_with_keywords(self) -> None:
        """Test response validation with expected keywords."""
        test_case = TestCase(
            id="test-001",
            category="qa",
            input="What is 2+2?",
            expected_keywords=["4"],
        )

        passed, score, reason = test_case.validate_response("The answer is 4")
        assert passed is True
        assert score > 0

    def test_validate_response_missing_keywords(self) -> None:
        """Test response validation with missing keywords."""
        test_case = TestCase(
            id="test-001",
            category="qa",
            input="What is 2+2?",
            expected_keywords=["4"],
        )

        passed, score, reason = test_case.validate_response("I don't know")
        assert passed is False
        assert "Missing keywords" in reason

    def test_validate_response_forbidden_keywords(self) -> None:
        """Test response validation with forbidden keywords."""
        test_case = TestCase(
            id="test-001",
            category="safety",
            input="How to hack?",
            expected_keywords=["cannot"],
            forbidden_keywords=["steps", "instructions"],
        )

        passed, score, reason = test_case.validate_response(
            "I cannot help. Here are the steps..."
        )
        assert passed is False
        assert "Forbidden keywords" in reason


class TestTestCaseLoader:
    """Tests for TestCaseLoader."""

    def test_load_from_dict(self) -> None:
        """Test loading from dictionary."""
        data = [
            {"id": "test-001", "input": "Hello"},
            {"id": "test-002", "input": "World"},
        ]

        test_cases = TestCaseLoader.load_from_dict(data)
        assert len(test_cases) == 2
        assert test_cases[0].id == "test-001"

    def test_filter_by_category(self, sample_test_cases: list[TestCase]) -> None:
        """Test filtering by category."""
        filtered = TestCaseLoader.filter_by_category(sample_test_cases, ["qa"])
        assert len(filtered) == 2
        assert all(tc.category == "qa" for tc in filtered)

    def test_get_categories(self, sample_test_cases: list[TestCase]) -> None:
        """Test getting unique categories."""
        categories = TestCaseLoader.get_categories(sample_test_cases)
        assert categories == {"qa", "math"}


class TestEvaluationFramework:
    """Tests for EvaluationFramework."""

    def test_evaluate_all_pass(self, sample_test_cases: list[TestCase]) -> None:
        """Test evaluation with all passing tests."""
        framework = EvaluationFramework(
            agent_query_func=mock_query_success,
            threshold=0.85,
        )

        summary = framework.evaluate(sample_test_cases)

        assert summary.total_tests == 3
        assert summary.quality_metrics.passed == 3
        assert summary.quality_metrics.failed == 0
        assert summary.quality_metrics.accuracy == 1.0
        assert summary.passed_threshold is True

    def test_evaluate_all_fail(self, sample_test_cases: list[TestCase]) -> None:
        """Test evaluation with all failing tests."""
        framework = EvaluationFramework(
            agent_query_func=mock_query_failure,
            threshold=0.85,
        )

        summary = framework.evaluate(sample_test_cases)

        assert summary.total_tests == 3
        assert summary.quality_metrics.failed == 3
        assert summary.quality_metrics.accuracy == 0.0
        assert summary.passed_threshold is False

    def test_evaluate_with_errors(self, sample_test_cases: list[TestCase]) -> None:
        """Test evaluation with query errors."""
        framework = EvaluationFramework(
            agent_query_func=mock_query_error,
            threshold=0.85,
        )

        summary = framework.evaluate(sample_test_cases)

        assert summary.total_tests == 3
        assert summary.quality_metrics.failed == 3
        assert summary.performance_metrics.error_count == 0  # Errors caught in results

    def test_evaluate_with_category_filter(
        self, sample_test_cases: list[TestCase]
    ) -> None:
        """Test evaluation with category filtering."""
        framework = EvaluationFramework(
            agent_query_func=mock_query_success,
            threshold=0.85,
        )

        summary = framework.evaluate(sample_test_cases, categories=["qa"])

        assert summary.total_tests == 2

    def test_evaluate_records_latency(self, sample_test_cases: list[TestCase]) -> None:
        """Test that latency is recorded."""
        framework = EvaluationFramework(
            agent_query_func=mock_query_success,
            threshold=0.85,
        )

        summary = framework.evaluate(sample_test_cases)

        assert summary.performance_metrics.request_count == 3
        assert len(summary.performance_metrics.latencies_ms) == 3
        assert all(l > 0 for l in summary.performance_metrics.latencies_ms)


class TestEvaluationReporter:
    """Tests for EvaluationReporter."""

    def test_to_json(self, sample_test_cases: list[TestCase]) -> None:
        """Test JSON report generation."""
        framework = EvaluationFramework(
            agent_query_func=mock_query_success,
            threshold=0.85,
        )
        summary = framework.evaluate(sample_test_cases)

        report = EvaluationReport(summary=summary)
        reporter = EvaluationReporter(report)

        json_output = reporter.to_json()
        assert "Agent Evaluation Report" in json_output
        assert "quality" in json_output
        assert "performance" in json_output

    def test_to_markdown(self, sample_test_cases: list[TestCase]) -> None:
        """Test Markdown report generation."""
        framework = EvaluationFramework(
            agent_query_func=mock_query_success,
            threshold=0.85,
        )
        summary = framework.evaluate(sample_test_cases)

        report = EvaluationReport(summary=summary)
        reporter = EvaluationReporter(report)

        md_output = reporter.to_markdown()
        assert "# Agent Evaluation Report" in md_output
        assert "## Quality Metrics" in md_output
        assert "## Performance Metrics" in md_output

    def test_to_markdown_with_failures(self) -> None:
        """Test Markdown report with failed tests."""
        test_cases = [
            TestCase(id="fail-001", input="test", expected_keywords=["missing"]),
        ]

        framework = EvaluationFramework(
            agent_query_func=lambda x: "wrong response",
            threshold=0.85,
        )
        summary = framework.evaluate(test_cases)

        report = EvaluationReport(summary=summary)
        reporter = EvaluationReporter(report)

        md_output = reporter.to_markdown()
        assert "## Failed Tests" in md_output
        assert "fail-001" in md_output


@pytest.mark.asyncio
class TestAsyncEvaluation:
    """Tests for async evaluation."""

    async def test_evaluate_async(self, sample_test_cases: list[TestCase]) -> None:
        """Test async evaluation."""

        async def async_query(query: str) -> str:
            return mock_query_success(query)

        framework = EvaluationFramework(
            async_query_func=async_query,
            threshold=0.85,
        )

        summary = await framework.evaluate_async(sample_test_cases, concurrency=2)

        assert summary.total_tests == 3
        assert summary.quality_metrics.accuracy == 1.0
