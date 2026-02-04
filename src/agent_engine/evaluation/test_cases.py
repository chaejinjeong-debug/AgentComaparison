"""Test case models and loaders for agent evaluation.

Provides structures for defining test cases and loading them from files.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single test case for agent evaluation.

    Attributes:
        id: Unique identifier for the test case
        category: Category/type of test (e.g., "qa", "math", "tool_use")
        input: Input query to send to the agent
        expected_output: Expected output (exact match or pattern)
        expected_keywords: Keywords that should appear in response
        forbidden_keywords: Keywords that should not appear
        metadata: Additional metadata for the test
    """

    id: str = Field(..., description="Unique test case ID")
    category: str = Field(default="general", description="Test category")
    input: str = Field(..., description="Input query")
    expected_output: str | None = Field(default=None, description="Expected exact output")
    expected_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that must appear in response",
    )
    forbidden_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that must not appear in response",
    )
    expected_tool_calls: list[str] = Field(
        default_factory=list,
        description="Expected tool calls",
    )
    timeout_seconds: float = Field(default=30.0, description="Timeout for this test")
    weight: float = Field(default=1.0, description="Weight for scoring")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def validate_response(self, response: str) -> tuple[bool, float, str]:
        """Validate an agent response against this test case.

        Args:
            response: Agent response to validate

        Returns:
            Tuple of (passed, score, reason)
        """
        response_lower = response.lower()
        score = 0.0
        reasons = []
        has_criteria = False

        # Check exact match if specified
        if self.expected_output is not None:
            has_criteria = True
            if self.expected_output.lower() in response_lower:
                score += 0.5
            else:
                reasons.append(f"Expected output not found: {self.expected_output[:50]}...")

        # Check required keywords
        if self.expected_keywords:
            has_criteria = True
            found_keywords = sum(1 for kw in self.expected_keywords if kw.lower() in response_lower)
            keyword_ratio = found_keywords / len(self.expected_keywords)
            # Score 0.6 for keywords when no expected_output, 0.3 when both exist
            keyword_score = 0.6 if self.expected_output is None else 0.3
            score += keyword_score * keyword_ratio

            if keyword_ratio < 1.0:
                missing = [kw for kw in self.expected_keywords if kw.lower() not in response_lower]
                reasons.append(f"Missing keywords: {missing}")

        # Check forbidden keywords
        if self.forbidden_keywords:
            found_forbidden = [kw for kw in self.forbidden_keywords if kw.lower() in response_lower]
            if found_forbidden:
                score -= 0.2
                reasons.append(f"Forbidden keywords found: {found_forbidden}")

        # Base score for non-empty response
        if response.strip() and not has_criteria:
            score = 0.7  # Default passing score if no specific criteria

        # Apply weight
        score = min(1.0, max(0.0, score)) * self.weight

        # Determine pass/fail
        passed = score >= 0.5 * self.weight and not reasons

        return passed, score, "; ".join(reasons) if reasons else "OK"


class TestCaseLoader:
    """Loader for test cases from various sources."""

    @staticmethod
    def load_from_json(path: str | Path) -> list[TestCase]:
        """Load test cases from a JSON file.

        Args:
            path: Path to JSON file

        Returns:
            List of TestCase objects
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        test_cases = []
        if isinstance(data, list):
            for item in data:
                test_cases.append(TestCase(**item))
        elif isinstance(data, dict) and "test_cases" in data:
            for item in data["test_cases"]:
                test_cases.append(TestCase(**item))

        return test_cases

    @staticmethod
    def load_from_dict(data: list[dict[str, Any]]) -> list[TestCase]:
        """Load test cases from a list of dictionaries.

        Args:
            data: List of test case dictionaries

        Returns:
            List of TestCase objects
        """
        return [TestCase(**item) for item in data]

    @staticmethod
    def filter_by_category(
        test_cases: list[TestCase],
        categories: list[str],
    ) -> list[TestCase]:
        """Filter test cases by category.

        Args:
            test_cases: List of test cases
            categories: Categories to include

        Returns:
            Filtered list of test cases
        """
        return [tc for tc in test_cases if tc.category in categories]

    @staticmethod
    def get_categories(test_cases: list[TestCase]) -> set[str]:
        """Get all unique categories in test cases.

        Args:
            test_cases: List of test cases

        Returns:
            Set of category names
        """
        return {tc.category for tc in test_cases}
