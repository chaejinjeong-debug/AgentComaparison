"""Tests for the built-in tools module."""

import sys
from pathlib import Path

import pytest


from agent_engine.exceptions import ToolExecutionError
from agent_engine.tools import calculate, convert_timezone, get_current_datetime, search
from agent_engine.tools.calculator import _safe_eval, _tokenize


class TestSearchTool:
    """Test suite for the search tool."""

    def test_search_basic(self) -> None:
        """Test basic search functionality."""
        result = search("test query")

        assert "query" in result
        assert "results" in result
        assert "count" in result
        assert result["query"] == "test query"
        assert isinstance(result["results"], list)

    def test_search_with_keyword_match(self) -> None:
        """Test search with keyword match."""
        result = search("weather forecast")

        assert result["count"] > 0
        assert any("weather" in r["title"].lower() for r in result["results"])

    def test_search_max_results(self) -> None:
        """Test search with max_results limit."""
        result = search("python programming", max_results=1)

        assert result["count"] <= 1

    def test_search_no_match(self) -> None:
        """Test search with no keyword match returns default."""
        result = search("xyz123abc")

        assert result["count"] > 0  # Should return default results
        assert isinstance(result["results"], list)


class TestCalculateTool:
    """Test suite for the calculate tool."""

    def test_calculate_addition(self) -> None:
        """Test basic addition."""
        result = calculate("2 + 3")

        assert result["result"] == 5
        assert result["type"] == "int"

    def test_calculate_subtraction(self) -> None:
        """Test subtraction."""
        result = calculate("10 - 4")

        assert result["result"] == 6

    def test_calculate_multiplication(self) -> None:
        """Test multiplication."""
        result = calculate("6 * 7")

        assert result["result"] == 42

    def test_calculate_division(self) -> None:
        """Test division."""
        result = calculate("20 / 4")

        assert result["result"] == 5

    def test_calculate_float_division(self) -> None:
        """Test float division."""
        result = calculate("7 / 2")

        assert result["result"] == 3.5
        assert result["type"] == "float"

    def test_calculate_power(self) -> None:
        """Test power operation."""
        result = calculate("2 ** 3")

        assert result["result"] == 8

    def test_calculate_power_caret(self) -> None:
        """Test power with caret operator."""
        result = calculate("2 ^ 4")

        assert result["result"] == 16

    def test_calculate_parentheses(self) -> None:
        """Test parentheses."""
        result = calculate("(2 + 3) * 4")

        assert result["result"] == 20

    def test_calculate_nested_parentheses(self) -> None:
        """Test nested parentheses."""
        result = calculate("((2 + 3) * (4 - 1))")

        assert result["result"] == 15

    def test_calculate_sqrt(self) -> None:
        """Test square root function."""
        result = calculate("sqrt(16)")

        assert result["result"] == 4

    def test_calculate_pi(self) -> None:
        """Test pi constant."""
        result = calculate("pi")

        assert abs(result["result"] - 3.14159) < 0.001

    def test_calculate_complex_expression(self) -> None:
        """Test complex expression."""
        result = calculate("sqrt(16) + 2 * 3")

        assert result["result"] == 10

    def test_calculate_division_by_zero(self) -> None:
        """Test division by zero raises error."""
        with pytest.raises(ToolExecutionError) as exc_info:
            calculate("1 / 0")

        assert "Division by zero" in str(exc_info.value)
        assert exc_info.value.tool_name == "calculate"

    def test_calculate_invalid_expression(self) -> None:
        """Test invalid expression raises error."""
        with pytest.raises(ToolExecutionError):
            calculate("2 + + 3")


class TestSafeEval:
    """Test suite for safe evaluation internals."""

    def test_tokenize_simple(self) -> None:
        """Test tokenizing simple expression."""
        tokens = _tokenize("2 + 3")

        assert tokens == [2.0, "+", 3.0]

    def test_tokenize_power(self) -> None:
        """Test tokenizing power operator."""
        tokens = _tokenize("2 ** 3")

        assert "**" in tokens

    def test_tokenize_floor_division(self) -> None:
        """Test tokenizing floor division."""
        tokens = _tokenize("7 // 2")

        assert "//" in tokens

    def test_safe_eval_constants(self) -> None:
        """Test constant replacement."""
        result = _safe_eval("pi")

        assert abs(result - 3.14159) < 0.001


class TestDateTimeTool:
    """Test suite for the datetime tool."""

    def test_get_current_datetime_utc(self) -> None:
        """Test getting current datetime in UTC."""
        result = get_current_datetime("UTC")

        assert "datetime" in result
        assert "timestamp" in result
        assert "timezone" in result
        assert "iso_format" in result
        assert "components" in result
        assert result["timezone"] == "UTC"

    def test_get_current_datetime_seoul(self) -> None:
        """Test getting current datetime in Seoul."""
        result = get_current_datetime("Asia/Seoul")

        assert result["timezone"] == "Asia/Seoul"

    def test_get_current_datetime_alias(self) -> None:
        """Test timezone alias resolution."""
        result = get_current_datetime("KST")

        assert result["timezone"] == "Asia/Seoul"

    def test_get_current_datetime_components(self) -> None:
        """Test datetime components."""
        result = get_current_datetime("UTC")

        components = result["components"]
        assert "year" in components
        assert "month" in components
        assert "day" in components
        assert "hour" in components
        assert "minute" in components
        assert "second" in components
        assert "weekday" in components

    def test_get_current_datetime_custom_format(self) -> None:
        """Test custom datetime format."""
        result = get_current_datetime("UTC", format_str="%Y-%m-%d")

        # Should be in YYYY-MM-DD format
        assert len(result["datetime"]) == 10

    def test_get_current_datetime_invalid_timezone(self) -> None:
        """Test invalid timezone raises error."""
        with pytest.raises(ToolExecutionError) as exc_info:
            get_current_datetime("Invalid/Timezone")

        assert "Unknown timezone" in str(exc_info.value)


class TestConvertTimezoneTool:
    """Test suite for the timezone conversion tool."""

    def test_convert_timezone_basic(self) -> None:
        """Test basic timezone conversion."""
        result = convert_timezone(
            datetime_str="2026-02-02 09:00:00",
            from_timezone="UTC",
            to_timezone="Asia/Seoul",
        )

        assert "original" in result
        assert "converted" in result
        assert "from_timezone" in result
        assert "to_timezone" in result
        assert result["from_timezone"] == "UTC"
        assert result["to_timezone"] == "Asia/Seoul"

    def test_convert_timezone_with_aliases(self) -> None:
        """Test timezone conversion with aliases."""
        result = convert_timezone(
            datetime_str="2026-02-02 09:00:00",
            from_timezone="UTC",
            to_timezone="KST",
        )

        assert result["to_timezone"] == "Asia/Seoul"

    def test_convert_timezone_utc_to_seoul(self) -> None:
        """Test UTC to Seoul conversion (9 hours ahead)."""
        result = convert_timezone(
            datetime_str="2026-02-02 00:00:00",
            from_timezone="UTC",
            to_timezone="Asia/Seoul",
        )

        # Seoul is UTC+9
        assert "09:00:00" in result["converted"]

    def test_convert_timezone_custom_format(self) -> None:
        """Test custom input/output formats."""
        result = convert_timezone(
            datetime_str="02/02/2026 09:00",
            from_timezone="UTC",
            to_timezone="Asia/Seoul",
            input_format="%m/%d/%Y %H:%M",
            output_format="%Y-%m-%d %H:%M",
        )

        assert "2026-02-02" in result["converted"]

    def test_convert_timezone_invalid_format(self) -> None:
        """Test invalid datetime format raises error."""
        with pytest.raises(ToolExecutionError) as exc_info:
            convert_timezone(
                datetime_str="invalid-date",
                from_timezone="UTC",
                to_timezone="Asia/Seoul",
            )

        assert "Invalid datetime format" in str(exc_info.value)

    def test_convert_timezone_invalid_timezone(self) -> None:
        """Test invalid timezone raises error."""
        with pytest.raises(ToolExecutionError) as exc_info:
            convert_timezone(
                datetime_str="2026-02-02 09:00:00",
                from_timezone="Invalid/TZ",
                to_timezone="Asia/Seoul",
            )

        assert "Unknown timezone" in str(exc_info.value)


class TestDefaultTools:
    """Test suite for default tools collection."""

    def test_default_tools_imported(self) -> None:
        """Test that default tools are properly imported."""
        from agent_engine.tools import DEFAULT_TOOLS

        assert len(DEFAULT_TOOLS) == 4
        assert search in DEFAULT_TOOLS
        assert calculate in DEFAULT_TOOLS
        assert get_current_datetime in DEFAULT_TOOLS
        assert convert_timezone in DEFAULT_TOOLS
