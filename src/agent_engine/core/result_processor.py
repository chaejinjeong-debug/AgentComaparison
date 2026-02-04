"""Result Processor for handling agent execution results.

This module provides utilities for processing agent results
into standardized response dictionaries.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agent_engine.core.base_agent import AgentResult


class ResultProcessor:
    """Utility class for processing agent execution results.

    This class handles the conversion of AgentResult objects into
    standardized response dictionaries suitable for API responses.

    The response format includes:
    - response: Agent response text
    - tool_calls: List of executed tools
    - usage: Token usage information
    - metadata: Query metadata (model, timestamps, latency, etc.)

    Example:
        >>> processor = ResultProcessor(model_name="gemini-2.5-pro")
        >>> response = processor.process(
        ...     result=agent_result,
        ...     user_id="user123",
        ...     session_id="sess456",
        ...     start_time=start,
        ... )
        >>> print(response["response"])
        "Hello! How can I help you today?"
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the result processor.

        Args:
            model_name: Model identifier for metadata
        """
        self.model_name = model_name

    def process(
        self,
        result: AgentResult,
        user_id: str | None = None,
        session_id: str | None = None,
        start_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Process an AgentResult into a response dictionary.

        Args:
            result: AgentResult from agent execution
            user_id: User identifier
            session_id: Session identifier
            start_time: Query start time for latency calculation

        Returns:
            Standardized response dictionary
        """
        now = datetime.now(UTC)
        latency_ms = 0.0
        if start_time:
            latency_ms = (now - start_time).total_seconds() * 1000

        return {
            "response": result.response_text,
            "tool_calls": result.tool_calls,
            "usage": result.usage,
            "metadata": {
                "model": self.model_name,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": now.isoformat(),
                "latency_ms": latency_ms,
                **result.metadata,
            },
        }

    def process_raw(
        self,
        result: Any,
        user_id: str | None = None,
        session_id: str | None = None,
        start_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Process a raw model result into a response dictionary.

        This method extracts data from model-specific result objects
        and creates a standardized response. Use this when you have
        a raw result from the model rather than an AgentResult.

        Args:
            result: Raw model result object
            user_id: User identifier
            session_id: Session identifier
            start_time: Query start time for latency calculation

        Returns:
            Standardized response dictionary
        """
        # Extract response text
        response_text = self._extract_response_text(result)

        # Extract tool calls
        tool_calls = self._extract_tool_calls(result)

        # Extract usage
        usage = self._extract_usage(result)

        now = datetime.now(UTC)
        latency_ms = 0.0
        if start_time:
            latency_ms = (now - start_time).total_seconds() * 1000

        return {
            "response": response_text,
            "tool_calls": tool_calls,
            "usage": usage,
            "metadata": {
                "model": self.model_name,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": now.isoformat(),
                "latency_ms": latency_ms,
            },
        }

    def _extract_response_text(self, result: Any) -> str:
        """Extract text from a model-specific result object.

        Args:
            result: Model-specific result object

        Returns:
            Extracted text string
        """
        if hasattr(result, "output"):
            return str(result.output)
        if hasattr(result, "text"):
            return str(result.text)
        if hasattr(result, "content"):
            return str(result.content)
        return str(result)

    def _extract_tool_calls(self, result: Any) -> list[dict[str, Any]]:
        """Extract tool calls from a model-specific result object.

        Args:
            result: Model-specific result object

        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        if hasattr(result, "tool_calls"):
            for tc in result.tool_calls:
                tool_calls.append(
                    {
                        "tool": getattr(tc, "name", "unknown"),
                        "args": getattr(tc, "args", {}),
                    }
                )
        return tool_calls

    def _extract_usage(self, result: Any) -> dict[str, int]:
        """Extract token usage from a model-specific result object.

        Args:
            result: Model-specific result object

        Returns:
            Dictionary with token usage information
        """
        if hasattr(result, "usage"):
            return {
                "prompt_tokens": getattr(result.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(result.usage, "completion_tokens", 0),
                "total_tokens": getattr(result.usage, "total_tokens", 0),
            }
        return {}

    @staticmethod
    def create_stream_metadata(
        model_name: str,
        user_id: str | None,
        session_id: str | None,
        latency_ms: float,
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create metadata dictionary for streaming responses.

        Args:
            model_name: Model identifier
            user_id: User identifier
            session_id: Session identifier
            latency_ms: Total latency in milliseconds
            tool_calls: List of tool calls made during streaming

        Returns:
            Metadata dictionary
        """
        return {
            "model": model_name,
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "latency_ms": latency_ms,
            "tool_calls": tool_calls,
        }
