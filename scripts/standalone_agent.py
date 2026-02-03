#!/usr/bin/env python3
"""Standalone agent for deployment to VertexAI Reasoning Engine.

This module provides a standalone agent class that can be pickled and
deployed without external module dependencies.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable, Sequence


class StandaloneAgent:
    """Standalone agent for VertexAI Reasoning Engine deployment.

    This class is designed to be self-contained and pickled without
    external module dependencies.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        project: str = "",
        location: str = "us-central1",
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize the standalone agent.

        Args:
            model: Gemini model name
            project: GCP project ID
            location: GCP region
            system_prompt: System prompt for the agent
            temperature: Model temperature (0.0-2.0)
            max_tokens: Maximum output tokens
        """
        self.model_name = model
        self.project = project
        self.location = location
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._agent: Any = None
        self._is_setup = False

    def set_up(self) -> None:
        """Initialize the Pydantic AI Agent and connect to VertexAI."""
        import vertexai
        from pydantic_ai import Agent
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        # Initialize VertexAI
        vertexai.init(project=self.project, location=self.location)

        # Create GoogleProvider with VertexAI
        provider = GoogleProvider(
            vertexai=True,
            project=self.project,
            location=self.location,
        )

        # Create GoogleModel
        google_model = GoogleModel(
            self.model_name,
            provider=provider,
        )

        # Define tools inline to avoid serialization issues
        def get_current_datetime(timezone: str = "UTC") -> str:
            """Get the current date and time.

            Args:
                timezone: Timezone name (e.g., 'UTC', 'Asia/Seoul')

            Returns:
                Current datetime string
            """
            from zoneinfo import ZoneInfo

            try:
                tz = ZoneInfo(timezone)
            except Exception:
                tz = ZoneInfo("UTC")

            now = datetime.now(tz)
            return now.strftime("%Y-%m-%d %H:%M:%S %Z")

        def calculate(expression: str) -> str:
            """Evaluate a mathematical expression.

            Args:
                expression: Mathematical expression to evaluate

            Returns:
                Result of the calculation
            """
            import ast
            import operator

            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.Mod: operator.mod,
            }

            def eval_node(node: ast.AST) -> float:
                if isinstance(node, ast.Constant):
                    return float(node.value)
                elif isinstance(node, ast.BinOp):
                    left = eval_node(node.left)
                    right = eval_node(node.right)
                    return ops[type(node.op)](left, right)
                elif isinstance(node, ast.UnaryOp):
                    if isinstance(node.op, ast.USub):
                        return -eval_node(node.operand)
                    elif isinstance(node.op, ast.UAdd):
                        return eval_node(node.operand)
                raise ValueError(f"Unsupported: {type(node)}")

            try:
                tree = ast.parse(expression, mode="eval")
                result = eval_node(tree.body)
                return str(result) if result == int(result) else f"{result:.6f}"
            except Exception as e:
                return f"Error: {e}"

        def search(query: str) -> str:
            """Search for information (mock implementation).

            Args:
                query: Search query

            Returns:
                Mock search results
            """
            return f"Mock search results for: '{query}'. This is a placeholder implementation."

        # Create Pydantic AI Agent with inline tools
        self._agent = Agent(
            model=google_model,
            system_prompt=self.system_prompt,
            tools=[get_current_datetime, calculate, search],
        )

        self._is_setup = True

    def query(
        self,
        message: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a query against the agent.

        Args:
            message: User message to process
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary containing response
        """
        if not self._is_setup or self._agent is None:
            self.set_up()

        start_time = datetime.now(UTC)

        try:
            # Run the agent synchronously
            result = self._agent.run_sync(message)

            # Extract response text
            response_text = str(result.output) if hasattr(result, "output") else str(result)

            return {
                "response": response_text,
                "metadata": {
                    "model": self.model_name,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "latency_ms": (datetime.now(UTC) - start_time).total_seconds() * 1000,
                },
            }

        except Exception as e:
            return {
                "response": f"Error processing query: {e}",
                "metadata": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            }
