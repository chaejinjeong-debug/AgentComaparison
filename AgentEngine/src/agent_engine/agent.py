"""Pydantic AI Agent Wrapper for VertexAI Agent Engine.

This module provides a wrapper class that integrates Pydantic AI with
VertexAI Agent Engine, conforming to the Agent Engine specification.

Requirements implemented:
- AC-001: Agent Engine specification compliance (__init__, set_up, query)
- AC-002: Pydantic AI Agent wrapping
- AC-003: GoogleProvider for Gemini model integration
- AC-004: Sync/Async query support
- AC-006: Error handling with graceful degradation
- TS-001: Tool registration mechanism
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import structlog

from agent_engine.config import AgentConfig
from agent_engine.exceptions import AgentConfigError, AgentQueryError, ToolExecutionError

logger = structlog.get_logger()


class PydanticAIAgentWrapper:
    """VertexAI Agent Engine compliant wrapper for Pydantic AI Agent.

    This class wraps a Pydantic AI Agent to conform to the VertexAI Agent Engine
    specification, providing the required __init__, set_up, and query methods.

    Attributes:
        model: Gemini model name (e.g., "gemini-2.5-pro")
        project: GCP project ID
        location: GCP region (e.g., "asia-northeast3")
        system_prompt: System prompt for the agent
        tools: Sequence of tool functions to register
        temperature: Model temperature setting
        max_tokens: Maximum output tokens

    Example:
        >>> agent = PydanticAIAgentWrapper(
        ...     model="gemini-2.5-pro",
        ...     project="my-project",
        ...     location="asia-northeast3",
        ...     system_prompt="You are a helpful assistant.",
        ...     tools=[search_tool, calculate_tool],
        ... )
        >>> agent.set_up()
        >>> response = agent.query(message="Hello!")
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        project: str = "",
        location: str = "asia-northeast3",
        system_prompt: str = "You are a helpful AI assistant.",
        tools: Sequence[Callable[..., Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize the agent wrapper with configuration parameters.

        Args:
            model: Gemini model name
            project: GCP project ID
            location: GCP region
            system_prompt: System prompt for the agent
            tools: Sequence of tool functions to register
            temperature: Model temperature (0.0-2.0)
            max_tokens: Maximum output tokens
        """
        self.model_name = model
        self.project = project
        self.location = location
        self.system_prompt = system_prompt
        self.tools = list(tools) if tools else []
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Will be initialized in set_up()
        self._agent: Any = None
        self._is_setup = False

        logger.info(
            "agent_wrapper_initialized",
            model=model,
            project=project,
            location=location,
            tool_count=len(self.tools),
        )

    @classmethod
    def from_config(
        cls, config: AgentConfig, tools: Sequence[Callable[..., Any]] | None = None
    ) -> PydanticAIAgentWrapper:
        """Create an agent wrapper from an AgentConfig instance.

        Args:
            config: AgentConfig instance
            tools: Optional sequence of tool functions

        Returns:
            PydanticAIAgentWrapper instance
        """
        return cls(
            model=config.model,
            project=config.project_id,
            location=config.location,
            system_prompt=config.system_prompt,
            tools=tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    def set_up(self) -> None:
        """Initialize the Pydantic AI Agent and connect to VertexAI.

        This method performs:
        - VertexAI initialization
        - GoogleProvider configuration
        - Pydantic AI Agent creation with tools

        Raises:
            AgentConfigError: If initialization fails
        """
        try:
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

            # Create Pydantic AI Agent
            self._agent = Agent(
                model=google_model,
                system_prompt=self.system_prompt,
                tools=self.tools,
            )

            self._is_setup = True

            logger.info(
                "agent_setup_complete",
                model=self.model_name,
                tool_count=len(self.tools),
            )

        except ImportError as e:
            raise AgentConfigError(
                f"Required package not installed: {e}",
                details={"missing_package": str(e)},
            ) from e
        except Exception as e:
            raise AgentConfigError(
                f"Failed to set up agent: {e}",
                details={"error_type": type(e).__name__},
            ) from e

    def _ensure_setup(self) -> None:
        """Ensure the agent has been set up."""
        if not self._is_setup or self._agent is None:
            raise AgentConfigError(
                "Agent not set up. Call set_up() before querying.",
                details={"is_setup": self._is_setup},
            )

    def query(
        self,
        message: str,
        user_id: str | None = None,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
        memories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute a synchronous query against the agent.

        Args:
            message: User message to process
            user_id: Optional user identifier
            session_id: Optional session identifier
            context: Optional additional context
            memories: Optional list of retrieved memories

        Returns:
            Dictionary containing:
                - response: Agent response text
                - tool_calls: List of executed tools
                - usage: Token usage information
                - metadata: Query metadata

        Raises:
            AgentQueryError: If the query fails
        """
        self._ensure_setup()

        start_time = datetime.now(UTC)

        try:
            # Build the full message with context
            full_message = self._build_message(message, context, memories)

            # Run the agent synchronously
            result = self._agent.run_sync(full_message)

            # Extract response data
            response_data = self._process_result(
                result=result,
                user_id=user_id,
                session_id=session_id,
                start_time=start_time,
            )

            logger.info(
                "query_completed",
                user_id=user_id,
                session_id=session_id,
                latency_ms=(datetime.now(UTC) - start_time).total_seconds() * 1000,
            )

            return response_data

        except ToolExecutionError:
            raise
        except Exception as e:
            logger.error(
                "query_failed",
                user_id=user_id,
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise AgentQueryError(
                f"Query failed: {e}",
                user_id=user_id,
                session_id=session_id,
                details={"error_type": type(e).__name__},
            ) from e

    async def aquery(
        self,
        message: str,
        user_id: str | None = None,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
        memories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute an asynchronous query against the agent.

        Args:
            message: User message to process
            user_id: Optional user identifier
            session_id: Optional session identifier
            context: Optional additional context
            memories: Optional list of retrieved memories

        Returns:
            Dictionary containing:
                - response: Agent response text
                - tool_calls: List of executed tools
                - usage: Token usage information
                - metadata: Query metadata

        Raises:
            AgentQueryError: If the query fails
        """
        self._ensure_setup()

        start_time = datetime.now(UTC)

        try:
            # Build the full message with context
            full_message = self._build_message(message, context, memories)

            # Run the agent asynchronously
            result = await self._agent.run(full_message)

            # Extract response data
            response_data = self._process_result(
                result=result,
                user_id=user_id,
                session_id=session_id,
                start_time=start_time,
            )

            logger.info(
                "async_query_completed",
                user_id=user_id,
                session_id=session_id,
                latency_ms=(datetime.now(UTC) - start_time).total_seconds() * 1000,
            )

            return response_data

        except ToolExecutionError:
            raise
        except Exception as e:
            logger.error(
                "async_query_failed",
                user_id=user_id,
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise AgentQueryError(
                f"Async query failed: {e}",
                user_id=user_id,
                session_id=session_id,
                details={"error_type": type(e).__name__},
            ) from e

    def _build_message(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        memories: list[str] | None = None,
    ) -> str:
        """Build the full message with context and memories.

        Args:
            message: Original user message
            context: Optional additional context
            memories: Optional list of memories

        Returns:
            Full message string
        """
        parts = []

        # Add memories if provided
        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories)
            parts.append(f"[Relevant memories about the user]\n{memory_text}\n")

        # Add context if provided
        if context:
            context_text = "\n".join(f"- {k}: {v}" for k, v in context.items())
            parts.append(f"[Additional context]\n{context_text}\n")

        # Add the user message
        parts.append(message)

        return "\n".join(parts)

    def _process_result(
        self,
        result: Any,
        user_id: str | None,
        session_id: str | None,
        start_time: datetime,
    ) -> dict[str, Any]:
        """Process the agent result into a response dictionary.

        Args:
            result: Pydantic AI result object
            user_id: User identifier
            session_id: Session identifier
            start_time: Query start time

        Returns:
            Response dictionary
        """
        # Extract response text
        response_text = str(result.output) if hasattr(result, "output") else str(result)

        # Extract tool calls if available
        tool_calls = []
        if hasattr(result, "tool_calls"):
            for tc in result.tool_calls:
                tool_calls.append(
                    {
                        "tool": getattr(tc, "name", "unknown"),
                        "args": getattr(tc, "args", {}),
                    }
                )

        # Extract usage information if available
        usage = {}
        if hasattr(result, "usage"):
            usage = {
                "prompt_tokens": getattr(result.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(result.usage, "completion_tokens", 0),
                "total_tokens": getattr(result.usage, "total_tokens", 0),
            }

        return {
            "response": response_text,
            "tool_calls": tool_calls,
            "usage": usage,
            "metadata": {
                "model": self.model_name,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "latency_ms": (datetime.now(UTC) - start_time).total_seconds() * 1000,
            },
        }

    def add_tool(self, tool: Callable[..., Any]) -> None:
        """Add a tool to the agent.

        Note: This should be called before set_up().

        Args:
            tool: Tool function to add
        """
        if self._is_setup:
            logger.warning(
                "tool_added_after_setup",
                message="Adding tool after set_up() may not take effect",
            )
        self.tools.append(tool)

    def register_tools(self, tools: Sequence[Callable[..., Any]]) -> None:
        """Register multiple tools at once.

        Note: This should be called before set_up().

        Args:
            tools: Sequence of tool functions
        """
        for tool in tools:
            self.add_tool(tool)
