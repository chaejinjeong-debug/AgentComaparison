"""Pydantic AI Agent implementation.

This module provides the PydanticAIAgent class which implements
the BaseAgent interface using Pydantic AI and Google's Gemini models.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING, Any

import structlog

from agent_engine.core.base_agent import AgentResult, BaseAgent
from agent_engine.exceptions import AgentConfigError, AgentQueryError

if TYPE_CHECKING:
    from agent_engine.config import AgentConfig

logger = structlog.get_logger()


class PydanticAIAgent(BaseAgent):
    """Pydantic AI implementation of BaseAgent.

    This class wraps Pydantic AI with GoogleProvider to create
    an agent that can run queries against Google's Gemini models
    via VertexAI.

    Attributes:
        model: Gemini model name (e.g., "gemini-2.5-pro")
        system_prompt: System prompt for the agent
        temperature: Model temperature setting
        max_tokens: Maximum output tokens

    Example:
        >>> agent = PydanticAIAgent(
        ...     model="gemini-2.5-pro",
        ...     system_prompt="You are a helpful assistant.",
        ...     tools=[search_tool],
        ... )
        >>> agent.initialize(project="my-project", location="asia-northeast3")
        >>> result = agent.run_sync("Hello!")
        >>> print(result.response_text)
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        system_prompt: str = "You are a helpful AI assistant.",
        tools: Sequence[Callable[..., Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize the Pydantic AI Agent.

        Args:
            model: Gemini model name
            system_prompt: System prompt for the agent
            tools: Sequence of tool functions to register
            temperature: Model temperature (0.0-2.0)
            max_tokens: Maximum output tokens
        """
        self._model_name = model
        self._system_prompt = system_prompt
        self._tools: list[Callable[..., Any]] = list(tools) if tools else []
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Will be initialized in initialize()
        self._agent: Any = None
        self._is_initialized = False
        self._project: str = ""
        self._location: str = ""

        logger.debug(
            "pydantic_ai_agent_created",
            model=model,
            tool_count=len(self._tools),
        )

    @property
    def is_initialized(self) -> bool:
        """Check if the agent has been initialized."""
        return self._is_initialized

    @property
    def model_name(self) -> str:
        """Get the model name being used."""
        return self._model_name

    @property
    def tools(self) -> list[Callable[..., Any]]:
        """Get the list of registered tools."""
        return self._tools

    @property
    def system_prompt(self) -> str:
        """Get the system prompt."""
        return self._system_prompt

    @property
    def temperature(self) -> float:
        """Get the temperature setting."""
        return self._temperature

    @property
    def max_tokens(self) -> int:
        """Get the max tokens setting."""
        return self._max_tokens

    def initialize(
        self,
        project: str,
        location: str,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize the Pydantic AI Agent with VertexAI.

        This method:
        - Initializes VertexAI SDK
        - Creates GoogleProvider with VertexAI
        - Creates the Pydantic AI Agent

        Args:
            project: GCP project ID
            location: GCP region
            config: Optional full agent configuration (unused, for interface compliance)

        Raises:
            AgentConfigError: If initialization fails
        """
        try:
            import vertexai
            from pydantic_ai import Agent
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google import GoogleProvider

            self._project = project
            self._location = location

            # Initialize VertexAI
            vertexai.init(project=project, location=location)

            # Create GoogleProvider with VertexAI
            provider = GoogleProvider(
                vertexai=True,
                project=project,
                location=location,
            )

            # Create GoogleModel
            google_model = GoogleModel(
                self._model_name,
                provider=provider,
            )

            # Create Pydantic AI Agent
            self._agent = Agent(
                model=google_model,
                system_prompt=self._system_prompt,
                tools=self._tools,
            )

            self._is_initialized = True

            logger.info(
                "pydantic_ai_agent_initialized",
                model=self._model_name,
                project=project,
                location=location,
                tool_count=len(self._tools),
            )

        except ImportError as e:
            raise AgentConfigError(
                f"Required package not installed: {e}",
                details={"missing_package": str(e)},
            ) from e
        except Exception as e:
            raise AgentConfigError(
                f"Failed to initialize Pydantic AI Agent: {e}",
                details={"error_type": type(e).__name__},
            ) from e

    def _ensure_initialized(self) -> None:
        """Ensure the agent has been initialized."""
        if not self._is_initialized or self._agent is None:
            raise AgentConfigError(
                "Agent not initialized. Call initialize() before running queries.",
                details={"is_initialized": self._is_initialized},
            )

    def run_sync(self, message: str, **kwargs: Any) -> AgentResult:
        """Execute a synchronous query against the agent.

        Args:
            message: The user message to process
            **kwargs: Additional parameters (unused)

        Returns:
            AgentResult containing the response and metadata

        Raises:
            AgentQueryError: If the query fails
        """
        self._ensure_initialized()

        try:
            result = self._agent.run_sync(message)
            return self._create_agent_result(result)

        except Exception as e:
            logger.error(
                "pydantic_ai_sync_query_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise AgentQueryError(
                f"Sync query failed: {e}",
                details={"error_type": type(e).__name__},
            ) from e

    async def run_async(self, message: str, **kwargs: Any) -> AgentResult:
        """Execute an asynchronous query against the agent.

        Args:
            message: The user message to process
            **kwargs: Additional parameters (unused)

        Returns:
            AgentResult containing the response and metadata

        Raises:
            AgentQueryError: If the query fails
        """
        self._ensure_initialized()

        try:
            result = await self._agent.run(message)
            return self._create_agent_result(result)

        except Exception as e:
            logger.error(
                "pydantic_ai_async_query_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise AgentQueryError(
                f"Async query failed: {e}",
                details={"error_type": type(e).__name__},
            ) from e

    async def run_stream(
        self, message: str, **kwargs: Any
    ) -> AsyncIterator[tuple[str, AgentResult | None]]:
        """Execute a streaming query against the agent.

        Yields text chunks as they are generated, with the final
        yield including the complete AgentResult.

        Args:
            message: The user message to process
            **kwargs: Additional parameters (unused)

        Yields:
            Tuples of (text_chunk, optional_final_result)

        Raises:
            AgentQueryError: If the streaming query fails
        """
        self._ensure_initialized()

        try:
            async with self._agent.run_stream(message) as stream:
                async for text in stream.stream_text():
                    yield (text, None)

                # Get the final result
                result = stream.result
                agent_result = self._create_agent_result(result)
                yield ("", agent_result)

        except Exception as e:
            logger.error(
                "pydantic_ai_stream_query_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise AgentQueryError(
                f"Stream query failed: {e}",
                details={"error_type": type(e).__name__},
            ) from e

    def _create_agent_result(self, result: Any) -> AgentResult:
        """Create an AgentResult from a Pydantic AI result.

        Args:
            result: Pydantic AI result object

        Returns:
            Standardized AgentResult
        """
        return AgentResult(
            output=result,
            response_text=self.extract_response_text(result),
            tool_calls=self.extract_tool_calls(result),
            usage=self.extract_usage(result),
        )

    def add_tool(self, tool: Callable[..., Any]) -> None:
        """Add a tool to the agent.

        Note: Tools should be added before initialize() is called.

        Args:
            tool: Tool function to add
        """
        if self._is_initialized:
            logger.warning(
                "tool_added_after_initialization",
                message="Adding tool after initialize() may not take effect",
            )
        self._tools.append(tool)
