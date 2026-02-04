"""Base Agent interface and common data structures.

This module defines the abstract interface that all agent implementations
must follow, enabling pluggable agent backends (Pydantic AI, Anthropic, OpenAI, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_engine.config import AgentConfig


@dataclass
class AgentResult:
    """Standardized result from agent execution.

    This dataclass provides a consistent structure for agent responses
    across different implementations.

    Attributes:
        output: The raw output from the agent (model-specific type)
        response_text: Extracted text response
        tool_calls: List of tool calls made during execution
        usage: Token usage information
        metadata: Additional metadata from the execution
    """

    output: Any
    response_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all agent implementations.

    This interface defines the contract that all agent implementations
    must follow, enabling the AgentEngineWrapper to work with different
    LLM backends (Pydantic AI, Anthropic, OpenAI, etc.)

    Implementations should handle:
    - Model initialization and configuration
    - Synchronous and asynchronous query execution
    - Streaming response support
    - Tool registration and execution

    Example:
        >>> class MyCustomAgent(BaseAgent):
        ...     def initialize(self, config):
        ...         self._model = CustomModel(config)
        ...
        ...     def run_sync(self, message, **kwargs):
        ...         result = self._model.generate(message)
        ...         return AgentResult(output=result, response_text=str(result))
    """

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the agent has been initialized.

        Returns:
            True if initialize() has been successfully called
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name being used.

        Returns:
            Model identifier string
        """
        ...

    @property
    @abstractmethod
    def tools(self) -> list[Callable[..., Any]]:
        """Get the list of registered tools.

        Returns:
            List of tool functions
        """
        ...

    @abstractmethod
    def initialize(
        self,
        project: str,
        location: str,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize the agent with project configuration.

        This method should set up the underlying model/provider
        and prepare the agent for query execution.

        Args:
            project: GCP project ID
            location: GCP region
            config: Optional full agent configuration

        Raises:
            AgentConfigError: If initialization fails
        """
        ...

    @abstractmethod
    def run_sync(self, message: str, **kwargs: Any) -> AgentResult:
        """Execute a synchronous query against the agent.

        Args:
            message: The user message to process
            **kwargs: Additional implementation-specific parameters

        Returns:
            AgentResult containing the response and metadata

        Raises:
            AgentQueryError: If the query fails
        """
        ...

    @abstractmethod
    async def run_async(self, message: str, **kwargs: Any) -> AgentResult:
        """Execute an asynchronous query against the agent.

        Args:
            message: The user message to process
            **kwargs: Additional implementation-specific parameters

        Returns:
            AgentResult containing the response and metadata

        Raises:
            AgentQueryError: If the query fails
        """
        ...

    @abstractmethod
    async def run_stream(
        self, message: str, **kwargs: Any
    ) -> AsyncIterator[tuple[str, AgentResult | None]]:
        """Execute a streaming query against the agent.

        Yields text chunks as they are generated, with the final
        yield including the complete AgentResult.

        Args:
            message: The user message to process
            **kwargs: Additional implementation-specific parameters

        Yields:
            Tuples of (text_chunk, optional_final_result)
            - During streaming: (chunk, None)
            - Final yield: ("", AgentResult)

        Raises:
            AgentQueryError: If the streaming query fails
        """
        ...

    def add_tool(self, tool: Callable[..., Any]) -> None:
        """Add a tool to the agent.

        Default implementation for tool registration.
        Override if custom handling is needed.

        Args:
            tool: Tool function to add
        """
        self.tools.append(tool)

    def register_tools(self, tools: Sequence[Callable[..., Any]]) -> None:
        """Register multiple tools at once.

        Args:
            tools: Sequence of tool functions
        """
        for tool in tools:
            self.add_tool(tool)

    @staticmethod
    def extract_response_text(result: Any) -> str:
        """Extract text from a model-specific result object.

        Utility method for implementations to extract text response
        from their model's result format.

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

    @staticmethod
    def extract_tool_calls(result: Any) -> list[dict[str, Any]]:
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

    @staticmethod
    def extract_usage(result: Any) -> dict[str, int]:
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
