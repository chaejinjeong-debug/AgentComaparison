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

Phase 2 additions:
- SM-001~005: Session management integration
- MB-001~006: Memory bank integration
- OB-001~003: Observability integration
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
        config: AgentConfig | None = None,
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
            config: Optional full AgentConfig for Phase 2 features
        """
        self.model_name = model
        self.project = project
        self.location = location
        self.system_prompt = system_prompt
        self.tools = list(tools) if tools else []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.config = config

        # Will be initialized in set_up()
        self._agent: Any = None
        self._is_setup = False

        # Phase 2: Session and Memory managers
        self._session_manager: Any = None
        self._memory_manager: Any = None
        self._tracing_manager: Any = None
        self._metrics_manager: Any = None
        self._logging_manager: Any = None

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
            config=config,
        )

    def set_up(self) -> None:
        """Initialize the Pydantic AI Agent and connect to VertexAI.

        This method performs:
        - VertexAI initialization
        - GoogleProvider configuration
        - Pydantic AI Agent creation with tools
        - Phase 2: Session, Memory, and Observability setup

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

            # Phase 2: Initialize observability
            self._setup_observability()

            # Phase 2: Initialize session manager
            self._setup_session_manager()

            # Phase 2: Initialize memory manager
            self._setup_memory_manager()

            self._is_setup = True

            logger.info(
                "agent_setup_complete",
                model=self.model_name,
                tool_count=len(self.tools),
                session_enabled=self._session_manager is not None,
                memory_enabled=self._memory_manager is not None,
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

    def _setup_observability(self) -> None:
        """Set up observability components (tracing, logging, metrics)."""
        if self.config is None:
            return

        try:
            from agent_engine.observability import (
                LoggingManager,
                MetricsManager,
                TracingManager,
            )

            obs_config = self.config.observability

            # Set up tracing
            if obs_config.tracing_enabled:
                self._tracing_manager = TracingManager.get_instance(
                    config=obs_config,
                    project_id=self.project,
                )
                self._tracing_manager.setup()

            # Set up logging
            if obs_config.logging_enabled:
                self._logging_manager = LoggingManager.get_instance(
                    config=obs_config,
                    project_id=self.project,
                    log_level=self.config.log_level,
                    log_format=self.config.log_format,
                )
                self._logging_manager.setup()

            # Set up metrics
            if obs_config.metrics_enabled:
                self._metrics_manager = MetricsManager.get_instance(
                    config=obs_config,
                    project_id=self.project,
                )
                self._metrics_manager.setup()

            logger.debug(
                "observability_initialized",
                tracing=obs_config.tracing_enabled,
                logging=obs_config.logging_enabled,
                metrics=obs_config.metrics_enabled,
            )

        except Exception as e:
            logger.warning("observability_setup_failed", error=str(e))

    def _setup_session_manager(self) -> None:
        """Set up session manager."""
        if self.config is None or not self.config.session.enabled:
            return

        try:
            from agent_engine.sessions import SessionManager

            self._session_manager = SessionManager(config=self.config.session)
            logger.debug("session_manager_initialized")

        except Exception as e:
            logger.warning("session_manager_setup_failed", error=str(e))

    def _setup_memory_manager(self) -> None:
        """Set up memory manager and memory tools."""
        if self.config is None or not self.config.memory.enabled:
            return

        try:
            from agent_engine.memory import MemoryManager
            from agent_engine.tools.memory_tools import set_memory_manager

            self._memory_manager = MemoryManager(config=self.config.memory)
            set_memory_manager(self._memory_manager)
            logger.debug("memory_manager_initialized")

        except Exception as e:
            logger.warning("memory_manager_setup_failed", error=str(e))

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

        # Record request metric
        if self._metrics_manager:
            self._metrics_manager.record_request(method="query", user_id=user_id)

        try:
            # Set user context for memory tools
            if user_id:
                from agent_engine.tools.memory_tools import set_current_user
                set_current_user(user_id)

            # Retrieve memories if memory manager is available
            if memories is None and self._memory_manager and user_id:
                memories = self._get_user_memories(user_id, message)

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

            # Record metrics
            latency_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            if self._metrics_manager:
                self._metrics_manager.record_latency(latency_ms, method="query")
                if "usage" in response_data and response_data["usage"]:
                    self._metrics_manager.record_tokens(
                        input_tokens=response_data["usage"].get("prompt_tokens", 0),
                        output_tokens=response_data["usage"].get("completion_tokens", 0),
                        model=self.model_name,
                    )

            logger.info(
                "query_completed",
                user_id=user_id,
                session_id=session_id,
                latency_ms=latency_ms,
            )

            return response_data

        except ToolExecutionError:
            if self._metrics_manager:
                self._metrics_manager.record_error("ToolExecutionError", method="query")
            raise
        except Exception as e:
            if self._metrics_manager:
                self._metrics_manager.record_error(type(e).__name__, method="query")
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

        # Record request metric
        if self._metrics_manager:
            self._metrics_manager.record_request(method="aquery", user_id=user_id)

        try:
            # Set user context for memory tools
            if user_id:
                from agent_engine.tools.memory_tools import set_current_user
                set_current_user(user_id)

            # Retrieve memories if memory manager is available
            if memories is None and self._memory_manager and user_id:
                memories = await self._aget_user_memories(user_id, message)

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

            # Record metrics
            latency_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            if self._metrics_manager:
                self._metrics_manager.record_latency(latency_ms, method="aquery")
                if "usage" in response_data and response_data["usage"]:
                    self._metrics_manager.record_tokens(
                        input_tokens=response_data["usage"].get("prompt_tokens", 0),
                        output_tokens=response_data["usage"].get("completion_tokens", 0),
                        model=self.model_name,
                    )

            logger.info(
                "async_query_completed",
                user_id=user_id,
                session_id=session_id,
                latency_ms=latency_ms,
            )

            return response_data

        except ToolExecutionError:
            if self._metrics_manager:
                self._metrics_manager.record_error("ToolExecutionError", method="aquery")
            raise
        except Exception as e:
            if self._metrics_manager:
                self._metrics_manager.record_error(type(e).__name__, method="aquery")
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

    async def query_with_session(
        self,
        message: str,
        user_id: str,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a query with automatic session management.

        This method:
        1. Creates or retrieves a session
        2. Records the user message as an event
        3. Retrieves relevant memories
        4. Executes the query
        5. Records the response as an event
        6. Optionally generates new memories

        Args:
            message: User message to process
            user_id: User identifier (required for session)
            session_id: Optional existing session ID
            context: Optional additional context

        Returns:
            Dictionary with response data and session_id

        Raises:
            AgentQueryError: If the query fails
        """
        self._ensure_setup()

        if self._session_manager is None:
            # Fallback to regular query if sessions disabled
            return await self.aquery(message=message, user_id=user_id, context=context)

        from agent_engine.sessions import EventAuthor

        # Get or create session
        if session_id:
            session = await self._session_manager.get_session(session_id)
            if session is None:
                # Session expired or not found, create new one
                session = await self._session_manager.create_session(user_id)
        else:
            session = await self._session_manager.create_session(user_id)

        session_id = session.session_id

        # Record user message event
        await self._session_manager.append_event(
            session_id=session_id,
            author=EventAuthor.USER,
            content={"text": message},
        )

        # Retrieve memories
        memories = None
        if self._memory_manager:
            memories = await self._aget_user_memories(user_id, message)

        # Execute query
        response_data = await self.aquery(
            message=message,
            user_id=user_id,
            session_id=session_id,
            context=context,
            memories=memories,
        )

        # Record agent response event
        await self._session_manager.append_event(
            session_id=session_id,
            author=EventAuthor.AGENT,
            content={
                "text": response_data["response"],
                "tool_calls": response_data.get("tool_calls", []),
            },
        )

        # Generate memories from session (if auto-generate enabled)
        if self._memory_manager and self.config and self.config.memory.auto_generate:
            await self._memory_manager.generate_from_session(session, user_id)

        # Update metrics
        if self._metrics_manager:
            self._metrics_manager.update_active_sessions(0)  # Count unchanged

        response_data["session_id"] = session_id
        return response_data

    def _get_user_memories(self, user_id: str, query: str) -> list[str]:
        """Retrieve user memories synchronously.

        Args:
            user_id: User identifier
            query: Query for similarity search

        Returns:
            List of memory strings
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            memories = loop.run_until_complete(
                self._memory_manager.retrieve_memories(
                    user_id=user_id,
                    query=query,
                    max_results=5,
                )
            )
            return [m.fact for m in memories]
        except Exception as e:
            logger.warning("memory_retrieval_failed", error=str(e))
            return []

    async def _aget_user_memories(self, user_id: str, query: str) -> list[str]:
        """Retrieve user memories asynchronously.

        Args:
            user_id: User identifier
            query: Query for similarity search

        Returns:
            List of memory strings
        """
        try:
            memories = await self._memory_manager.retrieve_memories(
                user_id=user_id,
                query=query,
                max_results=5,
            )
            return [m.fact for m in memories]
        except Exception as e:
            logger.warning("memory_retrieval_failed", error=str(e))
            return []

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

    # Phase 2: Session management accessors
    @property
    def session_manager(self) -> Any:
        """Get the session manager instance."""
        return self._session_manager

    @property
    def memory_manager(self) -> Any:
        """Get the memory manager instance."""
        return self._memory_manager

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics including Phase 2 metrics.

        Returns:
            Dictionary with agent statistics
        """
        stats: dict[str, Any] = {
            "model": self.model_name,
            "is_setup": self._is_setup,
            "tool_count": len(self.tools),
        }

        if self._session_manager:
            stats["sessions"] = self._session_manager.get_stats()

        if self._memory_manager:
            stats["memory"] = self._memory_manager.get_stats()

        if self._metrics_manager:
            stats["metrics"] = self._metrics_manager.get_stats()

        return stats
