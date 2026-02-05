"""Agent Engine Wrapper for deployment orchestration.

This module provides the AgentEngineWrapper class which handles
deployment-related concerns: session management, memory, observability,
and metrics recording while delegating actual query execution to
a BaseAgent implementation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from agent_engine.core.base_agent import BaseAgent
from agent_engine.core.message_builder import MessageBuilder
from agent_engine.core.result_processor import ResultProcessor
from agent_engine.exceptions import AgentConfigError, AgentQueryError, ToolExecutionError

if TYPE_CHECKING:
    from agent_engine.config import AgentConfig

logger = structlog.get_logger()


@dataclass
class StreamChunk:
    """Represents a single chunk in the streaming response.

    Attributes:
        chunk: The text content of this chunk
        done: Whether this is the final chunk
        tool_call: Optional tool call information if a tool was invoked
        metadata: Optional metadata about the chunk
    """

    chunk: str
    done: bool = False
    tool_call: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class AgentEngineWrapper:
    """Deployment wrapper conforming to VertexAI Agent Engine specification.

    This class wraps any BaseAgent implementation to provide:
    - Agent Engine spec compliance (__init__, set_up, query)
    - Session management integration
    - Memory bank integration
    - Observability (tracing, logging, metrics)
    - Query orchestration

    The wrapper delegates actual LLM interactions to the provided
    BaseAgent implementation, focusing on infrastructure concerns.

    Attributes:
        agent: The BaseAgent implementation to wrap
        config: Full agent configuration
        project: GCP project ID
        location: GCP region

    Example:
        >>> from agent_engine.core import AgentEngineWrapper, PydanticAIAgent
        >>>
        >>> agent = PydanticAIAgent(model="gemini-2.5-pro", tools=[my_tool])
        >>> wrapper = AgentEngineWrapper(
        ...     agent=agent,
        ...     project="my-project",
        ...     location="asia-northeast3",
        ... )
        >>> wrapper.set_up()
        >>> response = wrapper.query(message="Hello!")
    """

    def __init__(
        self,
        agent: BaseAgent,
        project: str = "",
        location: str = "asia-northeast3",
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize the Agent Engine wrapper.

        Args:
            agent: BaseAgent implementation to wrap
            project: GCP project ID
            location: GCP region
            config: Optional full AgentConfig for Phase 2 features
        """
        self.agent = agent
        self.project = project
        self.location = location
        self.config = config

        # Will be initialized in set_up()
        self._is_setup = False

        # Phase 2: Managers
        self._session_manager: Any = None
        self._memory_manager: Any = None
        self._tracing_manager: Any = None
        self._metrics_manager: Any = None
        self._logging_manager: Any = None

        # Utilities
        self._message_builder = MessageBuilder()
        self._result_processor: ResultProcessor | None = None

        logger.info(
            "agent_engine_wrapper_initialized",
            agent_type=type(agent).__name__,
            project=project,
            location=location,
        )

    def set_up(self) -> None:
        """Initialize the agent and all managers.

        This method:
        - Initializes the wrapped agent
        - Sets up observability (tracing, logging, metrics)
        - Sets up session manager
        - Sets up memory manager

        Raises:
            AgentConfigError: If initialization fails
        """
        try:
            # Initialize the wrapped agent
            self.agent.initialize(
                project=self.project,
                location=self.location,
                config=self.config,
            )

            # Initialize result processor with model name
            self._result_processor = ResultProcessor(self.agent.model_name)

            # Phase 2: Initialize observability
            self._setup_observability()

            # Phase 2: Initialize session manager
            self._setup_session_manager()

            # Phase 2: Initialize memory manager
            self._setup_memory_manager()

            self._is_setup = True

            logger.info(
                "agent_engine_wrapper_setup_complete",
                model=self.agent.model_name,
                tool_count=len(self.agent.tools),
                session_enabled=self._session_manager is not None,
                memory_enabled=self._memory_manager is not None,
            )

        except Exception as e:
            if isinstance(e, AgentConfigError):
                raise
            raise AgentConfigError(
                f"Failed to set up wrapper: {e}",
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
        """Ensure the wrapper has been set up."""
        if not self._is_setup:
            raise AgentConfigError(
                "Wrapper not set up. Call set_up() before querying.",
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
            Response dictionary

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

            # Retrieve memories if needed
            if memories is None and self._memory_manager and user_id:
                memories = self._get_user_memories(user_id, message)

            # Retrieve session history if needed
            session_history = None
            if session_id and self._session_manager:
                session_history = self._get_session_history_sync(session_id)

            # Build the full message
            full_message = self._message_builder.build(message, context, memories, session_history)

            # Run the agent
            result = self.agent.run_sync(full_message)

            # Process result
            response_data = self._result_processor.process(
                result=result,
                user_id=user_id,
                session_id=session_id,
                start_time=start_time,
            )

            # Record metrics
            self._record_metrics(response_data, start_time, "query")

            logger.info(
                "query_completed",
                user_id=user_id,
                session_id=session_id,
                latency_ms=response_data["metadata"]["latency_ms"],
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
            if isinstance(e, AgentQueryError):
                raise
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
            Response dictionary

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

            # Retrieve memories if needed
            if memories is None and self._memory_manager and user_id:
                memories = await self._aget_user_memories(user_id, message)

            # Retrieve session history if needed
            session_history = None
            if session_id and self._session_manager:
                session_history = await self._aget_session_history(session_id)

            # Build the full message
            full_message = self._message_builder.build(message, context, memories, session_history)

            # Run the agent
            result = await self.agent.run_async(full_message)

            # Process result
            response_data = self._result_processor.process(
                result=result,
                user_id=user_id,
                session_id=session_id,
                start_time=start_time,
            )

            # Record metrics
            self._record_metrics(response_data, start_time, "aquery")

            logger.info(
                "async_query_completed",
                user_id=user_id,
                session_id=session_id,
                latency_ms=response_data["metadata"]["latency_ms"],
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
            if isinstance(e, AgentQueryError):
                raise
            raise AgentQueryError(
                f"Async query failed: {e}",
                user_id=user_id,
                session_id=session_id,
                details={"error_type": type(e).__name__},
            ) from e

    async def stream_query(
        self,
        message: str,
        user_id: str | None = None,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
        memories: list[str] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Execute a streaming query against the agent.

        Args:
            message: User message to process
            user_id: Optional user identifier
            session_id: Optional session identifier
            context: Optional additional context
            memories: Optional list of retrieved memories

        Yields:
            StreamChunk objects

        Raises:
            AgentQueryError: If the streaming query fails
        """
        self._ensure_setup()

        start_time = datetime.now(UTC)

        # Record request metric
        if self._metrics_manager:
            self._metrics_manager.record_request(method="stream_query", user_id=user_id)

        try:
            # Set user context for memory tools
            if user_id:
                from agent_engine.tools.memory_tools import set_current_user

                set_current_user(user_id)

            # Retrieve memories if needed
            if memories is None and self._memory_manager and user_id:
                memories = await self._aget_user_memories(user_id, message)

            # Retrieve session history if needed
            session_history = None
            if session_id and self._session_manager:
                session_history = await self._aget_session_history(session_id)

            # Build the full message
            full_message = self._message_builder.build(message, context, memories, session_history)

            # Track tool calls
            tool_calls: list[dict[str, Any]] = []

            # Stream from agent
            async for text, final_result in self.agent.run_stream(full_message):
                if final_result is not None:
                    # Extract tool calls from final result
                    tool_calls = final_result.tool_calls
                    for tc in tool_calls:
                        yield StreamChunk(chunk="", done=False, tool_call=tc)
                else:
                    yield StreamChunk(chunk=text, done=False)

            # Record metrics
            latency_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            if self._metrics_manager:
                self._metrics_manager.record_latency(latency_ms, method="stream_query")

            # Yield final chunk with metadata
            yield StreamChunk(
                chunk="",
                done=True,
                metadata=ResultProcessor.create_stream_metadata(
                    model_name=self.agent.model_name,
                    user_id=user_id,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    tool_calls=tool_calls,
                ),
            )

            logger.info(
                "stream_query_completed",
                user_id=user_id,
                session_id=session_id,
                latency_ms=latency_ms,
            )

        except Exception as e:
            if self._metrics_manager:
                self._metrics_manager.record_error(type(e).__name__, method="stream_query")
            logger.error(
                "stream_query_failed",
                user_id=user_id,
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            if isinstance(e, AgentQueryError):
                raise
            raise AgentQueryError(
                f"Stream query failed: {e}",
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

        Args:
            message: User message to process
            user_id: User identifier (required)
            session_id: Optional existing session ID
            context: Optional additional context

        Returns:
            Response dictionary with session_id

        Raises:
            AgentQueryError: If the query fails
        """
        self._ensure_setup()

        if self._session_manager is None:
            return await self.aquery(message=message, user_id=user_id, context=context)

        from agent_engine.sessions import EventAuthor

        # Get or create session
        if session_id:
            session = await self._session_manager.get_session(session_id)
            if session is None:
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

        # Generate memories from session
        if self._memory_manager and self.config and self.config.memory.auto_generate:
            await self._memory_manager.generate_from_session(session, user_id)

        response_data["session_id"] = session_id
        return response_data

    async def stream_query_with_session(
        self,
        message: str,
        user_id: str,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Execute a streaming query with automatic session management.

        Args:
            message: User message to process
            user_id: User identifier (required)
            session_id: Optional existing session ID
            context: Optional additional context

        Yields:
            StreamChunk objects

        Raises:
            AgentQueryError: If the streaming query fails
        """
        self._ensure_setup()

        if self._session_manager is None:
            async for chunk in self.stream_query(message=message, user_id=user_id, context=context):
                yield chunk
            return

        from agent_engine.sessions import EventAuthor

        # Get or create session
        if session_id:
            session = await self._session_manager.get_session(session_id)
            if session is None:
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

        # Stream query and accumulate response
        accumulated_response = ""
        tool_calls: list[dict[str, Any]] = []

        async for chunk in self.stream_query(
            message=message,
            user_id=user_id,
            session_id=session_id,
            context=context,
            memories=memories,
        ):
            accumulated_response += chunk.chunk
            if chunk.tool_call:
                tool_calls.append(chunk.tool_call)

            # Add session_id to metadata on final chunk
            if chunk.done and chunk.metadata:
                chunk.metadata["session_id"] = session_id

            yield chunk

        # Record agent response event
        await self._session_manager.append_event(
            session_id=session_id,
            author=EventAuthor.AGENT,
            content={
                "text": accumulated_response,
                "tool_calls": tool_calls,
            },
        )

        # Generate memories from session
        if self._memory_manager and self.config and self.config.memory.auto_generate:
            await self._memory_manager.generate_from_session(session, user_id)

    def stream_query_sync(
        self,
        message: str,
        user_id: str | None = None,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
        memories: list[str] | None = None,
    ) -> Iterator[StreamChunk]:
        """Execute a synchronous streaming query.

        Args:
            message: User message to process
            user_id: Optional user identifier
            session_id: Optional session identifier
            context: Optional additional context
            memories: Optional list of retrieved memories

        Yields:
            StreamChunk objects

        Raises:
            AgentQueryError: If the streaming query fails
        """
        import asyncio

        async def _run_stream() -> list[StreamChunk]:
            chunks = []
            async for chunk in self.stream_query(
                message=message,
                user_id=user_id,
                session_id=session_id,
                context=context,
                memories=memories,
            ):
                chunks.append(chunk)
            return chunks

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        chunks = loop.run_until_complete(_run_stream())
        yield from chunks

    def _get_user_memories(self, user_id: str, query: str) -> list[str]:
        """Retrieve user memories synchronously."""
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
        """Retrieve user memories asynchronously."""
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

    def _get_session_history_sync(self, session_id: str) -> list[dict[str, str]]:
        """Retrieve session history synchronously."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            events = loop.run_until_complete(self._session_manager.list_events(session_id))
            return MessageBuilder.parse_session_events(events)
        except Exception as e:
            logger.warning("session_history_retrieval_failed", error=str(e))
            return []

    async def _aget_session_history(self, session_id: str) -> list[dict[str, str]]:
        """Retrieve session history asynchronously."""
        try:
            events = await self._session_manager.list_events(session_id)
            return MessageBuilder.parse_session_events(events)
        except Exception as e:
            logger.warning("session_history_retrieval_failed", error=str(e))
            return []

    def _record_metrics(
        self,
        response_data: dict[str, Any],
        start_time: datetime,
        method: str,
    ) -> None:
        """Record metrics for a query."""
        if not self._metrics_manager:
            return

        latency_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
        self._metrics_manager.record_latency(latency_ms, method=method)

        if "usage" in response_data and response_data["usage"]:
            self._metrics_manager.record_tokens(
                input_tokens=response_data["usage"].get("prompt_tokens", 0),
                output_tokens=response_data["usage"].get("completion_tokens", 0),
                model=self.agent.model_name,
            )

    # Property accessors for managers
    @property
    def session_manager(self) -> Any:
        """Get the session manager instance."""
        return self._session_manager

    @property
    def memory_manager(self) -> Any:
        """Get the memory manager instance."""
        return self._memory_manager

    def get_stats(self) -> dict[str, Any]:
        """Get wrapper statistics including Phase 2 metrics."""
        stats: dict[str, Any] = {
            "model": self.agent.model_name,
            "is_setup": self._is_setup,
            "tool_count": len(self.agent.tools),
        }

        if self._session_manager:
            stats["sessions"] = self._session_manager.get_stats()

        if self._memory_manager:
            stats["memory"] = self._memory_manager.get_stats()

        if self._metrics_manager:
            stats["metrics"] = self._metrics_manager.get_stats()

        return stats
