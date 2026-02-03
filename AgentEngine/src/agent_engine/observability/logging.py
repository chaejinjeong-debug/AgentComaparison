"""Structured logging for the Agent Engine platform.

Implements OB-002: Cloud Logging structured logging.
"""

import logging
import sys
from typing import Any

import structlog

try:
    from google.cloud import logging as cloud_logging

    CLOUD_LOGGING_AVAILABLE = True
except ImportError:
    CLOUD_LOGGING_AVAILABLE = False

from agent_engine.config import ObservabilityConfig

logger = structlog.get_logger(__name__)


class LoggingManager:
    """Manages structured logging with Cloud Logging integration.

    OB-002: Cloud Logging structured logging.

    Attributes:
        config: Observability configuration
        _initialized: Whether logging has been set up
    """

    _instance: "LoggingManager | None" = None

    def __init__(
        self,
        config: ObservabilityConfig | None = None,
        project_id: str | None = None,
        log_level: str = "INFO",
        log_format: str = "json",
    ) -> None:
        """Initialize the logging manager.

        Args:
            config: Observability configuration
            project_id: GCP project ID for Cloud Logging
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Log format (json or text)
        """
        self.config = config or ObservabilityConfig()
        self.project_id = project_id
        self.log_level = log_level.upper()
        self.log_format = log_format.lower()
        self._initialized = False

    @classmethod
    def get_instance(
        cls,
        config: ObservabilityConfig | None = None,
        project_id: str | None = None,
        log_level: str = "INFO",
        log_format: str = "json",
    ) -> "LoggingManager":
        """Get or create the singleton logging manager.

        Args:
            config: Observability configuration
            project_id: GCP project ID
            log_level: Logging level
            log_format: Log format

        Returns:
            LoggingManager instance
        """
        if cls._instance is None:
            cls._instance = cls(
                config=config,
                project_id=project_id,
                log_level=log_level,
                log_format=log_format,
            )
        return cls._instance

    def setup(self) -> None:
        """Set up structured logging with Cloud Logging.

        Configures structlog for JSON output compatible with
        Cloud Logging's structured log format.
        """
        if not self.config.logging_enabled:
            return

        if self._initialized:
            return

        # Configure standard logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, self.log_level, logging.INFO),
        )

        # Determine processors based on format
        if self.log_format == "json":
            processors = [
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ]
        else:
            processors = [
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer(colors=True),
            ]

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, self.log_level, logging.INFO)
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Set up Cloud Logging client if available
        if CLOUD_LOGGING_AVAILABLE and self.project_id:
            try:
                client = cloud_logging.Client(project=self.project_id)
                client.setup_logging()
                logger.info(
                    "Cloud Logging client configured",
                    project_id=self.project_id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to set up Cloud Logging client",
                    error=str(e),
                )

        self._initialized = True
        logger.info(
            "Logging initialized",
            level=self.log_level,
            format=self.log_format,
        )

    def get_logger(self, name: str | None = None) -> Any:
        """Get a structlog logger.

        Args:
            name: Logger name

        Returns:
            Structlog bound logger
        """
        return structlog.get_logger(name)

    def bind_context(self, **kwargs: Any) -> None:
        """Bind context variables to all subsequent logs.

        Args:
            **kwargs: Context key-value pairs
        """
        structlog.contextvars.bind_contextvars(**kwargs)

    def unbind_context(self, *keys: str) -> None:
        """Unbind context variables.

        Args:
            *keys: Context keys to unbind
        """
        structlog.contextvars.unbind_contextvars(*keys)

    def clear_context(self) -> None:
        """Clear all bound context variables."""
        structlog.contextvars.clear_contextvars()

    @property
    def is_enabled(self) -> bool:
        """Check if logging is enabled and initialized."""
        return self._initialized


def log_agent_request(
    session_id: str | None = None,
    user_id: str | None = None,
    query: str | None = None,
    **kwargs: Any,
) -> None:
    """Log an agent request with structured context.

    Args:
        session_id: Session identifier
        user_id: User identifier
        query: Query text (truncated for logging)
        **kwargs: Additional context
    """
    log = structlog.get_logger("agent.request")
    log.info(
        "Agent request",
        session_id=session_id,
        user_id=user_id,
        query_preview=query[:100] if query else None,
        **kwargs,
    )


def log_agent_response(
    session_id: str | None = None,
    user_id: str | None = None,
    response_length: int | None = None,
    token_usage: dict[str, int] | None = None,
    latency_ms: float | None = None,
    **kwargs: Any,
) -> None:
    """Log an agent response with structured context.

    Args:
        session_id: Session identifier
        user_id: User identifier
        response_length: Length of response
        token_usage: Token usage statistics
        latency_ms: Request latency in milliseconds
        **kwargs: Additional context
    """
    log = structlog.get_logger("agent.response")
    log.info(
        "Agent response",
        session_id=session_id,
        user_id=user_id,
        response_length=response_length,
        token_usage=token_usage,
        latency_ms=latency_ms,
        **kwargs,
    )


def log_tool_execution(
    tool_name: str,
    success: bool,
    execution_time_ms: float | None = None,
    error: str | None = None,
    **kwargs: Any,
) -> None:
    """Log a tool execution with structured context.

    Args:
        tool_name: Name of the tool
        success: Whether execution was successful
        execution_time_ms: Execution time in milliseconds
        error: Error message if failed
        **kwargs: Additional context
    """
    log = structlog.get_logger("tool.execution")
    if success:
        log.info(
            "Tool executed",
            tool_name=tool_name,
            success=success,
            execution_time_ms=execution_time_ms,
            **kwargs,
        )
    else:
        log.error(
            "Tool execution failed",
            tool_name=tool_name,
            success=success,
            error=error,
            execution_time_ms=execution_time_ms,
            **kwargs,
        )
