"""Custom exceptions for the Agent Engine platform."""

from typing import Any


class AgentError(Exception):
    """Base exception for Agent Engine errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class AgentConfigError(AgentError):
    """Raised when there's a configuration error."""

    pass


class AgentQueryError(AgentError):
    """Raised when a query fails."""

    def __init__(
        self,
        message: str,
        user_id: str | None = None,
        session_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.user_id = user_id
        self.session_id = session_id


class ToolExecutionError(AgentError):
    """Raised when a tool execution fails."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
