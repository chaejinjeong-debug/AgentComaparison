"""Pydantic AI Agent Platform on VertexAI Agent Engine."""

from agent_engine.agent import PydanticAIAgentWrapper
from agent_engine.config import (
    AgentConfig,
    MemoryConfig,
    ObservabilityConfig,
    SessionConfig,
)
from agent_engine.exceptions import (
    AgentConfigError,
    AgentError,
    AgentQueryError,
    ToolExecutionError,
)

__version__ = "0.2.0"

__all__ = [
    # Core
    "PydanticAIAgentWrapper",
    # Configuration
    "AgentConfig",
    "SessionConfig",
    "MemoryConfig",
    "ObservabilityConfig",
    # Exceptions
    "AgentError",
    "AgentConfigError",
    "AgentQueryError",
    "ToolExecutionError",
]
