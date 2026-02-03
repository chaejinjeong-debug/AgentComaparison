"""Pydantic AI Agent Platform on VertexAI Agent Engine."""

from agent_engine.agent import PydanticAIAgentWrapper
from agent_engine.config import (
    AgentConfig,
    EvaluationConfig,
    MemoryBackendType,
    MemoryConfig,
    ObservabilityConfig,
    SessionBackendType,
    SessionConfig,
)
from agent_engine.exceptions import (
    AgentConfigError,
    AgentError,
    AgentQueryError,
    ToolExecutionError,
)

# Phase 3: Version Management
from agent_engine.version import (
    DeploymentInfo,
    RollbackManager,
    Version,
    VersionRegistry,
    VersionStatus,
)

__version__ = "0.3.0"

__all__ = [
    # Core
    "PydanticAIAgentWrapper",
    # Configuration
    "AgentConfig",
    "SessionConfig",
    "SessionBackendType",
    "MemoryConfig",
    "MemoryBackendType",
    "ObservabilityConfig",
    "EvaluationConfig",
    # Exceptions
    "AgentError",
    "AgentConfigError",
    "AgentQueryError",
    "ToolExecutionError",
    # Phase 3: Version Management
    "Version",
    "VersionStatus",
    "DeploymentInfo",
    "VersionRegistry",
    "RollbackManager",
]
