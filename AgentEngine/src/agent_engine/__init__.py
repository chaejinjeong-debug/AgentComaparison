"""Pydantic AI Agent Platform on VertexAI Agent Engine."""

from agent_engine.agent import PydanticAIAgentWrapper, StreamChunk
from agent_engine.config import (
    AgentConfig,
    EvaluationConfig,
    MemoryBackendType,
    MemoryConfig,
    ObservabilityConfig,
    SessionBackendType,
    SessionConfig,
)

# Core module (refactored components)
from agent_engine.core import (
    AgentEngineWrapper,
    AgentResult,
    BaseAgent,
    MessageBuilder,
    PydanticAIAgent,
    ResultProcessor,
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

__version__ = "0.5.0"

__all__ = [
    # Core (Facade - backward compatible)
    "PydanticAIAgentWrapper",
    "StreamChunk",
    # Core (Refactored components)
    "BaseAgent",
    "AgentResult",
    "PydanticAIAgent",
    "AgentEngineWrapper",
    "MessageBuilder",
    "ResultProcessor",
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
