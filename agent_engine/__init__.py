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
from agent_engine.constants import (
    DEFAULT_DESCRIPTION,
    DEFAULT_DISPLAY_NAME,
    DEFAULT_LOCATION,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
)
from agent_engine.core import (
    AgentEngineWrapper,
    AgentResult,
    BaseAgent,
    MessageBuilder,
    PydanticAIAgent,
    ResultProcessor,
)
from agent_engine.envs import Env
from agent_engine.exceptions import (
    AgentConfigError,
    AgentError,
    AgentQueryError,
    ToolExecutionError,
)

# Phase 3: Version Management - 주석 처리 (Agent Engine 호환성)
# from agent_engine.version import (
#     DeploymentInfo,
#     RollbackManager,
#     Version,
#     VersionRegistry,
#     VersionStatus,
# )

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
    # Environment Variables
    "Env",
    # Constants (commonly used)
    "DEFAULT_MODEL",
    "DEFAULT_LOCATION",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_DISPLAY_NAME",
    "DEFAULT_DESCRIPTION",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_FORMAT",
    # Exceptions
    "AgentError",
    "AgentConfigError",
    "AgentQueryError",
    "ToolExecutionError",
    # Phase 3: Version Management - 주석 처리 (Agent Engine 호환성)
    # "Version",
    # "VersionStatus",
    # "DeploymentInfo",
    # "VersionRegistry",
    # "RollbackManager",
]
