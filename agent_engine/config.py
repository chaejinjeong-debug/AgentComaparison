"""Configuration management for the Agent Engine platform."""

from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from agent_engine.constants import (
    DEFAULT_DESCRIPTION,
    DEFAULT_DISPLAY_NAME,
    DEFAULT_ERROR_RATE_THRESHOLD,
    DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    DEFAULT_LOCATION,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_EVENTS_PER_SESSION,
    DEFAULT_MAX_MEMORIES_PER_USER,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_P50_THRESHOLD_MS,
    DEFAULT_P99_THRESHOLD_MS,
    DEFAULT_QUALITY_THRESHOLD,
    DEFAULT_SESSION_TTL_SECONDS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TEST_DATA_PATH,
    DEFAULT_TRACE_SAMPLE_RATE,
    VALID_LOG_FORMATS,
    VALID_LOG_LEVELS,
)
from agent_engine.envs import Env


class SessionBackendType(str, Enum):
    """Session storage backend type."""

    IN_MEMORY = "in_memory"  # Local development/testing
    VERTEX_AI = "vertex_ai"  # Production with VertexAI Sessions API


class MemoryBackendType(str, Enum):
    """Memory storage backend type."""

    IN_MEMORY = "in_memory"  # Local development/testing
    VERTEX_AI = "vertex_ai"  # Production with VertexAI Memory Bank API


class SessionConfig(BaseModel):
    """Configuration for Session Management.

    Attributes:
        enabled: Whether session management is enabled
        default_ttl_seconds: Default session TTL in seconds (default: 24 hours)
        max_events_per_session: Maximum events to store per session
        backend: Storage backend type (in_memory or vertex_ai)
        agent_engine_id: Agent Engine ID (required for vertex_ai backend)
    """

    enabled: bool = Field(default=True, description="Enable session management")
    default_ttl_seconds: int = Field(
        default=DEFAULT_SESSION_TTL_SECONDS, ge=60, description="Default session TTL (24 hours)"
    )
    max_events_per_session: int = Field(
        default=DEFAULT_MAX_EVENTS_PER_SESSION, ge=10, description="Max events per session"
    )
    backend: SessionBackendType = Field(
        default=SessionBackendType.IN_MEMORY,
        description="Storage backend (in_memory for local, vertex_ai for production)",
    )
    agent_engine_id: str | None = Field(
        default=None,
        description="Agent Engine ID (required for vertex_ai backend)",
    )


class MemoryConfig(BaseModel):
    """Configuration for Memory Bank.

    Attributes:
        enabled: Whether memory bank is enabled
        auto_generate: Automatically extract memories from conversations
        max_memories_per_user: Maximum memories to store per user
        similarity_threshold: Minimum similarity score for memory retrieval
        backend: Storage backend type (in_memory or vertex_ai)
        agent_engine_id: Agent Engine ID (required for vertex_ai backend)
    """

    enabled: bool = Field(default=True, description="Enable memory bank")
    auto_generate: bool = Field(default=True, description="Auto-generate memories from sessions")
    max_memories_per_user: int = Field(
        default=DEFAULT_MAX_MEMORIES_PER_USER, ge=10, description="Max memories per user"
    )
    similarity_threshold: float = Field(
        default=DEFAULT_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for retrieval",
    )
    backend: MemoryBackendType = Field(
        default=MemoryBackendType.IN_MEMORY,
        description="Storage backend (in_memory for local, vertex_ai for production)",
    )
    agent_engine_id: str | None = Field(
        default=None,
        description="Agent Engine ID (required for vertex_ai backend)",
    )


class ObservabilityConfig(BaseModel):
    """Configuration for Observability (Tracing, Logging, Metrics).

    Attributes:
        tracing_enabled: Enable Cloud Trace integration
        logging_enabled: Enable structured logging
        metrics_enabled: Enable metrics collection
        sample_rate: Trace sampling rate (0.0-1.0)
    """

    tracing_enabled: bool = Field(default=True, description="Enable Cloud Trace")
    logging_enabled: bool = Field(default=True, description="Enable structured logging")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    sample_rate: float = Field(
        default=DEFAULT_TRACE_SAMPLE_RATE, ge=0.0, le=1.0, description="Trace sampling rate"
    )


class EvaluationConfig(BaseModel):
    """Configuration for Agent Evaluation (Phase 3).

    Attributes:
        enabled: Enable evaluation in CI/CD
        quality_threshold: Minimum accuracy threshold (0.0-1.0)
        p50_threshold_ms: Maximum P50 latency in milliseconds
        p99_threshold_ms: Maximum P99 latency in milliseconds
        error_rate_threshold: Maximum acceptable error rate
        test_data_path: Path to golden test data
        timeout_seconds: Default timeout per test case
    """

    enabled: bool = Field(default=True, description="Enable evaluation")
    quality_threshold: float = Field(
        default=DEFAULT_QUALITY_THRESHOLD, ge=0.0, le=1.0, description="Minimum accuracy threshold"
    )
    p50_threshold_ms: float = Field(
        default=DEFAULT_P50_THRESHOLD_MS, gt=0, description="Maximum P50 latency (ms)"
    )
    p99_threshold_ms: float = Field(
        default=DEFAULT_P99_THRESHOLD_MS, gt=0, description="Maximum P99 latency (ms)"
    )
    error_rate_threshold: float = Field(
        default=DEFAULT_ERROR_RATE_THRESHOLD, ge=0.0, le=1.0, description="Maximum error rate"
    )
    test_data_path: str = Field(
        default=DEFAULT_TEST_DATA_PATH,
        description="Path to test data file",
    )
    timeout_seconds: float = Field(
        default=DEFAULT_EVALUATION_TIMEOUT_SECONDS, gt=0, description="Default timeout per test"
    )


class AgentConfig(BaseModel):
    """Configuration for the Pydantic AI Agent.

    Attributes:
        project_id: GCP project ID
        location: GCP region (default: asia-northeast3)
        model: Gemini model name (default: gemini-2.5-pro)
        temperature: Model temperature (0.0-2.0)
        max_tokens: Maximum output tokens
        system_prompt: System prompt for the agent
        display_name: Agent display name for Agent Engine
        description: Agent description
        log_level: Logging level
        log_format: Logging format (json or text)
        session: Session management configuration
        memory: Memory bank configuration
        observability: Observability configuration
    """

    project_id: str = Field(..., description="GCP project ID")
    location: str = Field(default=DEFAULT_LOCATION, description="GCP region")
    model: str = Field(default=DEFAULT_MODEL, description="Gemini model name")
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, gt=0)
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt for the agent",
    )
    display_name: str = Field(default=DEFAULT_DISPLAY_NAME, description="Agent display name")
    description: str = Field(
        default=DEFAULT_DESCRIPTION,
        description="Agent description",
    )
    log_level: str = Field(default=DEFAULT_LOG_LEVEL)
    log_format: str = Field(default=DEFAULT_LOG_FORMAT)

    # Phase 2 configurations
    session: SessionConfig = Field(default_factory=SessionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # Phase 3 configurations
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        v_upper = v.upper()
        if v_upper not in VALID_LOG_LEVELS:
            raise ValueError(f"log_level must be one of {VALID_LOG_LEVELS}")
        return v_upper

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        v_lower = v.lower()
        if v_lower not in VALID_LOG_FORMATS:
            raise ValueError(f"log_format must be one of {VALID_LOG_FORMATS}")
        return v_lower

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "AgentConfig":
        """Load configuration from environment variables.

        Args:
            env_file: Optional path to .env file

        Returns:
            AgentConfig instance
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        # Get system prompt (supports file loading)
        system_prompt = Env.get_system_prompt()

        # Session configuration
        session_backend_str = Env.SESSION_BACKEND.lower()
        session_backend = (
            SessionBackendType.VERTEX_AI
            if session_backend_str == "vertex_ai"
            else SessionBackendType.IN_MEMORY
        )
        session_config = SessionConfig(
            enabled=Env.SESSION_ENABLED,
            default_ttl_seconds=Env.SESSION_TTL_SECONDS,
            max_events_per_session=Env.SESSION_MAX_EVENTS,
            backend=session_backend,
            agent_engine_id=Env.SESSION_AGENT_ENGINE_ID,
        )

        # Memory configuration
        memory_backend_str = Env.MEMORY_BACKEND.lower()
        memory_backend = (
            MemoryBackendType.VERTEX_AI
            if memory_backend_str == "vertex_ai"
            else MemoryBackendType.IN_MEMORY
        )
        memory_config = MemoryConfig(
            enabled=Env.MEMORY_ENABLED,
            auto_generate=Env.MEMORY_AUTO_GENERATE,
            max_memories_per_user=Env.MEMORY_MAX_PER_USER,
            similarity_threshold=Env.MEMORY_SIMILARITY_THRESHOLD,
            backend=memory_backend,
            agent_engine_id=Env.MEMORY_AGENT_ENGINE_ID,
        )

        # Observability configuration
        observability_config = ObservabilityConfig(
            tracing_enabled=Env.TRACING_ENABLED,
            logging_enabled=Env.LOGGING_ENABLED,
            metrics_enabled=Env.METRICS_ENABLED,
            sample_rate=Env.TRACE_SAMPLE_RATE,
        )

        # Evaluation configuration (Phase 3)
        evaluation_config = EvaluationConfig(
            enabled=Env.EVALUATION_ENABLED,
            quality_threshold=Env.EVALUATION_QUALITY_THRESHOLD,
            p50_threshold_ms=Env.EVALUATION_P50_THRESHOLD_MS,
            p99_threshold_ms=Env.EVALUATION_P99_THRESHOLD_MS,
            error_rate_threshold=Env.EVALUATION_ERROR_RATE_THRESHOLD,
            test_data_path=Env.EVALUATION_TEST_DATA_PATH,
            timeout_seconds=Env.EVALUATION_TIMEOUT_SECONDS,
        )

        return cls(
            project_id=Env.AGENT_PROJECT_ID,
            location=Env.AGENT_LOCATION,
            model=Env.AGENT_MODEL,
            temperature=Env.AGENT_TEMPERATURE,
            max_tokens=Env.AGENT_MAX_TOKENS,
            system_prompt=system_prompt,
            display_name=Env.AGENT_DISPLAY_NAME,
            description=Env.AGENT_DESCRIPTION,
            log_level=Env.LOG_LEVEL,
            log_format=Env.LOG_FORMAT,
            session=session_config,
            memory=memory_config,
            observability=observability_config,
            evaluation=evaluation_config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()
