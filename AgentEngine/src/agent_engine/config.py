"""Configuration management for the Agent Engine platform."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


class SessionConfig(BaseModel):
    """Configuration for Session Management.

    Attributes:
        enabled: Whether session management is enabled
        default_ttl_seconds: Default session TTL in seconds (default: 24 hours)
        max_events_per_session: Maximum events to store per session
    """

    enabled: bool = Field(default=True, description="Enable session management")
    default_ttl_seconds: int = Field(
        default=86400, ge=60, description="Default session TTL (24 hours)"
    )
    max_events_per_session: int = Field(
        default=1000, ge=10, description="Max events per session"
    )


class MemoryConfig(BaseModel):
    """Configuration for Memory Bank.

    Attributes:
        enabled: Whether memory bank is enabled
        auto_generate: Automatically extract memories from conversations
        max_memories_per_user: Maximum memories to store per user
        similarity_threshold: Minimum similarity score for memory retrieval
    """

    enabled: bool = Field(default=True, description="Enable memory bank")
    auto_generate: bool = Field(default=True, description="Auto-generate memories from sessions")
    max_memories_per_user: int = Field(
        default=1000, ge=10, description="Max memories per user"
    )
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Similarity threshold for retrieval"
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
        default=1.0, ge=0.0, le=1.0, description="Trace sampling rate"
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
    location: str = Field(default="asia-northeast3", description="GCP region")
    model: str = Field(default="gemini-2.5-pro", description="Gemini model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="System prompt for the agent",
    )
    display_name: str = Field(default="pydantic-ai-agent", description="Agent display name")
    description: str = Field(
        default="Pydantic AI Agent on VertexAI Agent Engine",
        description="Agent description",
    )
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # Phase 2 configurations
    session: SessionConfig = Field(default_factory=SessionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        valid_formats = {"json", "text"}
        v_lower = v.lower()
        if v_lower not in valid_formats:
            raise ValueError(f"log_format must be one of {valid_formats}")
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

        system_prompt = os.getenv("AGENT_SYSTEM_PROMPT", "You are a helpful AI assistant.")
        system_prompt_file = os.getenv("AGENT_SYSTEM_PROMPT_FILE")

        if system_prompt_file and Path(system_prompt_file).exists():
            system_prompt = Path(system_prompt_file).read_text(encoding="utf-8").strip()

        # Session configuration
        session_config = SessionConfig(
            enabled=os.getenv("SESSION_ENABLED", "true").lower() == "true",
            default_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "86400")),
            max_events_per_session=int(os.getenv("SESSION_MAX_EVENTS", "1000")),
        )

        # Memory configuration
        memory_config = MemoryConfig(
            enabled=os.getenv("MEMORY_ENABLED", "true").lower() == "true",
            auto_generate=os.getenv("MEMORY_AUTO_GENERATE", "true").lower() == "true",
            max_memories_per_user=int(os.getenv("MEMORY_MAX_PER_USER", "1000")),
            similarity_threshold=float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.7")),
        )

        # Observability configuration
        observability_config = ObservabilityConfig(
            tracing_enabled=os.getenv("TRACING_ENABLED", "true").lower() == "true",
            logging_enabled=os.getenv("LOGGING_ENABLED", "true").lower() == "true",
            metrics_enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            sample_rate=float(os.getenv("TRACE_SAMPLE_RATE", "1.0")),
        )

        return cls(
            project_id=os.getenv("AGENT_PROJECT_ID", ""),
            location=os.getenv("AGENT_LOCATION", "asia-northeast3"),
            model=os.getenv("AGENT_MODEL", "gemini-2.5-pro"),
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "4096")),
            system_prompt=system_prompt,
            display_name=os.getenv("AGENT_DISPLAY_NAME", "pydantic-ai-agent"),
            description=os.getenv(
                "AGENT_DESCRIPTION", "Pydantic AI Agent on VertexAI Agent Engine"
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "json"),
            session=session_config,
            memory=memory_config,
            observability=observability_config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()
