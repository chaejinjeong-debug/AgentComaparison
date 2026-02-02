"""Configuration management for the Agent Engine platform."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


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
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()
