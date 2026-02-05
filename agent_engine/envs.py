"""Environment variable management for the Agent Engine platform.

This module centralizes all environment variable access to ensure consistency
and provide better testability through the override mechanism.

Usage:
    from agent_engine.envs import Env

    # Access environment variables as properties
    project_id = Env.AGENT_PROJECT_ID
    model = Env.AGENT_MODEL

    # For testing - override environment variables
    Env.override(AGENT_PROJECT_ID="test-project")
    Env.clear_overrides()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

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
)


class _EnvVars:
    """Environment variable manager with caching and override support.

    This class provides a centralized way to access environment variables
    with proper defaults and type conversion. It supports overriding values
    for testing purposes.

    All environment variables are accessed via properties to ensure
    IDE autocompletion support and to centralize default value management.
    """

    _overrides: ClassVar[dict[str, str | None]] = {}

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get(self, key: str, default: str) -> str:
        """Get string environment variable with default.

        Args:
            key: Environment variable name
            default: Default value if not set

        Returns:
            Environment variable value or default
        """
        if key in self._overrides:
            value = self._overrides[key]
            return value if value is not None else default
        return os.getenv(key, default)

    def _get_bool(self, key: str, default: bool) -> bool:
        """Get boolean environment variable with default.

        Args:
            key: Environment variable name
            default: Default value if not set

        Returns:
            Boolean value (true/1/yes = True, false/0/no = False)
        """
        value = self._get(key, str(default).lower())
        return value.lower() in ("true", "1", "yes")

    def _get_int(self, key: str, default: int) -> int:
        """Get integer environment variable with default.

        Args:
            key: Environment variable name
            default: Default value if not set

        Returns:
            Integer value
        """
        value = self._get(key, str(default))
        return int(value)

    def _get_float(self, key: str, default: float) -> float:
        """Get float environment variable with default.

        Args:
            key: Environment variable name
            default: Default value if not set

        Returns:
            Float value
        """
        value = self._get(key, str(default))
        return float(value)

    # =========================================================================
    # Core Agent Settings
    # =========================================================================

    @property
    def AGENT_PROJECT_ID(self) -> str:
        """GCP project ID for the agent."""
        return self._get("AGENT_PROJECT_ID", "")

    @property
    def GOOGLE_CLOUD_PROJECT(self) -> str:
        """GCP project ID (set by Agent Engine runtime)."""
        return self._get("GOOGLE_CLOUD_PROJECT", "")

    @property
    def AGENT_LOCATION(self) -> str:
        """GCP region for the agent."""
        return self._get("AGENT_LOCATION", DEFAULT_LOCATION)

    @property
    def AGENT_MODEL(self) -> str:
        """Gemini model name."""
        return self._get("AGENT_MODEL", DEFAULT_MODEL)

    @property
    def AGENT_TEMPERATURE(self) -> float:
        """Model temperature setting."""
        return self._get_float("AGENT_TEMPERATURE", DEFAULT_TEMPERATURE)

    @property
    def AGENT_MAX_TOKENS(self) -> int:
        """Maximum output tokens."""
        return self._get_int("AGENT_MAX_TOKENS", DEFAULT_MAX_TOKENS)

    @property
    def AGENT_SYSTEM_PROMPT(self) -> str:
        """System prompt for the agent."""
        return self._get("AGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    @property
    def AGENT_SYSTEM_PROMPT_FILE(self) -> str:
        """Path to system prompt file."""
        return self._get("AGENT_SYSTEM_PROMPT_FILE", "")

    @property
    def AGENT_DISPLAY_NAME(self) -> str:
        """Agent display name for Agent Engine."""
        return self._get("AGENT_DISPLAY_NAME", DEFAULT_DISPLAY_NAME)

    @property
    def AGENT_DESCRIPTION(self) -> str:
        """Agent description."""
        return self._get("AGENT_DESCRIPTION", DEFAULT_DESCRIPTION)

    @property
    def AGENT_ENGINE_ID(self) -> str:
        """Deployed Agent Engine ID."""
        return self._get("AGENT_ENGINE_ID", "")

    # =========================================================================
    # Logging Settings
    # =========================================================================

    @property
    def LOG_LEVEL(self) -> str:
        """Logging level."""
        return self._get("LOG_LEVEL", DEFAULT_LOG_LEVEL)

    @property
    def LOG_FORMAT(self) -> str:
        """Logging format (json or text)."""
        return self._get("LOG_FORMAT", DEFAULT_LOG_FORMAT)

    # =========================================================================
    # Session Settings
    # =========================================================================

    @property
    def SESSION_ENABLED(self) -> bool:
        """Whether session management is enabled."""
        return self._get_bool("SESSION_ENABLED", True)

    @property
    def SESSION_TTL_SECONDS(self) -> int:
        """Default session TTL in seconds."""
        return self._get_int("SESSION_TTL_SECONDS", DEFAULT_SESSION_TTL_SECONDS)

    @property
    def SESSION_MAX_EVENTS(self) -> int:
        """Maximum events per session."""
        return self._get_int("SESSION_MAX_EVENTS", DEFAULT_MAX_EVENTS_PER_SESSION)

    @property
    def SESSION_BACKEND(self) -> str:
        """Session storage backend (in_memory or vertex_ai)."""
        return self._get("SESSION_BACKEND", "in_memory")

    @property
    def SESSION_AGENT_ENGINE_ID(self) -> str | None:
        """Agent Engine ID for session backend."""
        value = self._get("SESSION_AGENT_ENGINE_ID", "")
        return value if value else None

    # =========================================================================
    # Memory Settings
    # =========================================================================

    @property
    def MEMORY_ENABLED(self) -> bool:
        """Whether memory bank is enabled."""
        return self._get_bool("MEMORY_ENABLED", True)

    @property
    def MEMORY_AUTO_GENERATE(self) -> bool:
        """Whether to auto-generate memories from sessions."""
        return self._get_bool("MEMORY_AUTO_GENERATE", True)

    @property
    def MEMORY_MAX_PER_USER(self) -> int:
        """Maximum memories per user."""
        return self._get_int("MEMORY_MAX_PER_USER", DEFAULT_MAX_MEMORIES_PER_USER)

    @property
    def MEMORY_SIMILARITY_THRESHOLD(self) -> float:
        """Similarity threshold for memory retrieval."""
        return self._get_float("MEMORY_SIMILARITY_THRESHOLD", DEFAULT_SIMILARITY_THRESHOLD)

    @property
    def MEMORY_BACKEND(self) -> str:
        """Memory storage backend (in_memory or vertex_ai)."""
        return self._get("MEMORY_BACKEND", "in_memory")

    @property
    def MEMORY_AGENT_ENGINE_ID(self) -> str | None:
        """Agent Engine ID for memory backend."""
        value = self._get("MEMORY_AGENT_ENGINE_ID", "")
        return value if value else None

    # =========================================================================
    # Observability Settings
    # =========================================================================

    @property
    def TRACING_ENABLED(self) -> bool:
        """Whether Cloud Trace is enabled."""
        return self._get_bool("TRACING_ENABLED", True)

    @property
    def LOGGING_ENABLED(self) -> bool:
        """Whether structured logging is enabled."""
        return self._get_bool("LOGGING_ENABLED", True)

    @property
    def METRICS_ENABLED(self) -> bool:
        """Whether metrics collection is enabled."""
        return self._get_bool("METRICS_ENABLED", True)

    @property
    def TRACE_SAMPLE_RATE(self) -> float:
        """Trace sampling rate (0.0-1.0)."""
        return self._get_float("TRACE_SAMPLE_RATE", DEFAULT_TRACE_SAMPLE_RATE)

    # =========================================================================
    # Evaluation Settings
    # =========================================================================

    @property
    def EVALUATION_ENABLED(self) -> bool:
        """Whether evaluation is enabled."""
        return self._get_bool("EVALUATION_ENABLED", True)

    @property
    def EVALUATION_QUALITY_THRESHOLD(self) -> float:
        """Minimum quality threshold for evaluation."""
        return self._get_float("EVALUATION_QUALITY_THRESHOLD", DEFAULT_QUALITY_THRESHOLD)

    @property
    def EVALUATION_P50_THRESHOLD_MS(self) -> float:
        """P50 latency threshold in milliseconds."""
        return self._get_float("EVALUATION_P50_THRESHOLD_MS", DEFAULT_P50_THRESHOLD_MS)

    @property
    def EVALUATION_P99_THRESHOLD_MS(self) -> float:
        """P99 latency threshold in milliseconds."""
        return self._get_float("EVALUATION_P99_THRESHOLD_MS", DEFAULT_P99_THRESHOLD_MS)

    @property
    def EVALUATION_ERROR_RATE_THRESHOLD(self) -> float:
        """Maximum error rate threshold."""
        return self._get_float("EVALUATION_ERROR_RATE_THRESHOLD", DEFAULT_ERROR_RATE_THRESHOLD)

    @property
    def EVALUATION_TEST_DATA_PATH(self) -> str:
        """Path to evaluation test data."""
        return self._get("EVALUATION_TEST_DATA_PATH", DEFAULT_TEST_DATA_PATH)

    @property
    def EVALUATION_TIMEOUT_SECONDS(self) -> float:
        """Evaluation timeout in seconds."""
        return self._get_float("EVALUATION_TIMEOUT_SECONDS", DEFAULT_EVALUATION_TIMEOUT_SECONDS)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_system_prompt(self) -> str:
        """Get system prompt, loading from file if specified.

        Returns:
            System prompt string (from file or environment variable)
        """
        prompt_file = self.AGENT_SYSTEM_PROMPT_FILE
        if prompt_file and Path(prompt_file).exists():
            return Path(prompt_file).read_text(encoding="utf-8").strip()
        return self.AGENT_SYSTEM_PROMPT

    def get_project_id(self) -> str:
        """Get project ID with fallback to GOOGLE_CLOUD_PROJECT.

        Returns:
            Project ID string
        """
        return self.AGENT_PROJECT_ID or self.GOOGLE_CLOUD_PROJECT

    # =========================================================================
    # Test Support Methods
    # =========================================================================

    @classmethod
    def override(cls, **kwargs: str | None) -> None:
        """Override environment variables for testing.

        Args:
            **kwargs: Environment variable name-value pairs to override.
                      Use None to simulate unset variables.

        Example:
            Env.override(AGENT_PROJECT_ID="test-project", AGENT_MODEL="gemini-2.5-pro")
        """
        cls._overrides.update(kwargs)

    @classmethod
    def clear_overrides(cls) -> None:
        """Clear all environment variable overrides."""
        cls._overrides.clear()


# Singleton instance
Env = _EnvVars()
