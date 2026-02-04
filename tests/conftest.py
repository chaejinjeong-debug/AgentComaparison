"""Pytest configuration and fixtures for Agent Engine tests."""

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


from agent_engine.config import (
    AgentConfig,
    MemoryConfig,
    ObservabilityConfig,
    SessionConfig,
)
from agent_engine.tools import calculate, convert_timezone, get_current_datetime, search


@pytest.fixture
def sample_config() -> AgentConfig:
    """Create a sample AgentConfig for testing."""
    return AgentConfig(
        project_id="test-project",
        location="asia-northeast3",
        model="gemini-2.5-pro",
        temperature=0.7,
        max_tokens=4096,
        system_prompt="You are a helpful test assistant.",
        display_name="test-agent",
        description="Test agent for unit tests",
        log_level="DEBUG",
        log_format="text",
    )


@pytest.fixture
def session_config() -> SessionConfig:
    """Create a sample SessionConfig for testing."""
    return SessionConfig(
        enabled=True,
        default_ttl_seconds=3600,  # 1 hour for tests
        max_events_per_session=100,
    )


@pytest.fixture
def memory_config() -> MemoryConfig:
    """Create a sample MemoryConfig for testing."""
    return MemoryConfig(
        enabled=True,
        auto_generate=True,
        max_memories_per_user=100,
        similarity_threshold=0.5,
    )


@pytest.fixture
def observability_config() -> ObservabilityConfig:
    """Create a sample ObservabilityConfig for testing."""
    return ObservabilityConfig(
        tracing_enabled=False,  # Disable for tests
        logging_enabled=True,
        metrics_enabled=True,
        sample_rate=1.0,
    )


@pytest.fixture
def full_config(
    session_config: SessionConfig,
    memory_config: MemoryConfig,
    observability_config: ObservabilityConfig,
) -> AgentConfig:
    """Create a full AgentConfig with Phase 2 features for testing."""
    return AgentConfig(
        project_id="test-project",
        location="asia-northeast3",
        model="gemini-2.5-pro",
        temperature=0.7,
        max_tokens=4096,
        system_prompt="You are a helpful test assistant.",
        display_name="test-agent",
        description="Test agent for unit tests",
        log_level="DEBUG",
        log_format="text",
        session=session_config,
        memory=memory_config,
        observability=observability_config,
    )


@pytest.fixture
def mock_vertexai() -> MagicMock:
    """Mock VertexAI initialization."""
    with patch("vertexai.init") as mock:
        yield mock


@pytest.fixture
def mock_pydantic_agent() -> MagicMock:
    """Mock Pydantic AI Agent."""
    mock_result = MagicMock()
    mock_result.output = "This is a test response."
    mock_result.tool_calls = []
    mock_result.usage = MagicMock(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )

    mock_agent = MagicMock()
    mock_agent.run_sync.return_value = mock_result
    mock_agent.run.return_value = mock_result

    return mock_agent


@pytest.fixture
def mock_google_provider() -> MagicMock:
    """Mock GoogleProvider."""
    with patch("pydantic_ai.providers.google.GoogleProvider") as mock:
        yield mock


@pytest.fixture
def mock_google_model() -> MagicMock:
    """Mock GoogleModel."""
    with patch("pydantic_ai.models.google.GoogleModel") as mock:
        yield mock


@pytest.fixture
def mock_agent_class(mock_pydantic_agent: MagicMock) -> MagicMock:
    """Mock Pydantic AI Agent class."""
    with patch("pydantic_ai.Agent") as mock:
        mock.return_value = mock_pydantic_agent
        yield mock


@pytest.fixture
def sample_tools() -> list:
    """Get sample tools for testing."""
    return [search, calculate, get_current_datetime, convert_timezone]


@pytest.fixture
def env_vars() -> dict[str, str]:
    """Sample environment variables for testing."""
    return {
        "AGENT_PROJECT_ID": "test-project",
        "AGENT_LOCATION": "asia-northeast3",
        "AGENT_MODEL": "gemini-2.5-pro",
        "AGENT_TEMPERATURE": "0.7",
        "AGENT_MAX_TOKENS": "4096",
        "AGENT_SYSTEM_PROMPT": "You are a test assistant.",
        "LOG_LEVEL": "DEBUG",
        "LOG_FORMAT": "text",
        # Phase 2 environment variables
        "SESSION_ENABLED": "true",
        "SESSION_TTL_SECONDS": "3600",
        "MEMORY_ENABLED": "true",
        "MEMORY_AUTO_GENERATE": "true",
        "TRACING_ENABLED": "false",
        "METRICS_ENABLED": "true",
    }


@pytest.fixture
def mock_env(env_vars: dict[str, str]) -> Any:
    """Mock environment variables."""
    with patch.dict(os.environ, env_vars, clear=False):
        yield


class MockAgentResult:
    """Mock result object for Pydantic AI Agent."""

    def __init__(
        self,
        output: str = "Test response",
        tool_calls: list | None = None,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
    ):
        self.output = output
        self.tool_calls = tool_calls or []
        self.usage = MagicMock(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


@pytest.fixture
def mock_agent_result() -> MockAgentResult:
    """Create a mock agent result."""
    return MockAgentResult()


# Phase 2 fixtures


@pytest.fixture
def session_manager(session_config: SessionConfig):
    """Create a SessionManager for testing."""
    from agent_engine.sessions import SessionManager

    return SessionManager(config=session_config)


@pytest.fixture
def memory_manager(memory_config: MemoryConfig):
    """Create a MemoryManager for testing."""
    from agent_engine.memory import MemoryManager

    return MemoryManager(config=memory_config)


@pytest.fixture
def memory_retriever():
    """Create a MemoryRetriever for testing."""
    from agent_engine.memory import MemoryRetriever

    return MemoryRetriever(similarity_threshold=0.5)


@pytest.fixture
def metrics_manager(observability_config: ObservabilityConfig):
    """Create a MetricsManager for testing."""
    from agent_engine.observability import MetricsManager

    manager = MetricsManager(config=observability_config)
    manager.setup()
    return manager


@pytest.fixture
def tracing_manager(observability_config: ObservabilityConfig):
    """Create a TracingManager for testing."""
    from agent_engine.observability import TracingManager

    return TracingManager(config=observability_config)


@pytest.fixture
def logging_manager(observability_config: ObservabilityConfig):
    """Create a LoggingManager for testing."""
    from agent_engine.observability import LoggingManager

    manager = LoggingManager(config=observability_config)
    return manager
