"""Pytest configuration and fixtures for Agent Engine tests."""

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_engine.config import AgentConfig
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
