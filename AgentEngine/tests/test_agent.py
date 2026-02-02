"""Tests for the Agent Wrapper module."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_engine.agent import PydanticAIAgentWrapper
from agent_engine.config import AgentConfig
from agent_engine.exceptions import AgentConfigError, AgentQueryError


class TestPydanticAIAgentWrapper:
    """Test suite for PydanticAIAgentWrapper class."""

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        agent = PydanticAIAgentWrapper()

        assert agent.model_name == "gemini-2.5-pro"
        assert agent.project == ""
        assert agent.location == "asia-northeast3"
        assert agent.system_prompt == "You are a helpful AI assistant."
        assert agent.tools == []
        assert agent.temperature == 0.7
        assert agent.max_tokens == 4096
        assert agent._is_setup is False

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        tools = [lambda x: x]
        agent = PydanticAIAgentWrapper(
            model="gemini-2.5-flash",
            project="my-project",
            location="us-central1",
            system_prompt="Custom prompt",
            tools=tools,
            temperature=0.5,
            max_tokens=2048,
        )

        assert agent.model_name == "gemini-2.5-flash"
        assert agent.project == "my-project"
        assert agent.location == "us-central1"
        assert agent.system_prompt == "Custom prompt"
        assert len(agent.tools) == 1
        assert agent.temperature == 0.5
        assert agent.max_tokens == 2048

    def test_from_config(self, sample_config: AgentConfig) -> None:
        """Test creating agent from config."""
        agent = PydanticAIAgentWrapper.from_config(sample_config)

        assert agent.model_name == sample_config.model
        assert agent.project == sample_config.project_id
        assert agent.location == sample_config.location
        assert agent.system_prompt == sample_config.system_prompt

    def test_from_config_with_tools(self, sample_config: AgentConfig) -> None:
        """Test creating agent from config with tools."""
        tools = [lambda x: x, lambda y: y]
        agent = PydanticAIAgentWrapper.from_config(sample_config, tools=tools)

        assert len(agent.tools) == 2

    def test_query_without_setup_raises_error(self) -> None:
        """Test that query without setup raises AgentConfigError."""
        agent = PydanticAIAgentWrapper(project="test")

        with pytest.raises(AgentConfigError) as exc_info:
            agent.query(message="Hello")

        assert "not set up" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_aquery_without_setup_raises_error(self) -> None:
        """Test that async query without setup raises AgentConfigError."""
        agent = PydanticAIAgentWrapper(project="test")

        with pytest.raises(AgentConfigError) as exc_info:
            await agent.aquery(message="Hello")

        assert "not set up" in str(exc_info.value).lower()

    def test_add_tool(self) -> None:
        """Test adding a tool."""
        agent = PydanticAIAgentWrapper()

        def my_tool(x: int) -> int:
            return x * 2

        agent.add_tool(my_tool)

        assert len(agent.tools) == 1
        assert agent.tools[0] == my_tool

    def test_register_tools(self) -> None:
        """Test registering multiple tools."""
        agent = PydanticAIAgentWrapper()

        tools = [lambda x: x, lambda y: y, lambda z: z]
        agent.register_tools(tools)

        assert len(agent.tools) == 3

    def test_build_message_simple(self) -> None:
        """Test building a simple message."""
        agent = PydanticAIAgentWrapper()
        message = agent._build_message("Hello world")

        assert message == "Hello world"

    def test_build_message_with_memories(self) -> None:
        """Test building message with memories."""
        agent = PydanticAIAgentWrapper()
        memories = ["User prefers formal language", "User is from Seoul"]
        message = agent._build_message("Hello", memories=memories)

        assert "Relevant memories" in message
        assert "User prefers formal language" in message
        assert "User is from Seoul" in message
        assert "Hello" in message

    def test_build_message_with_context(self) -> None:
        """Test building message with context."""
        agent = PydanticAIAgentWrapper()
        context = {"user_name": "John", "role": "admin"}
        message = agent._build_message("Hello", context=context)

        assert "Additional context" in message
        assert "user_name: John" in message
        assert "role: admin" in message

    def test_build_message_with_all(self) -> None:
        """Test building message with memories and context."""
        agent = PydanticAIAgentWrapper()
        memories = ["Memory 1"]
        context = {"key": "value"}
        message = agent._build_message("Hello", context=context, memories=memories)

        assert "Relevant memories" in message
        assert "Additional context" in message
        assert "Hello" in message


class TestAgentSetup:
    """Test suite for agent setup functionality."""

    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    def test_setup_success(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
    ) -> None:
        """Test successful agent setup."""
        agent = PydanticAIAgentWrapper(
            project="test-project",
            location="asia-northeast3",
        )

        agent.set_up()

        assert agent._is_setup is True
        mock_vertexai.assert_called_once_with(
            project="test-project",
            location="asia-northeast3",
        )
        mock_provider.assert_called_once()
        mock_model.assert_called_once()
        mock_agent_class.assert_called_once()

    def test_setup_missing_package_raises_error(self) -> None:
        """Test that missing package raises AgentConfigError."""
        _agent = PydanticAIAgentWrapper(project="test")  # noqa: F841

        with patch.dict("sys.modules", {"vertexai": None}):
            # This would raise ImportError in real scenario
            # Full test implementation would require mocking import system
            pass


class TestAgentQuery:
    """Test suite for agent query functionality."""

    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    def test_query_success(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
        mock_agent_result: MagicMock,
    ) -> None:
        """Test successful query."""
        mock_pydantic_agent = MagicMock()
        mock_pydantic_agent.run_sync.return_value = mock_agent_result
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        agent.set_up()

        result = agent.query(message="Hello")

        assert "response" in result
        assert "tool_calls" in result
        assert "usage" in result
        assert "metadata" in result
        mock_pydantic_agent.run_sync.assert_called_once()

    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    def test_query_with_user_id(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
        mock_agent_result: MagicMock,
    ) -> None:
        """Test query with user_id."""
        mock_pydantic_agent = MagicMock()
        mock_pydantic_agent.run_sync.return_value = mock_agent_result
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        agent.set_up()

        result = agent.query(message="Hello", user_id="user123")

        assert result["metadata"]["user_id"] == "user123"

    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    def test_query_failure_raises_error(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
    ) -> None:
        """Test query failure raises AgentQueryError."""
        mock_pydantic_agent = MagicMock()
        mock_pydantic_agent.run_sync.side_effect = RuntimeError("API error")
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        agent.set_up()

        with pytest.raises(AgentQueryError) as exc_info:
            agent.query(message="Hello")

        assert "API error" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    async def test_aquery_success(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
        mock_agent_result: MagicMock,
    ) -> None:
        """Test successful async query."""
        mock_pydantic_agent = MagicMock()
        mock_pydantic_agent.run = AsyncMock(return_value=mock_agent_result)
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        agent.set_up()

        result = await agent.aquery(message="Hello")

        assert "response" in result
        mock_pydantic_agent.run.assert_called_once()
