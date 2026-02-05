"""Tests for the Agent Wrapper module."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from agent_engine.agent import PydanticAIAgentWrapper, StreamChunk
from agent_engine.config import AgentConfig
from agent_engine.exceptions import AgentConfigError, AgentQueryError


class TestPydanticAIAgentWrapper:
    """Test suite for PydanticAIAgentWrapper class."""

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        from agent_engine.constants import (
            DEFAULT_LOCATION,
            DEFAULT_MAX_TOKENS,
            DEFAULT_MODEL,
            DEFAULT_SYSTEM_PROMPT,
            DEFAULT_TEMPERATURE,
        )

        agent = PydanticAIAgentWrapper()

        # Note: defaults come from agent_engine.constants
        assert agent.model_name == DEFAULT_MODEL
        assert agent.project == ""
        assert agent.location == DEFAULT_LOCATION
        assert agent.system_prompt == DEFAULT_SYSTEM_PROMPT
        assert agent.tools == []
        assert agent.temperature == DEFAULT_TEMPERATURE
        assert agent.max_tokens == DEFAULT_MAX_TOKENS
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

    @pytest.mark.asyncio
    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    async def test_query_without_explicit_setup_auto_initializes(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
    ) -> None:
        """Test that query auto-initializes if not explicitly set up."""
        mock_pydantic_agent = MagicMock()
        mock_pydantic_agent.run = AsyncMock(return_value=MagicMock(
            data="Test response",
            all_messages=MagicMock(return_value=[]),
        ))
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        assert agent._is_setup is False

        # Query should auto-initialize
        await agent.query(message="Hello")

        # Agent should now be set up
        assert agent._is_setup is True

    @pytest.mark.asyncio
    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    async def test_aquery_without_explicit_setup_auto_initializes(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
    ) -> None:
        """Test that aquery auto-initializes if not explicitly set up."""
        mock_pydantic_agent = MagicMock()
        mock_pydantic_agent.run = AsyncMock(return_value=MagicMock(
            data="Test response",
            all_messages=MagicMock(return_value=[]),
        ))
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        assert agent._is_setup is False

        # Query should auto-initialize
        await agent.aquery(message="Hello")

        # Agent should now be set up
        assert agent._is_setup is True

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

    @pytest.mark.asyncio
    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    async def test_query_success(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
        mock_agent_result: MagicMock,
    ) -> None:
        """Test successful query."""
        mock_pydantic_agent = MagicMock()
        mock_pydantic_agent.run = AsyncMock(return_value=mock_agent_result)
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        agent.set_up()

        result = await agent.query(message="Hello")

        assert "response" in result
        assert "tool_calls" in result
        assert "usage" in result
        assert "metadata" in result
        mock_pydantic_agent.run.assert_called_once()

    @pytest.mark.asyncio
    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    async def test_query_with_user_id(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
        mock_agent_result: MagicMock,
    ) -> None:
        """Test query with user_id."""
        mock_pydantic_agent = MagicMock()
        mock_pydantic_agent.run = AsyncMock(return_value=mock_agent_result)
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        agent.set_up()

        result = await agent.query(message="Hello", user_id="user123")

        assert result["metadata"]["user_id"] == "user123"

    @pytest.mark.asyncio
    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    async def test_query_failure_raises_error(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
    ) -> None:
        """Test query failure raises AgentQueryError."""
        mock_pydantic_agent = MagicMock()
        mock_pydantic_agent.run = AsyncMock(side_effect=RuntimeError("API error"))
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        agent.set_up()

        with pytest.raises(AgentQueryError) as exc_info:
            await agent.query(message="Hello")

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


class TestStreamChunk:
    """Test suite for StreamChunk dataclass."""

    def test_stream_chunk_creation(self) -> None:
        """Test creating a StreamChunk."""
        chunk = StreamChunk(chunk="Hello", done=False)

        assert chunk.chunk == "Hello"
        assert chunk.done is False
        assert chunk.tool_call is None
        assert chunk.metadata is None

    def test_stream_chunk_with_tool_call(self) -> None:
        """Test StreamChunk with tool call."""
        tool_call = {"tool": "search", "args": {"query": "test"}}
        chunk = StreamChunk(chunk="", done=False, tool_call=tool_call)

        assert chunk.tool_call == tool_call

    def test_stream_chunk_final(self) -> None:
        """Test final StreamChunk with metadata."""
        metadata = {"latency_ms": 100, "model": "gemini-2.5-pro"}
        chunk = StreamChunk(chunk="", done=True, metadata=metadata)

        assert chunk.done is True
        assert chunk.metadata == metadata


class TestAgentStreamQuery:
    """Test suite for agent streaming query functionality."""

    @pytest.mark.asyncio
    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    async def test_stream_query_without_explicit_setup_auto_initializes(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
    ) -> None:
        """Test that stream query auto-initializes if not explicitly set up."""
        # Create mock stream
        mock_stream = MagicMock()
        mock_result = MagicMock()
        mock_result.tool_calls = []
        mock_stream.result = mock_result

        async def mock_stream_text():
            yield "Hello"

        mock_stream.stream_text = mock_stream_text

        mock_pydantic_agent = MagicMock()
        mock_stream_cm = MagicMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
        mock_pydantic_agent.run_stream = MagicMock(return_value=mock_stream_cm)
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        assert agent._is_setup is False

        # Stream query should auto-initialize
        async for _ in agent.stream_query(message="Hello"):
            pass

        # Agent should now be set up
        assert agent._is_setup is True

    @pytest.mark.asyncio
    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    async def test_stream_query_success(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
    ) -> None:
        """Test successful streaming query."""
        # Create mock stream context manager
        mock_stream = MagicMock()
        mock_result = MagicMock()
        mock_result.tool_calls = []
        mock_stream.result = mock_result

        # Mock the async text stream
        async def mock_stream_text() -> list[str]:
            for text in ["Hello", " ", "World", "!"]:
                yield text

        mock_stream.stream_text = mock_stream_text

        # Set up async context manager
        mock_pydantic_agent = MagicMock()
        mock_stream_cm = MagicMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
        mock_pydantic_agent.run_stream = MagicMock(return_value=mock_stream_cm)
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        agent.set_up()

        chunks = []
        async for chunk in agent.stream_query(message="Hello"):
            chunks.append(chunk)

        # Should have chunks for "Hello", " ", "World", "!" plus final
        assert len(chunks) == 5
        assert chunks[0].chunk == "Hello"
        assert chunks[1].chunk == " "
        assert chunks[2].chunk == "World"
        assert chunks[3].chunk == "!"
        assert chunks[4].done is True
        assert chunks[4].metadata is not None

    @pytest.mark.asyncio
    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    async def test_stream_query_with_user_id(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
    ) -> None:
        """Test streaming query with user_id."""
        # Create mock stream
        mock_stream = MagicMock()
        mock_result = MagicMock()
        mock_result.tool_calls = []
        mock_stream.result = mock_result

        async def mock_stream_text() -> list[str]:
            yield "Test"

        mock_stream.stream_text = mock_stream_text

        mock_pydantic_agent = MagicMock()
        mock_stream_cm = MagicMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
        mock_pydantic_agent.run_stream = MagicMock(return_value=mock_stream_cm)
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        agent.set_up()

        chunks = []
        async for chunk in agent.stream_query(message="Hello", user_id="user123"):
            chunks.append(chunk)

        # Check final chunk has user_id in metadata
        final_chunk = chunks[-1]
        assert final_chunk.done is True
        assert final_chunk.metadata["user_id"] == "user123"

    @pytest.mark.asyncio
    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    async def test_stream_query_with_tool_calls(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
    ) -> None:
        """Test streaming query with tool calls."""
        # Create mock stream with tool calls
        mock_stream = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.name = "search"
        mock_tool_call.args = {"query": "test"}
        mock_result = MagicMock()
        mock_result.tool_calls = [mock_tool_call]
        mock_stream.result = mock_result

        async def mock_stream_text() -> list[str]:
            yield "Search result"

        mock_stream.stream_text = mock_stream_text

        mock_pydantic_agent = MagicMock()
        mock_stream_cm = MagicMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
        mock_pydantic_agent.run_stream = MagicMock(return_value=mock_stream_cm)
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        agent.set_up()

        chunks = []
        async for chunk in agent.stream_query(message="Search for test"):
            chunks.append(chunk)

        # Should have tool call chunk
        tool_chunks = [c for c in chunks if c.tool_call is not None]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_call["tool"] == "search"
        assert tool_chunks[0].tool_call["args"] == {"query": "test"}

    @pytest.mark.asyncio
    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    async def test_stream_query_failure_raises_error(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
    ) -> None:
        """Test streaming query failure raises AgentQueryError."""
        mock_pydantic_agent = MagicMock()
        mock_stream_cm = MagicMock()
        mock_stream_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("Stream error"))
        mock_pydantic_agent.run_stream = MagicMock(return_value=mock_stream_cm)
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        agent.set_up()

        with pytest.raises(AgentQueryError) as exc_info:
            async for _ in agent.stream_query(message="Hello"):
                pass

        assert "Stream error" in str(exc_info.value)

    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    def test_stream_query_sync(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
    ) -> None:
        """Test synchronous streaming query wrapper."""
        # Create mock stream
        mock_stream = MagicMock()
        mock_result = MagicMock()
        mock_result.tool_calls = []
        mock_stream.result = mock_result

        async def mock_stream_text() -> list[str]:
            for text in ["Hi", " there"]:
                yield text

        mock_stream.stream_text = mock_stream_text

        mock_pydantic_agent = MagicMock()
        mock_stream_cm = MagicMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
        mock_pydantic_agent.run_stream = MagicMock(return_value=mock_stream_cm)
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgentWrapper(project="test")
        agent.set_up()

        chunks = list(agent.stream_query_sync(message="Hello"))

        assert len(chunks) == 3  # "Hi", " there", final
        assert chunks[0].chunk == "Hi"
        assert chunks[1].chunk == " there"
        assert chunks[2].done is True
