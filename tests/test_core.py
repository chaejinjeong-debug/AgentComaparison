"""Tests for the refactored core module components."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from agent_engine.core import (
    AgentEngineWrapper,
    AgentResult,
    BaseAgent,
    MessageBuilder,
    PydanticAIAgent,
    ResultProcessor,
    StreamChunk,
)
from agent_engine.exceptions import AgentConfigError


class TestAgentResult:
    """Test suite for AgentResult dataclass."""

    def test_agent_result_default(self) -> None:
        """Test AgentResult with minimal parameters."""
        result = AgentResult(output="test output")

        assert result.output == "test output"
        assert result.response_text == ""
        assert result.tool_calls == []
        assert result.usage == {}
        assert result.metadata == {}

    def test_agent_result_full(self) -> None:
        """Test AgentResult with all parameters."""
        result = AgentResult(
            output={"key": "value"},
            response_text="Hello world",
            tool_calls=[{"tool": "search", "args": {"q": "test"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            metadata={"model": "gemini-2.5-pro"},
        )

        assert result.output == {"key": "value"}
        assert result.response_text == "Hello world"
        assert len(result.tool_calls) == 1
        assert result.usage["prompt_tokens"] == 10
        assert result.metadata["model"] == "gemini-2.5-pro"


class TestMessageBuilder:
    """Test suite for MessageBuilder class."""

    def test_build_simple_message(self) -> None:
        """Test building a simple message."""
        builder = MessageBuilder()
        message = builder.build("Hello world")

        assert message == "Hello world"

    def test_build_message_with_memories(self) -> None:
        """Test building message with memories."""
        builder = MessageBuilder()
        memories = ["User prefers formal language", "User is from Seoul"]
        message = builder.build("Hello", memories=memories)

        assert "[Relevant memories about the user]" in message
        assert "User prefers formal language" in message
        assert "User is from Seoul" in message
        assert message.endswith("Hello")

    def test_build_message_with_context(self) -> None:
        """Test building message with context."""
        builder = MessageBuilder()
        context = {"user_name": "John", "role": "admin"}
        message = builder.build("Hello", context=context)

        assert "[Additional context]" in message
        assert "user_name: John" in message
        assert "role: admin" in message

    def test_build_message_with_session_history(self) -> None:
        """Test building message with session history."""
        builder = MessageBuilder()
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        message = builder.build("How are you?", session_history=history)

        assert "[Previous conversation]" in message
        assert "user: Hi" in message
        assert "assistant: Hello!" in message

    def test_build_message_with_all(self) -> None:
        """Test building message with all components."""
        builder = MessageBuilder()
        message = builder.build(
            "Final question",
            memories=["Memory 1"],
            context={"key": "value"},
            session_history=[{"role": "user", "content": "Previous"}],
        )

        assert "[Previous conversation]" in message
        assert "[Relevant memories about the user]" in message
        assert "[Additional context]" in message
        assert message.endswith("Final question")

    def test_parse_session_events(self) -> None:
        """Test parsing session events."""
        mock_event = MagicMock()
        mock_event.author.value = "user"
        mock_event.content = {"text": "Hello"}

        mock_event2 = MagicMock()
        mock_event2.author.value = "agent"
        mock_event2.content = {"text": "Hi there!"}

        events = [mock_event, mock_event2]
        result = MessageBuilder.parse_session_events(events)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there!"}


class TestResultProcessor:
    """Test suite for ResultProcessor class."""

    def test_process_agent_result(self) -> None:
        """Test processing an AgentResult."""
        processor = ResultProcessor(model_name="gemini-2.5-pro")
        agent_result = AgentResult(
            output={"key": "value"},
            response_text="Test response",
            tool_calls=[{"tool": "search", "args": {}}],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        from datetime import UTC, datetime

        result = processor.process(
            result=agent_result,
            user_id="user123",
            session_id="sess456",
            start_time=datetime.now(UTC),
        )

        assert result["response"] == "Test response"
        assert result["tool_calls"] == [{"tool": "search", "args": {}}]
        assert result["usage"]["prompt_tokens"] == 10
        assert result["metadata"]["model"] == "gemini-2.5-pro"
        assert result["metadata"]["user_id"] == "user123"
        assert result["metadata"]["session_id"] == "sess456"

    def test_process_raw_result(self) -> None:
        """Test processing a raw model result."""
        processor = ResultProcessor(model_name="gemini-2.5-pro")

        mock_result = MagicMock()
        mock_result.output = "Raw response"
        mock_result.tool_calls = []
        mock_result.usage = MagicMock()
        mock_result.usage.prompt_tokens = 5
        mock_result.usage.completion_tokens = 10
        mock_result.usage.total_tokens = 15

        result = processor.process_raw(result=mock_result)

        assert result["response"] == "Raw response"
        assert result["usage"]["prompt_tokens"] == 5

    def test_create_stream_metadata(self) -> None:
        """Test creating stream metadata."""
        metadata = ResultProcessor.create_stream_metadata(
            model_name="gemini-2.5-pro",
            user_id="user123",
            session_id="sess456",
            latency_ms=100.5,
            tool_calls=[{"tool": "test"}],
        )

        assert metadata["model"] == "gemini-2.5-pro"
        assert metadata["user_id"] == "user123"
        assert metadata["latency_ms"] == 100.5
        assert metadata["tool_calls"] == [{"tool": "test"}]


class TestPydanticAIAgent:
    """Test suite for PydanticAIAgent class."""

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        agent = PydanticAIAgent()

        assert agent.model_name == "gemini-2.5-pro"
        assert agent.system_prompt == "You are a helpful AI assistant."
        assert agent.tools == []
        assert agent.temperature == 0.7
        assert agent.max_tokens == 4096
        assert agent.is_initialized is False

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        tools = [lambda x: x]
        agent = PydanticAIAgent(
            model="gemini-2.5-flash",
            system_prompt="Custom prompt",
            tools=tools,
            temperature=0.5,
            max_tokens=2048,
        )

        assert agent.model_name == "gemini-2.5-flash"
        assert agent.system_prompt == "Custom prompt"
        assert len(agent.tools) == 1
        assert agent.temperature == 0.5
        assert agent.max_tokens == 2048

    def test_run_sync_without_init_raises_error(self) -> None:
        """Test that run_sync without initialization raises error."""
        agent = PydanticAIAgent()

        with pytest.raises(AgentConfigError) as exc_info:
            agent.run_sync("Hello")

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_run_async_without_init_raises_error(self) -> None:
        """Test that run_async without initialization raises error."""
        agent = PydanticAIAgent()

        with pytest.raises(AgentConfigError) as exc_info:
            await agent.run_async("Hello")

        assert "not initialized" in str(exc_info.value).lower()

    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    def test_initialize_success(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
    ) -> None:
        """Test successful initialization."""
        agent = PydanticAIAgent(model="gemini-2.5-pro")
        agent.initialize(project="test-project", location="asia-northeast3")

        assert agent.is_initialized is True
        mock_vertexai.assert_called_once_with(
            project="test-project", location="asia-northeast3"
        )

    @patch("vertexai.init")
    @patch("pydantic_ai.providers.google.GoogleProvider")
    @patch("pydantic_ai.models.google.GoogleModel")
    @patch("pydantic_ai.Agent")
    def test_run_sync_success(
        self,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_provider: MagicMock,
        mock_vertexai: MagicMock,
    ) -> None:
        """Test successful sync execution."""
        mock_pydantic_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "Hello response"
        mock_result.tool_calls = []
        mock_pydantic_agent.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_pydantic_agent

        agent = PydanticAIAgent()
        agent.initialize(project="test", location="asia-northeast3")
        result = agent.run_sync("Hello")

        assert isinstance(result, AgentResult)
        assert result.response_text == "Hello response"


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


class TestAgentEngineWrapper:
    """Test suite for AgentEngineWrapper class."""

    def test_init(self) -> None:
        """Test wrapper initialization."""
        mock_agent = MagicMock(spec=BaseAgent)
        mock_agent.model_name = "gemini-2.5-pro"
        mock_agent.tools = []

        wrapper = AgentEngineWrapper(
            agent=mock_agent,
            project="test-project",
            location="asia-northeast3",
        )

        assert wrapper.agent == mock_agent
        assert wrapper.project == "test-project"
        assert wrapper.location == "asia-northeast3"

    def test_query_without_setup_raises_error(self) -> None:
        """Test that query without setup raises error."""
        mock_agent = MagicMock(spec=BaseAgent)
        wrapper = AgentEngineWrapper(agent=mock_agent, project="test")

        with pytest.raises(AgentConfigError) as exc_info:
            wrapper.query(message="Hello")

        assert "not set up" in str(exc_info.value).lower()

    def test_set_up_initializes_agent(self) -> None:
        """Test that set_up initializes the wrapped agent."""
        mock_agent = MagicMock(spec=BaseAgent)
        mock_agent.model_name = "gemini-2.5-pro"
        mock_agent.tools = []

        wrapper = AgentEngineWrapper(agent=mock_agent, project="test")
        wrapper.set_up()

        mock_agent.initialize.assert_called_once()
        assert wrapper._is_setup is True

    def test_query_delegates_to_agent(self) -> None:
        """Test that query delegates to the wrapped agent."""
        mock_agent = MagicMock(spec=BaseAgent)
        mock_agent.model_name = "gemini-2.5-pro"
        mock_agent.tools = []
        mock_result = AgentResult(
            output="test",
            response_text="Hello response",
            tool_calls=[],
            usage={},
        )
        mock_agent.run_sync.return_value = mock_result

        wrapper = AgentEngineWrapper(agent=mock_agent, project="test")
        wrapper.set_up()
        response = wrapper.query(message="Hello")

        mock_agent.run_sync.assert_called_once()
        assert response["response"] == "Hello response"

    def test_get_stats(self) -> None:
        """Test getting wrapper statistics."""
        mock_agent = MagicMock(spec=BaseAgent)
        mock_agent.model_name = "gemini-2.5-pro"
        mock_agent.tools = [lambda x: x]

        wrapper = AgentEngineWrapper(agent=mock_agent, project="test")
        wrapper.set_up()
        stats = wrapper.get_stats()

        assert stats["model"] == "gemini-2.5-pro"
        assert stats["is_setup"] is True
        assert stats["tool_count"] == 1
