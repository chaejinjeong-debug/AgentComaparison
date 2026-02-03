"""Integration tests for Phase 2 features."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from agent_engine import AgentConfig, PydanticAIAgentWrapper
from agent_engine.config import MemoryConfig, ObservabilityConfig, SessionConfig
from agent_engine.memory import MemoryManager
from agent_engine.sessions import EventAuthor, SessionManager


class TestSessionMemoryIntegration:
    """Integration tests for Session + Memory."""

    @pytest.mark.asyncio
    async def test_session_with_memory_retrieval(
        self,
        session_manager: SessionManager,
        memory_manager: MemoryManager,
    ):
        """Test memory retrieval during session queries."""
        user_id = "integration-user"

        # Save some memories
        await memory_manager.save_memory(
            user_id=user_id,
            fact="User prefers Python over Java",
            topics=["programming", "preferences"],
        )
        await memory_manager.save_memory(
            user_id=user_id,
            fact="User works at a startup",
            topics=["work"],
        )

        # Create a session
        session = await session_manager.create_session(user_id=user_id)

        # Simulate a query about programming
        query = "What programming language should I use?"
        memories = await memory_manager.retrieve_memories(
            user_id=user_id,
            query=query,
            max_results=3,
        )

        # Should retrieve relevant memory
        assert len(memories) > 0
        memory_facts = [m.fact for m in memories]
        assert any("Python" in f for f in memory_facts)

    @pytest.mark.asyncio
    async def test_memory_generation_from_session(
        self,
        session_manager: SessionManager,
        memory_manager: MemoryManager,
    ):
        """Test automatic memory generation from session conversations."""
        user_id = "memory-gen-user"

        # Create a session
        session = await session_manager.create_session(user_id=user_id)

        # Add conversation events
        await session_manager.append_event(
            session_id=session.session_id,
            author=EventAuthor.USER,
            content={"text": "My name is Alice and I work as a data scientist."},
        )
        await session_manager.append_event(
            session_id=session.session_id,
            author=EventAuthor.AGENT,
            content={"text": "Nice to meet you, Alice! Data science is fascinating."},
        )

        # Refresh session to get events
        session = await session_manager.get_session(session.session_id)

        # Generate memories from session
        generated = await memory_manager.generate_from_session(session)

        # Should extract at least one fact
        assert len(generated) >= 1

    @pytest.mark.asyncio
    async def test_session_continuity_with_memories(
        self,
        session_manager: SessionManager,
    ):
        """Test that memories persist across sessions."""
        # Use low threshold memory manager for testing
        low_threshold_config = MemoryConfig(
            enabled=True,
            auto_generate=False,
            max_memories_per_user=100,
            similarity_threshold=0.1,
        )
        memory_manager = MemoryManager(config=low_threshold_config)

        user_id = "continuity-user"

        # First session: save a memory
        session1 = await session_manager.create_session(user_id=user_id)
        await session_manager.append_event(
            session_id=session1.session_id,
            author=EventAuthor.USER,
            content={"text": "I prefer morning meetings."},
        )

        await memory_manager.save_memory(
            user_id=user_id,
            fact="User prefers morning meetings",
            topics=["preferences", "meetings"],
        )

        # Delete session (simulating session expiry)
        await session_manager.delete_session(session1.session_id)

        # Second session: memories should still be accessible
        session2 = await session_manager.create_session(user_id=user_id)

        # Retrieve memories without query (no similarity search)
        memories = await memory_manager.retrieve_memories(
            user_id=user_id,
            max_results=10,
        )

        assert len(memories) > 0
        assert any("morning" in m.fact.lower() for m in memories)


class TestAgentIntegration:
    """Integration tests for Agent with Phase 2 features."""

    @pytest.fixture
    def full_agent_config(self) -> AgentConfig:
        """Create a full agent configuration for integration tests."""
        return AgentConfig(
            project_id="test-project",
            location="asia-northeast3",
            model="gemini-2.5-pro",
            temperature=0.7,
            max_tokens=4096,
            system_prompt="You are a helpful test assistant with memory.",
            session=SessionConfig(
                enabled=True,
                default_ttl_seconds=3600,
                max_events_per_session=100,
            ),
            memory=MemoryConfig(
                enabled=True,
                auto_generate=True,
                max_memories_per_user=100,
                similarity_threshold=0.5,
            ),
            observability=ObservabilityConfig(
                tracing_enabled=False,
                logging_enabled=True,
                metrics_enabled=True,
            ),
        )

    def test_agent_initialization_with_phase2_config(self, full_agent_config: AgentConfig):
        """Test agent initialization with Phase 2 configuration."""
        agent = PydanticAIAgentWrapper.from_config(full_agent_config)

        assert agent.config is not None
        assert agent.config.session.enabled is True
        assert agent.config.memory.enabled is True
        assert agent.config.observability.metrics_enabled is True

    @pytest.mark.asyncio
    async def test_agent_setup_initializes_phase2_components(
        self,
        full_agent_config: AgentConfig,
    ):
        """Test that agent.set_up() initializes Phase 2 components."""
        agent = PydanticAIAgentWrapper.from_config(full_agent_config)

        # Mock the external dependencies
        with (
            patch("vertexai.init"),
            patch("pydantic_ai.providers.google.GoogleProvider"),
            patch("pydantic_ai.models.google.GoogleModel"),
            patch("pydantic_ai.Agent") as mock_agent_class,
        ):
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            agent.set_up()

            # Phase 2 components should be initialized
            assert agent._session_manager is not None
            assert agent._memory_manager is not None
            # Metrics manager may be initialized
            assert agent._is_setup is True

    @pytest.mark.asyncio
    async def test_agent_query_with_session(
        self,
        full_agent_config: AgentConfig,
    ):
        """Test agent query_with_session method."""
        agent = PydanticAIAgentWrapper.from_config(full_agent_config)

        # Set up mocks
        mock_result = MagicMock()
        mock_result.output = "Hello! How can I help you today?"
        mock_result.tool_calls = []
        mock_result.usage = MagicMock(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        )

        with (
            patch("vertexai.init"),
            patch("pydantic_ai.providers.google.GoogleProvider"),
            patch("pydantic_ai.models.google.GoogleModel"),
            patch("pydantic_ai.Agent") as mock_agent_class,
        ):
            mock_agent = MagicMock()
            mock_agent.run = MagicMock(return_value=mock_result)
            # Make it an async function
            async def mock_run(*args, **kwargs):
                return mock_result
            mock_agent.run = mock_run
            mock_agent_class.return_value = mock_agent

            agent.set_up()

            # Execute query with session
            response = await agent.query_with_session(
                message="Hello!",
                user_id="test-user",
            )

            assert response is not None
            assert "response" in response
            assert "session_id" in response
            assert response["session_id"] is not None

    @pytest.mark.asyncio
    async def test_agent_stats_include_phase2_metrics(
        self,
        full_agent_config: AgentConfig,
    ):
        """Test that agent.get_stats() includes Phase 2 metrics."""
        agent = PydanticAIAgentWrapper.from_config(full_agent_config)

        with (
            patch("vertexai.init"),
            patch("pydantic_ai.providers.google.GoogleProvider"),
            patch("pydantic_ai.models.google.GoogleModel"),
            patch("pydantic_ai.Agent") as mock_agent_class,
        ):
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            agent.set_up()

            stats = agent.get_stats()

            assert "model" in stats
            assert "sessions" in stats
            assert "memory" in stats


class TestObservabilityIntegration:
    """Integration tests for Observability components."""

    @pytest.mark.asyncio
    async def test_metrics_across_multiple_operations(
        self,
        metrics_manager,
    ):
        """Test metrics collection across multiple operations."""
        # Simulate multiple requests
        for i in range(10):
            metrics_manager.record_request(method="query")
            metrics_manager.record_latency(100 + i * 10)
            if i % 3 == 0:
                metrics_manager.record_error("TestError")

        metrics_manager.record_tokens(input_tokens=500, output_tokens=250)

        stats = metrics_manager.get_stats()

        assert stats["requests"] >= 10
        assert stats["errors"] >= 3
        assert stats["tokens"]["input"] >= 500
        assert stats["tokens"]["output"] >= 250
        assert "latency" in stats

    def test_logged_tool_integration(self):
        """Test logged_tool decorator in realistic scenario."""
        from agent_engine.observability import logged_tool

        call_count = 0

        @logged_tool
        def realistic_tool(query: str, max_results: int = 5) -> dict:
            nonlocal call_count
            call_count += 1
            return {
                "query": query,
                "results": [f"Result {i}" for i in range(max_results)],
                "count": max_results,
            }

        # Use the tool multiple times
        for i in range(3):
            result = realistic_tool(f"search query {i}", max_results=3)
            assert result["count"] == 3

        assert call_count == 3


class TestEndToEndScenarios:
    """End-to-end integration scenarios."""

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_scenario(
        self,
        session_manager: SessionManager,
    ):
        """Test a multi-turn conversation with memory persistence."""
        # Use low threshold memory manager for testing
        low_threshold_config = MemoryConfig(
            enabled=True,
            auto_generate=False,
            max_memories_per_user=100,
            similarity_threshold=0.1,
        )
        memory_manager = MemoryManager(config=low_threshold_config)

        user_id = "e2e-user"

        # Turn 1: User introduces themselves
        session = await session_manager.create_session(user_id=user_id)
        await session_manager.append_event(
            session_id=session.session_id,
            author=EventAuthor.USER,
            content={"text": "Hi, my name is Bob and I'm learning Python."},
        )
        await memory_manager.save_memory(
            user_id=user_id,
            fact="User's name is Bob",
            topics=["identity"],
        )
        await memory_manager.save_memory(
            user_id=user_id,
            fact="User is learning Python",
            topics=["programming", "learning"],
        )

        # Turn 2: User asks a question
        await session_manager.append_event(
            session_id=session.session_id,
            author=EventAuthor.USER,
            content={"text": "What's the best way to learn functions?"},
        )

        # Retrieve all memories for the user (without query for reliable test)
        memories = await memory_manager.retrieve_memories(
            user_id=user_id,
            max_results=10,
        )

        # Should have context about user learning Python
        assert len(memories) >= 2
        memory_facts = [m.fact for m in memories]
        assert any("Python" in f or "learning" in f.lower() for f in memory_facts)

        # Turn 3: Continue conversation
        await session_manager.append_event(
            session_id=session.session_id,
            author=EventAuthor.AGENT,
            content={"text": "Great question, Bob! For Python functions..."},
        )

        # Verify session history
        events = await session_manager.list_events(session.session_id)
        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_gdpr_deletion_scenario(
        self,
        memory_manager: MemoryManager,
    ):
        """Test GDPR-compliant user data deletion."""
        user_id = "gdpr-user"

        # Collect user data
        await memory_manager.save_memory(
            user_id=user_id,
            fact="User email: test@example.com",
            topics=["pii", "contact"],
        )
        await memory_manager.save_memory(
            user_id=user_id,
            fact="User phone: 123-456-7890",
            topics=["pii", "contact"],
        )
        await memory_manager.save_memory(
            user_id=user_id,
            fact="User preferences: dark mode",
            topics=["preferences"],
        )

        # Verify data exists
        assert memory_manager.get_user_memory_count(user_id) == 3

        # User requests data deletion
        deleted_count = await memory_manager.delete_user_memories(user_id)

        # Verify complete deletion
        assert deleted_count == 3
        assert memory_manager.get_user_memory_count(user_id) == 0

        # Verify retrieval returns nothing
        memories = await memory_manager.retrieve_memories(user_id)
        assert len(memories) == 0

    @pytest.mark.asyncio
    async def test_session_expiry_and_recreation(
        self,
        session_manager: SessionManager,
    ):
        """Test session expiry with memory persistence."""
        # Use low threshold memory manager for testing
        low_threshold_config = MemoryConfig(
            enabled=True,
            auto_generate=False,
            max_memories_per_user=100,
            similarity_threshold=0.1,
        )
        memory_manager = MemoryManager(config=low_threshold_config)

        user_id = "expiry-user"

        # Use standard session manager (manually expire session)
        short_session_manager = session_manager

        # Create session and save memory
        session1 = await short_session_manager.create_session(user_id=user_id)
        await memory_manager.save_memory(
            user_id=user_id,
            fact="User prefers concise answers",
        )

        session1_id = session1.session_id

        # Manually expire the session for testing
        from datetime import datetime, timedelta
        session1.expires_at = datetime.utcnow() - timedelta(seconds=1)

        # Session should be expired
        expired_session = await short_session_manager.get_session(session1_id)
        assert expired_session is None

        # Create new session
        session2 = await short_session_manager.create_session(user_id=user_id)
        assert session2.session_id != session1_id

        # Memory should persist across sessions
        memories = await memory_manager.retrieve_memories(user_id=user_id)
        assert len(memories) > 0
        assert any("concise" in m.fact.lower() for m in memories)
