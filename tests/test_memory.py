"""Tests for Memory Bank module (MB-001~MB-006)."""

import pytest

from agent_engine.config import MemoryConfig
from agent_engine.memory import Memory, MemoryManager, MemoryRetriever, MemoryScope
from agent_engine.sessions import EventAuthor, Session, SessionEvent


class TestMemoryModels:
    """Tests for memory models."""

    def test_memory_creation(self):
        """Test Memory creation with defaults."""
        memory = Memory(
            user_id="user-123",
            fact="User likes Python programming",
        )

        assert memory.memory_id is not None
        assert memory.user_id == "user-123"
        assert memory.fact == "User likes Python programming"
        assert memory.scope == MemoryScope.USER
        assert memory.source == "conversation"
        assert memory.access_count == 0

    def test_memory_mark_accessed(self):
        """Test Memory access tracking."""
        memory = Memory(
            user_id="user-123",
            fact="Test fact",
        )

        initial_time = memory.accessed_at
        memory.mark_accessed()

        assert memory.access_count == 1
        assert memory.accessed_at >= initial_time

    def test_memory_to_dict(self):
        """Test Memory serialization."""
        memory = Memory(
            user_id="user-123",
            fact="User works at Acme Corp",
            topics=["work", "company"],
            scope=MemoryScope.USER,
        )

        data = memory.to_dict()

        assert data["user_id"] == "user-123"
        assert data["fact"] == "User works at Acme Corp"
        assert data["topics"] == ["work", "company"]
        assert data["scope"] == "user"

    def test_memory_to_context_string(self):
        """Test Memory context string generation."""
        memory = Memory(
            user_id="user-123",
            fact="User prefers dark mode",
            topics=["preferences", "ui"],
        )

        context = memory.to_context_string()

        assert "User prefers dark mode" in context
        assert "preferences" in context


class TestMemoryRetriever:
    """Tests for MemoryRetriever (MB-003)."""

    @pytest.mark.asyncio
    async def test_generate_embedding(self, memory_retriever: MemoryRetriever):
        """Test embedding generation."""
        embedding = await memory_retriever.generate_embedding("Hello world")

        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_compute_similarity(self, memory_retriever: MemoryRetriever):
        """Test similarity computation."""
        emb1 = await memory_retriever.generate_embedding("Hello world")
        emb2 = await memory_retriever.generate_embedding("Hello world")
        emb3 = await memory_retriever.generate_embedding("Goodbye universe")

        # Same text should have high similarity
        sim_same = await memory_retriever.compute_similarity(emb1, emb2)
        assert sim_same > 0.9

        # Different text should have lower similarity
        sim_diff = await memory_retriever.compute_similarity(emb1, emb3)
        assert sim_diff < sim_same

    @pytest.mark.asyncio
    async def test_search(self, memory_retriever: MemoryRetriever):
        """MB-003: Test similarity search."""
        # Use low threshold retriever for testing
        retriever = MemoryRetriever(similarity_threshold=0.1)

        memories = [
            Memory(user_id="user-123", fact="User likes Python programming"),
            Memory(user_id="user-123", fact="User works at a tech company"),
            Memory(user_id="user-123", fact="User enjoys hiking on weekends"),
        ]

        # Generate embeddings for memories
        for memory in memories:
            memory.embedding = await retriever.generate_embedding(memory.fact)

        # Search for programming-related memories using similar words
        results = await retriever.search(
            query="Python programming language",
            memories=memories,
            max_results=3,
        )

        # With low threshold, should return results
        assert len(results) >= 1
        # Check that we got at least some results
        memory_facts = [r.memory.fact for r in results]
        assert len(memory_facts) > 0

    @pytest.mark.asyncio
    async def test_search_with_threshold(self):
        """Test search with high threshold."""
        retriever = MemoryRetriever(similarity_threshold=0.99)

        memories = [
            Memory(user_id="user-123", fact="Random fact about cats"),
        ]
        memories[0].embedding = await retriever.generate_embedding(memories[0].fact)

        # Query about completely different topic
        results = await retriever.search(
            query="What is the weather today?",
            memories=memories,
            max_results=5,
        )

        # High threshold should filter out low-similarity results
        assert len(results) == 0


class TestMemoryManager:
    """Tests for MemoryManager (MB-001~MB-006)."""

    @pytest.mark.asyncio
    async def test_save_memory(self, memory_manager: MemoryManager):
        """MB-004: Test explicit memory storage."""
        memory = await memory_manager.save_memory(
            user_id="user-123",
            fact="User prefers dark mode",
            topics=["preferences", "ui"],
        )

        assert memory is not None
        assert memory.user_id == "user-123"
        assert memory.fact == "User prefers dark mode"
        assert memory.topics == ["preferences", "ui"]
        assert memory.source == "explicit"

    @pytest.mark.asyncio
    async def test_retrieve_memories_by_user(self, memory_manager: MemoryManager):
        """MB-002: Test user-specific memory retrieval."""
        # Save memories for different users
        await memory_manager.save_memory(
            user_id="user-1",
            fact="User 1 likes Python",
        )
        await memory_manager.save_memory(
            user_id="user-2",
            fact="User 2 likes Java",
        )

        # Retrieve only user-1's memories
        memories = await memory_manager.retrieve_memories(user_id="user-1")

        assert len(memories) == 1
        assert memories[0].user_id == "user-1"

    @pytest.mark.asyncio
    async def test_retrieve_memories_with_query(self, memory_config: MemoryConfig):
        """MB-003: Test retrieval with similarity search."""
        # Create manager with low similarity threshold for testing
        from agent_engine.memory import MemoryManager
        low_threshold_config = MemoryConfig(
            enabled=True,
            auto_generate=False,
            max_memories_per_user=100,
            similarity_threshold=0.1,  # Low threshold for testing
        )
        memory_manager = MemoryManager(config=low_threshold_config)

        # Save multiple memories
        await memory_manager.save_memory(
            user_id="user-123",
            fact="User works as a software engineer",
        )
        await memory_manager.save_memory(
            user_id="user-123",
            fact="User enjoys hiking on weekends",
        )
        await memory_manager.save_memory(
            user_id="user-123",
            fact="User is learning Python programming",
        )

        # Query for work-related memories using similar words
        memories = await memory_manager.retrieve_memories(
            user_id="user-123",
            query="software engineer work",
            max_results=3,
        )

        # With low threshold, should return at least one result
        assert len(memories) >= 1

    @pytest.mark.asyncio
    async def test_delete_memory(self, memory_manager: MemoryManager):
        """Test single memory deletion."""
        memory = await memory_manager.save_memory(
            user_id="user-123",
            fact="Test memory",
        )

        # Delete the memory
        result = await memory_manager.delete_memory(memory.memory_id)
        assert result is True

        # Verify it's gone
        retrieved = await memory_manager.get_memory(memory.memory_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_user_memories(self, memory_manager: MemoryManager):
        """MB-006: Test GDPR-compliant user memory deletion."""
        # Save multiple memories for a user
        for i in range(5):
            await memory_manager.save_memory(
                user_id="user-to-delete",
                fact=f"Memory {i}",
            )

        # Also save memories for another user
        await memory_manager.save_memory(
            user_id="user-to-keep",
            fact="Keep this memory",
        )

        # Delete all memories for user-to-delete
        deleted_count = await memory_manager.delete_user_memories("user-to-delete")

        assert deleted_count == 5

        # Verify user-to-delete has no memories
        memories = await memory_manager.retrieve_memories("user-to-delete")
        assert len(memories) == 0

        # Verify user-to-keep still has memories
        kept_memories = await memory_manager.retrieve_memories("user-to-keep")
        assert len(kept_memories) == 1

    @pytest.mark.asyncio
    async def test_generate_from_session(self, memory_manager: MemoryManager):
        """MB-001: Test auto memory generation from session."""
        from datetime import datetime, timedelta

        # Create a mock session with user messages
        session = Session(
            user_id="user-123",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )

        # Add events with personal information
        session.events = [
            SessionEvent(
                author=EventAuthor.USER,
                content={"text": "My name is John and I work at a tech company."},
            ),
            SessionEvent(
                author=EventAuthor.AGENT,
                content={"text": "Nice to meet you, John!"},
            ),
            SessionEvent(
                author=EventAuthor.USER,
                content={"text": "I like Python programming."},
            ),
        ]

        # Generate memories from session
        generated = await memory_manager.generate_from_session(session)

        # Should extract facts from user messages
        assert len(generated) >= 1

    @pytest.mark.asyncio
    async def test_max_memories_per_user_eviction(self, memory_config: MemoryConfig):
        """Test memory eviction when limit is reached."""
        limited_config = MemoryConfig(
            enabled=True,
            auto_generate=False,
            max_memories_per_user=10,  # Minimum allowed is 10
            similarity_threshold=0.5,
        )
        manager = MemoryManager(config=limited_config)

        # Save more memories than the limit
        for i in range(15):
            await manager.save_memory(
                user_id="user-123",
                fact=f"Memory number {i}",
            )

        # Should only have max_memories_per_user memories
        assert manager.get_user_memory_count("user-123") == 10

    @pytest.mark.asyncio
    async def test_update_memory(self, memory_manager: MemoryManager):
        """Test memory update."""
        memory = await memory_manager.save_memory(
            user_id="user-123",
            fact="Original fact",
            topics=["original"],
        )

        # Update the memory
        updated = await memory_manager.update_memory(
            memory_id=memory.memory_id,
            fact="Updated fact",
            topics=["updated", "new"],
        )

        assert updated is not None
        assert updated.fact == "Updated fact"
        assert updated.topics == ["updated", "new"]

    @pytest.mark.asyncio
    async def test_global_scope_memories(self, memory_manager: MemoryManager):
        """Test global scope memory retrieval."""
        # Save a global memory
        await memory_manager.save_memory(
            user_id="admin",
            fact="Company policy: all meetings must have agendas",
            scope=MemoryScope.GLOBAL,
        )

        # Save a user-specific memory
        await memory_manager.save_memory(
            user_id="user-123",
            fact="User prefers morning meetings",
        )

        # User should see both their memories and global memories
        memories = await memory_manager.retrieve_memories(
            user_id="user-123",
            include_global=True,
        )

        # Should include both memories
        assert len(memories) == 2

    @pytest.mark.asyncio
    async def test_get_stats(self, memory_manager: MemoryManager):
        """Test memory manager statistics."""
        await memory_manager.save_memory(
            user_id="user-1",
            fact="Fact 1",
        )
        await memory_manager.save_memory(
            user_id="user-2",
            fact="Fact 2",
            scope=MemoryScope.GLOBAL,
        )

        stats = memory_manager.get_stats()

        assert stats["total_memories"] == 2
        assert stats["total_users"] == 2
        assert stats["scope_distribution"]["user"] == 1
        assert stats["scope_distribution"]["global"] == 1
