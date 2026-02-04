"""Memory manager for the Agent Engine platform.

Implements MB-001~MB-006:
- MB-001: Auto memory generation (fact extraction)
- MB-002: User-specific memory retrieval
- MB-003: Similarity search
- MB-004: Agent explicit memory storage
- MB-006: Memory deletion (GDPR compliance)

Uses a pluggable backend system for storage:
- InMemoryBackend: Local development/testing
- VertexAIBackend: Production with persistent storage
"""

from typing import Any

import structlog

from agent_engine.config import MemoryBackendType, MemoryConfig
from agent_engine.memory.backends.base import MemoryBackend
from agent_engine.memory.backends.in_memory import InMemoryMemoryBackend
from agent_engine.memory.backends.vertex_ai import VertexAIMemoryBackend
from agent_engine.memory.models import Memory, MemoryScope
from agent_engine.memory.retriever import MemoryRetriever
from agent_engine.sessions.models import EventAuthor, Session

logger = structlog.get_logger(__name__)


class MemoryManager:
    """Manages user memories with pluggable backends.

    This manager handles memory storage, retrieval, and automatic
    fact extraction from conversations. It delegates storage to
    a backend implementation.

    Attributes:
        config: Memory configuration
        retriever: Memory retriever for similarity search
        _backend: Storage backend implementation
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        project_id: str | None = None,
        location: str | None = None,
    ) -> None:
        """Initialize the memory manager.

        Args:
            config: Memory configuration, uses defaults if not provided
            project_id: GCP project ID (required for VertexAI backend)
            location: GCP region (required for VertexAI backend)
        """
        self.config = config or MemoryConfig()
        self._project_id = project_id
        self._location = location
        self.retriever = MemoryRetriever(similarity_threshold=self.config.similarity_threshold)
        self._backend = self._create_backend()

        logger.info(
            "MemoryManager initialized",
            backend=self.config.backend.value,
            auto_generate=self.config.auto_generate,
            max_per_user=self.config.max_memories_per_user,
            similarity_threshold=self.config.similarity_threshold,
        )

    def _create_backend(self) -> MemoryBackend:
        """Create the appropriate backend based on configuration."""
        if self.config.backend == MemoryBackendType.VERTEX_AI:
            if not self.config.agent_engine_id:
                raise ValueError(
                    "agent_engine_id is required for VertexAI backend. "
                    "Set MEMORY_AGENT_ENGINE_ID environment variable."
                )
            if not self._project_id or not self._location:
                raise ValueError("project_id and location are required for VertexAI backend.")

            return VertexAIMemoryBackend(
                project_id=self._project_id,
                location=self._location,
                agent_engine_id=self.config.agent_engine_id,
                config=self.config,
            )

        # Default to in-memory backend
        return InMemoryMemoryBackend(config=self.config)

    async def save_memory(
        self,
        user_id: str,
        fact: str,
        topics: list[str] | None = None,
        scope: MemoryScope = MemoryScope.USER,
        source: str = "explicit",
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Save a memory for a user.

        MB-004: Agent explicit memory storage.

        Args:
            user_id: User identifier
            fact: The memory/fact to store
            topics: Related topics/tags
            scope: Memory visibility scope
            source: Where this memory came from
            metadata: Optional metadata

        Returns:
            Created Memory
        """
        # Generate embedding for similarity search (if using in-memory backend)
        embedding = None
        if self.config.backend == MemoryBackendType.IN_MEMORY:
            embedding = await self.retriever.generate_embedding(fact)

        return await self._backend.save_memory(
            user_id=user_id,
            fact=fact,
            embedding=embedding,
            topics=topics,
            scope=scope,
            source=source,
            metadata=metadata,
        )

    async def retrieve_memories(
        self,
        user_id: str,
        query: str | None = None,
        max_results: int = 10,
        include_global: bool = True,
    ) -> list[Memory]:
        """Retrieve memories for a user.

        MB-002: User-specific memory retrieval.
        MB-003: Similarity search (when query provided).

        Args:
            user_id: User identifier
            query: Optional search query for similarity matching
            max_results: Maximum number of memories to return
            include_global: Whether to include global scope memories

        Returns:
            List of relevant memories
        """
        # For VertexAI backend, let it handle similarity search
        if self.config.backend == MemoryBackendType.VERTEX_AI:
            memories = await self._backend.retrieve_memories(
                user_id=user_id,
                query=query,
                max_results=max_results,
                include_global=include_global,
            )
        else:
            # For in-memory backend, use retriever for similarity search
            all_memories = await self._backend.retrieve_memories(
                user_id=user_id,
                max_results=max_results * 2,  # Get more for filtering
                include_global=include_global,
            )

            if not all_memories:
                return []

            if query:
                # Use similarity search
                results = await self.retriever.search(
                    query=query,
                    memories=all_memories,
                    max_results=max_results,
                )
                memories = [r.memory for r in results]
            else:
                # Return most recently accessed
                memories = sorted(
                    all_memories,
                    key=lambda m: m.accessed_at,
                    reverse=True,
                )[:max_results]

        # Mark memories as accessed
        for memory in memories:
            memory.mark_accessed()

        logger.debug(
            "Memories retrieved",
            user_id=user_id,
            query=query,
            count=len(memories),
        )

        return memories

    async def generate_from_session(
        self,
        session: Session,
        user_id: str | None = None,
    ) -> list[Memory]:
        """Generate memories from a session conversation.

        MB-001: Auto memory generation (fact extraction).

        Args:
            session: Session to extract memories from
            user_id: Override user ID (defaults to session's user_id)

        Returns:
            List of generated memories
        """
        if not self.config.auto_generate:
            logger.debug("Auto-generate disabled, skipping")
            return []

        user_id = user_id or session.user_id
        generated_memories: list[Memory] = []

        # Extract facts from user messages
        for event in session.events:
            if event.author == EventAuthor.USER:
                facts = await self._extract_facts_from_content(event.content)
                for fact in facts:
                    # Check for duplicate facts
                    if not await self._is_duplicate_fact(user_id, fact):
                        memory = await self.save_memory(
                            user_id=user_id,
                            fact=fact,
                            source="conversation",
                            metadata={"session_id": session.session_id},
                        )
                        generated_memories.append(memory)

        if generated_memories:
            logger.info(
                "Memories generated from session",
                session_id=session.session_id,
                user_id=user_id,
                count=len(generated_memories),
            )

        return generated_memories

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory.

        Args:
            memory_id: Memory identifier

        Returns:
            True if deleted, False if not found
        """
        return await self._backend.delete_memory(memory_id)

    async def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user.

        MB-006: Memory deletion (GDPR compliance).

        Args:
            user_id: User identifier

        Returns:
            Number of memories deleted
        """
        return await self._backend.delete_user_memories(user_id)

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory if found, None otherwise
        """
        return await self._backend.get_memory(memory_id)

    async def update_memory(
        self,
        memory_id: str,
        fact: str | None = None,
        topics: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory | None:
        """Update an existing memory.

        Args:
            memory_id: Memory identifier
            fact: New fact content
            topics: New topics
            metadata: New metadata (merged with existing)

        Returns:
            Updated Memory or None if not found
        """
        # Generate new embedding if fact is updated
        embedding = None
        if fact is not None and self.config.backend == MemoryBackendType.IN_MEMORY:
            embedding = await self.retriever.generate_embedding(fact)

        return await self._backend.update_memory(
            memory_id=memory_id,
            fact=fact,
            embedding=embedding,
            topics=topics,
            metadata=metadata,
        )

    async def _extract_facts_from_content(
        self,
        content: dict[str, Any],
    ) -> list[str]:
        """Extract facts from event content.

        This is a simplified implementation. In production, this could
        use an LLM to extract meaningful facts.

        Args:
            content: Event content dictionary

        Returns:
            List of extracted facts
        """
        facts: list[str] = []

        # Look for explicit user info
        if "text" in content:
            text = content["text"]
            # Simple heuristics for fact extraction
            # In production, use LLM for better extraction
            fact_indicators = [
                "my name is",
                "i am",
                "i'm",
                "i work",
                "i live",
                "i like",
                "i prefer",
                "i need",
                "i want",
            ]
            text_lower = text.lower()
            for indicator in fact_indicators:
                if indicator in text_lower:
                    # Extract the sentence containing the indicator
                    sentences = text.split(".")
                    for sentence in sentences:
                        if indicator in sentence.lower():
                            fact = sentence.strip()
                            if fact and len(fact) > 10:
                                facts.append(fact)
                            break

        return facts

    async def _is_duplicate_fact(self, user_id: str, fact: str) -> bool:
        """Check if a similar fact already exists.

        Args:
            user_id: User identifier
            fact: Fact to check

        Returns:
            True if duplicate exists
        """
        existing_memories = await self.retrieve_memories(
            user_id=user_id,
            query=fact,
            max_results=1,
        )

        if existing_memories:
            # Check similarity
            results = await self.retriever.search(
                query=fact,
                memories=existing_memories,
                max_results=1,
            )
            if results and results[0].score > 0.9:
                return True

        return False

    def get_user_memory_count(self, user_id: str) -> int:
        """Get the number of memories for a user."""
        return self._backend.get_user_memory_count(user_id)

    def get_stats(self) -> dict[str, Any]:
        """Get memory manager statistics."""
        stats = self._backend.get_stats()
        stats["config"] = {
            "auto_generate": self.config.auto_generate,
            "max_per_user": self.config.max_memories_per_user,
            "similarity_threshold": self.config.similarity_threshold,
        }
        return stats

    @property
    def backend_type(self) -> MemoryBackendType:
        """Get the current backend type."""
        return self.config.backend
