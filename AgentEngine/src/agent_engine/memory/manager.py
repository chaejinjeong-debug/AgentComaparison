"""Memory manager for the Agent Engine platform.

Implements MB-001~MB-006:
- MB-001: Auto memory generation (fact extraction)
- MB-002: User-specific memory retrieval
- MB-003: Similarity search
- MB-004: Agent explicit memory storage
- MB-006: Memory deletion (GDPR compliance)
"""

from datetime import datetime
from typing import Any

import structlog

from agent_engine.config import MemoryConfig
from agent_engine.memory.models import Memory, MemoryScope
from agent_engine.memory.retriever import MemoryRetriever
from agent_engine.sessions.models import EventAuthor, Session

logger = structlog.get_logger(__name__)


class MemoryManager:
    """Manages user memories with similarity search support.

    This manager handles memory storage, retrieval, and automatic
    fact extraction from conversations.

    Attributes:
        config: Memory configuration
        retriever: Memory retriever for similarity search
        _memories: In-memory storage (for local/testing use)
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        """Initialize the memory manager.

        Args:
            config: Memory configuration, uses defaults if not provided
        """
        self.config = config or MemoryConfig()
        self.retriever = MemoryRetriever(
            similarity_threshold=self.config.similarity_threshold
        )
        self._memories: dict[str, Memory] = {}  # memory_id -> Memory
        self._user_index: dict[str, set[str]] = {}  # user_id -> set of memory_ids

        logger.info(
            "MemoryManager initialized",
            auto_generate=self.config.auto_generate,
            max_per_user=self.config.max_memories_per_user,
            similarity_threshold=self.config.similarity_threshold,
        )

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
        # Check user memory limit
        user_memories = self._user_index.get(user_id, set())
        if len(user_memories) >= self.config.max_memories_per_user:
            # Remove oldest accessed memory
            await self._evict_oldest_memory(user_id)

        # Generate embedding for similarity search
        embedding = await self.retriever.generate_embedding(fact)

        memory = Memory(
            user_id=user_id,
            fact=fact,
            topics=topics or [],
            scope=scope,
            source=source,
            embedding=embedding,
            metadata=metadata or {},
        )

        self._memories[memory.memory_id] = memory

        if user_id not in self._user_index:
            self._user_index[user_id] = set()
        self._user_index[user_id].add(memory.memory_id)

        logger.info(
            "Memory saved",
            memory_id=memory.memory_id,
            user_id=user_id,
            topics=topics,
            source=source,
        )

        return memory

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
        user_memory_ids = self._user_index.get(user_id, set())
        user_memories = [
            self._memories[mid]
            for mid in user_memory_ids
            if mid in self._memories
        ]

        if include_global:
            global_memories = [
                m for m in self._memories.values()
                if m.scope == MemoryScope.GLOBAL and m.memory_id not in user_memory_ids
            ]
            user_memories.extend(global_memories)

        if not user_memories:
            return []

        if query:
            # Use similarity search
            results = await self.retriever.search(
                query=query,
                memories=user_memories,
                max_results=max_results,
            )
            memories = [r.memory for r in results]
        else:
            # Return most recently accessed
            memories = sorted(
                user_memories,
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
        if memory_id not in self._memories:
            return False

        memory = self._memories[memory_id]
        user_id = memory.user_id

        del self._memories[memory_id]

        if user_id in self._user_index:
            self._user_index[user_id].discard(memory_id)

        logger.info("Memory deleted", memory_id=memory_id, user_id=user_id)
        return True

    async def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user.

        MB-006: Memory deletion (GDPR compliance).

        Args:
            user_id: User identifier

        Returns:
            Number of memories deleted
        """
        memory_ids = self._user_index.get(user_id, set()).copy()

        for memory_id in memory_ids:
            if memory_id in self._memories:
                del self._memories[memory_id]

        deleted_count = len(memory_ids)
        self._user_index.pop(user_id, None)

        logger.info(
            "User memories deleted (GDPR)",
            user_id=user_id,
            count=deleted_count,
        )

        return deleted_count

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory if found, None otherwise
        """
        return self._memories.get(memory_id)

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
        memory = self._memories.get(memory_id)
        if memory is None:
            return None

        if fact is not None:
            memory.fact = fact
            # Regenerate embedding
            memory.embedding = await self.retriever.generate_embedding(fact)

        if topics is not None:
            memory.topics = topics

        if metadata is not None:
            memory.metadata.update(metadata)

        logger.debug("Memory updated", memory_id=memory_id)
        return memory

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

    async def _evict_oldest_memory(self, user_id: str) -> None:
        """Evict the oldest accessed memory for a user.

        Args:
            user_id: User identifier
        """
        memory_ids = self._user_index.get(user_id, set())
        if not memory_ids:
            return

        # Find oldest accessed memory
        oldest_memory = min(
            (self._memories[mid] for mid in memory_ids if mid in self._memories),
            key=lambda m: m.accessed_at,
            default=None,
        )

        if oldest_memory:
            await self.delete_memory(oldest_memory.memory_id)
            logger.debug(
                "Evicted oldest memory",
                user_id=user_id,
                memory_id=oldest_memory.memory_id,
            )

    def get_user_memory_count(self, user_id: str) -> int:
        """Get the number of memories for a user."""
        return len(self._user_index.get(user_id, set()))

    def get_stats(self) -> dict[str, Any]:
        """Get memory manager statistics."""
        total_memories = len(self._memories)
        total_users = len(self._user_index)

        scope_counts = {
            "user": 0,
            "session": 0,
            "global": 0,
        }
        for memory in self._memories.values():
            scope_counts[memory.scope.value] += 1

        return {
            "total_memories": total_memories,
            "total_users": total_users,
            "scope_distribution": scope_counts,
            "config": {
                "auto_generate": self.config.auto_generate,
                "max_per_user": self.config.max_memories_per_user,
                "similarity_threshold": self.config.similarity_threshold,
            },
        }
