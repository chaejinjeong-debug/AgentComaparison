"""In-memory memory storage backend for local development and testing."""

from typing import Any

import structlog

from agent_engine.config import MemoryConfig
from agent_engine.memory.backends.base import MemoryBackend
from agent_engine.memory.models import Memory, MemoryScope

logger = structlog.get_logger(__name__)


class InMemoryMemoryBackend(MemoryBackend):
    """In-memory memory storage backend.

    This backend stores memories in memory and is suitable for:
    - Local development
    - Testing
    - Single-instance deployments

    Note: Memories are lost when the process restarts.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        """Initialize the in-memory backend.

        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        self._memories: dict[str, Memory] = {}  # memory_id -> Memory
        self._user_index: dict[str, set[str]] = {}  # user_id -> set of memory_ids

        logger.info(
            "InMemoryMemoryBackend initialized",
            max_per_user=self.config.max_memories_per_user,
        )

    async def save_memory(
        self,
        user_id: str,
        fact: str,
        embedding: list[float] | None = None,
        topics: list[str] | None = None,
        scope: MemoryScope = MemoryScope.USER,
        source: str = "explicit",
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Save a memory for a user."""
        # Check user memory limit
        user_memories = self._user_index.get(user_id, set())
        if len(user_memories) >= self.config.max_memories_per_user:
            # Remove oldest accessed memory
            await self._evict_oldest_memory(user_id)

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
        query_embedding: list[float] | None = None,
        max_results: int = 10,
        include_global: bool = True,
    ) -> list[Memory]:
        """Retrieve memories for a user.

        Note: This backend returns all memories and lets the MemoryManager
        handle similarity search using the retriever.
        """
        user_memory_ids = self._user_index.get(user_id, set())
        user_memories = [self._memories[mid] for mid in user_memory_ids if mid in self._memories]

        if include_global:
            global_memories = [
                m
                for m in self._memories.values()
                if m.scope == MemoryScope.GLOBAL and m.memory_id not in user_memory_ids
            ]
            user_memories.extend(global_memories)

        if not user_memories:
            return []

        # If no query, return most recently accessed
        if not query and not query_embedding:
            memories = sorted(
                user_memories,
                key=lambda m: m.accessed_at,
                reverse=True,
            )[:max_results]
        else:
            # Return all for external similarity search
            memories = user_memories[:max_results]

        logger.debug(
            "Memories retrieved",
            user_id=user_id,
            count=len(memories),
        )

        return memories

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID."""
        return self._memories.get(memory_id)

    async def update_memory(
        self,
        memory_id: str,
        fact: str | None = None,
        embedding: list[float] | None = None,
        topics: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory | None:
        """Update an existing memory."""
        memory = self._memories.get(memory_id)
        if memory is None:
            return None

        if fact is not None:
            memory.fact = fact

        if embedding is not None:
            memory.embedding = embedding

        if topics is not None:
            memory.topics = topics

        if metadata is not None:
            memory.metadata.update(metadata)

        logger.debug("Memory updated", memory_id=memory_id)
        return memory

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory."""
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
        """Delete all memories for a user (GDPR compliance)."""
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

    async def _evict_oldest_memory(self, user_id: str) -> None:
        """Evict the oldest accessed memory for a user."""
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
        """Get backend statistics."""
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
            "backend": "in_memory",
            "total_memories": total_memories,
            "total_users": total_users,
            "scope_distribution": scope_counts,
            "config": {
                "max_per_user": self.config.max_memories_per_user,
            },
        }
