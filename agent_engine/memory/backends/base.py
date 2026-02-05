"""Abstract base class for memory storage backends."""

from abc import ABC, abstractmethod
from typing import Any

from agent_engine.memory.models import Memory, MemoryScope


class MemoryBackend(ABC):
    """Abstract base class for memory storage backends.

    This defines the interface that all memory backends must implement.
    Implementations can store memories in memory, VertexAI, or other storage systems.
    """

    @abstractmethod
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
        """Save a memory for a user.

        Args:
            user_id: User identifier
            fact: The memory/fact to store
            embedding: Pre-computed embedding (optional)
            topics: Related topics/tags
            scope: Memory visibility scope
            source: Where this memory came from
            metadata: Optional metadata

        Returns:
            Created Memory
        """
        pass

    @abstractmethod
    async def retrieve_memories(
        self,
        user_id: str,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        max_results: int = 10,
        include_global: bool = True,
    ) -> list[Memory]:
        """Retrieve memories for a user.

        Args:
            user_id: User identifier
            query: Optional search query for similarity matching
            query_embedding: Pre-computed query embedding
            max_results: Maximum number of memories to return
            include_global: Whether to include global scope memories

        Returns:
            List of relevant memories
        """
        pass

    @abstractmethod
    async def get_memory(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory if found, None otherwise
        """
        pass

    @abstractmethod
    async def update_memory(
        self,
        memory_id: str,
        fact: str | None = None,
        embedding: list[float] | None = None,
        topics: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory | None:
        """Update an existing memory.

        Args:
            memory_id: Memory identifier
            fact: New fact content
            embedding: New embedding
            topics: New topics
            metadata: New metadata (merged with existing)

        Returns:
            Updated Memory or None if not found
        """
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory.

        Args:
            memory_id: Memory identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user (GDPR compliance).

        Args:
            user_id: User identifier

        Returns:
            Number of memories deleted
        """
        pass

    @abstractmethod
    def get_user_memory_count(self, user_id: str) -> int:
        """Get the number of memories for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of memories
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics.

        Returns:
            Dictionary with backend statistics
        """
        pass
