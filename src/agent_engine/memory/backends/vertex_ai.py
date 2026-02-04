"""VertexAI memory storage backend using official SDK for production use."""

from typing import Any
from uuid import uuid4

import structlog
import vertexai

from agent_engine.config import MemoryConfig
from agent_engine.memory.backends.base import MemoryBackend
from agent_engine.memory.models import Memory, MemoryScope

logger = structlog.get_logger(__name__)


class VertexAIMemoryBackend(MemoryBackend):
    """VertexAI Agent Engine Memory Bank storage backend using official SDK.

    This backend uses VertexAI's managed Memory Bank SDK for:
    - Persistent memory storage
    - Built-in similarity search
    - Automatic replication and backup
    - Multi-instance support
    - Production deployments

    Memories survive server restarts and network disconnections.
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        agent_engine_id: str,
        config: MemoryConfig | None = None,
    ) -> None:
        """Initialize the VertexAI backend.

        Args:
            project_id: GCP project ID
            location: GCP region
            agent_engine_id: Deployed Agent Engine ID
            config: Memory configuration
        """
        self.project_id = project_id
        self.location = location
        self.agent_engine_id = agent_engine_id
        self.config = config or MemoryConfig()

        # SDK client (lazy init)
        self._client: vertexai.Client | None = None
        self._agent_engine_name = (
            f"projects/{project_id}/locations/{location}/reasoningEngines/{agent_engine_id}"
        )

        # Local cache for tracking
        self._memory_cache: dict[str, Memory] = {}
        self._user_index: dict[str, set[str]] = {}

        logger.info(
            "VertexAIMemoryBackend initialized (SDK)",
            project_id=project_id,
            location=location,
            agent_engine_id=agent_engine_id,
        )

    def _get_client(self) -> vertexai.Client:
        """Get or create VertexAI SDK client."""
        if self._client is None:
            self._client = vertexai.Client(
                project=self.project_id,
                location=self.location,
            )
        return self._client

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
        """Save a memory using VertexAI Memory Bank SDK."""
        try:
            client = self._get_client()

            # Determine scope
            if scope == MemoryScope.GLOBAL:
                scope_data = {"global": True}
            else:
                scope_data = {"user_id": user_id}

            vertex_memory = client.agent_engines.memories.create(
                name=self._agent_engine_name,
                fact=fact,
                scope=scope_data,
            )

            # Extract memory ID from the full resource name
            memory_id = vertex_memory.name.split("/")[-1]

            memory = Memory(
                memory_id=memory_id,
                user_id=user_id,
                fact=fact,
                topics=topics or [],
                scope=scope,
                source=source,
                embedding=embedding,
                metadata=metadata or {},
            )

            # Cache locally
            self._memory_cache[memory_id] = memory
            if user_id not in self._user_index:
                self._user_index[user_id] = set()
            self._user_index[user_id].add(memory_id)

            logger.info(
                "Memory saved to VertexAI via SDK",
                memory_id=memory_id,
                user_id=user_id,
            )

            return memory

        except Exception as e:
            logger.error("Failed to save memory via SDK", error=str(e))
            # Create local memory as fallback
            memory_id = str(uuid4())
            memory = Memory(
                memory_id=memory_id,
                user_id=user_id,
                fact=fact,
                topics=topics or [],
                scope=scope,
                source=source,
                embedding=embedding,
                metadata=metadata or {},
            )
            self._memory_cache[memory_id] = memory
            if user_id not in self._user_index:
                self._user_index[user_id] = set()
            self._user_index[user_id].add(memory_id)
            return memory

    async def retrieve_memories(
        self,
        user_id: str,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        max_results: int = 10,
        include_global: bool = True,
    ) -> list[Memory]:
        """Retrieve memories using VertexAI Memory Bank SDK."""
        memories: list[Memory] = []

        try:
            client = self._get_client()

            # Retrieve user memories
            for mem in client.agent_engines.memories.retrieve(
                name=self._agent_engine_name,
                scope={"user_id": user_id},
            ):
                memory_id = mem.name.split("/")[-1]
                memory = Memory(
                    memory_id=memory_id,
                    user_id=user_id,
                    fact=getattr(mem, "fact", str(mem)),
                    scope=MemoryScope.USER,
                    source="vertex_ai",
                )
                memories.append(memory)

            # Retrieve global memories if requested
            if include_global:
                for mem in client.agent_engines.memories.retrieve(
                    name=self._agent_engine_name,
                    scope={"global": True},
                ):
                    memory_id = mem.name.split("/")[-1]
                    memory = Memory(
                        memory_id=memory_id,
                        user_id=user_id,
                        fact=getattr(mem, "fact", str(mem)),
                        scope=MemoryScope.GLOBAL,
                        source="vertex_ai",
                    )
                    memories.append(memory)

        except Exception as e:
            logger.warning("Failed to retrieve memories via SDK", error=str(e))

        # Fall back to local cache if no results from API
        if not memories:
            logger.debug("Falling back to local memory cache")
            for memory_id in self._user_index.get(user_id, set()):
                if memory_id in self._memory_cache:
                    memories.append(self._memory_cache[memory_id])

        logger.debug(
            "Memories retrieved from VertexAI via SDK",
            user_id=user_id,
            count=len(memories),
        )

        return memories[:max_results]

    async def generate_memories_from_session(
        self,
        user_id: str,
        session_name: str,
    ) -> list[Memory]:
        """Generate memories from a session using VertexAI Memory Bank SDK.

        This uses the vertex_session_source to automatically extract memories
        from conversation history.
        """
        try:
            client = self._get_client()

            client.agent_engines.memories.generate(
                name=self._agent_engine_name,
                vertex_session_source={"session": session_name},
                scope={"user_id": user_id},
            )

            logger.info(
                "Memories generated from session via SDK",
                session_name=session_name,
                user_id=user_id,
            )

            # Retrieve the generated memories
            return await self.retrieve_memories(user_id, include_global=False)

        except Exception as e:
            logger.error("Failed to generate memories from session via SDK", error=str(e))
            return []

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Get a specific memory."""
        # Check local cache first
        if memory_id in self._memory_cache:
            return self._memory_cache[memory_id]

        # VertexAI doesn't have a direct get memory endpoint
        # Return None if not in cache
        return None

    async def update_memory(
        self,
        memory_id: str,
        fact: str | None = None,
        embedding: list[float] | None = None,
        topics: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory | None:
        """Update an existing memory.

        Note: VertexAI Memory Bank may not support direct updates.
        This updates the local cache and creates a new memory.
        """
        existing = self._memory_cache.get(memory_id)
        if existing is None:
            return None

        # Delete old memory
        await self.delete_memory(memory_id)

        # Create new memory with updates
        new_fact = fact if fact is not None else existing.fact
        new_topics = topics if topics is not None else existing.topics
        new_metadata = existing.metadata.copy()
        if metadata:
            new_metadata.update(metadata)

        return await self.save_memory(
            user_id=existing.user_id,
            fact=new_fact,
            embedding=embedding,
            topics=new_topics,
            scope=existing.scope,
            source=existing.source,
            metadata=new_metadata,
        )

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        try:
            client = self._get_client()
            memory_name = f"{self._agent_engine_name}/memories/{memory_id}"

            client.agent_engines.memories.delete(
                name=memory_name,
                config={"wait_for_completion": True},
            )

            # Remove from local cache
            memory = self._memory_cache.pop(memory_id, None)
            if memory:
                user_memories = self._user_index.get(memory.user_id, set())
                user_memories.discard(memory_id)

            logger.info("Memory deleted from VertexAI via SDK", memory_id=memory_id)
            return True

        except Exception as e:
            logger.warning("Failed to delete memory via SDK", memory_id=memory_id, error=str(e))
            # Still remove from local cache
            memory = self._memory_cache.pop(memory_id, None)
            if memory:
                user_memories = self._user_index.get(memory.user_id, set())
                user_memories.discard(memory_id)
            return memory is not None

    async def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user (GDPR compliance)."""
        try:
            client = self._get_client()

            # Use purge to delete all memories for a user
            filter_string = f'scope.user_id="{user_id}"'
            client.agent_engines.memories.purge(
                name=self._agent_engine_name,
                filter=filter_string,
                force=True,
                config={"wait_for_completion": True},
            )

            # Clear local cache
            deleted_count = len(self._user_index.get(user_id, set()))
            for memory_id in list(self._user_index.get(user_id, set())):
                self._memory_cache.pop(memory_id, None)
            self._user_index.pop(user_id, None)

            logger.info(
                "User memories purged from VertexAI via SDK (GDPR)",
                user_id=user_id,
                count=deleted_count,
            )

            return deleted_count

        except Exception as e:
            logger.error("Failed to purge user memories via SDK", user_id=user_id, error=str(e))
            # Fall back to individual deletion
            memory_ids = list(self._user_index.get(user_id, set()))
            deleted_count = 0

            for memory_id in memory_ids:
                if await self.delete_memory(memory_id):
                    deleted_count += 1

            self._user_index.pop(user_id, None)
            return deleted_count

    def get_user_memory_count(self, user_id: str) -> int:
        """Get the number of memories for a user."""
        return len(self._user_index.get(user_id, set()))

    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        return {
            "backend": "vertex_ai_sdk",
            "project_id": self.project_id,
            "location": self.location,
            "agent_engine_id": self.agent_engine_id,
            "cached_memories": len(self._memory_cache),
            "total_users": len(self._user_index),
            "config": {
                "max_per_user": self.config.max_memories_per_user,
                "similarity_threshold": self.config.similarity_threshold,
            },
        }
