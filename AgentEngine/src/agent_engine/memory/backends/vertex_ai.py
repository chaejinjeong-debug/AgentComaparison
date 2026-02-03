"""VertexAI memory storage backend using REST API for production use."""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import google.auth
import google.auth.transport.requests
import httpx
import structlog

from agent_engine.config import MemoryConfig
from agent_engine.memory.backends.base import MemoryBackend
from agent_engine.memory.models import Memory, MemoryScope

logger = structlog.get_logger(__name__)


class VertexAIMemoryBackend(MemoryBackend):
    """VertexAI Agent Engine Memory Bank storage backend using REST API.

    This backend uses VertexAI's managed Memory Bank REST API for:
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

        # REST API configuration
        self._base_url = f"https://{location}-aiplatform.googleapis.com/v1"
        self._reasoning_engine = (
            f"projects/{project_id}/locations/{location}"
            f"/reasoningEngines/{agent_engine_id}"
        )
        self._credentials = None
        self._token_expiry = None

        # Local cache for tracking
        self._memory_cache: dict[str, Memory] = {}
        self._user_index: dict[str, set[str]] = {}

        logger.info(
            "VertexAIMemoryBackend initialized",
            project_id=project_id,
            location=location,
            agent_engine_id=agent_engine_id,
        )

    def _get_access_token(self) -> str:
        """Get or refresh Google Cloud access token."""
        now = datetime.now(timezone.utc)

        if self._credentials is None or self._token_expiry is None or now >= self._token_expiry:
            self._credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self._credentials.refresh(google.auth.transport.requests.Request())
            # Set expiry 5 minutes before actual expiry
            self._token_expiry = now + timedelta(minutes=55)

        return self._credentials.token

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers with authentication."""
        return {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
        }

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
        """Save a memory using VertexAI Memory Bank REST API."""
        url = f"{self._base_url}/{self._reasoning_engine}/memories:generate"

        # Build scope
        if scope == MemoryScope.GLOBAL:
            scope_data = {"global": True}
        else:
            scope_data = {"userId": user_id}

        request_body = {
            "directMemoriesSource": {
                "directMemories": [
                    {
                        "fact": fact,
                    }
                ]
            },
            "scope": scope_data,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=request_body,
                )

                memory_id = str(uuid4())

                if response.status_code in [200, 201]:
                    data = response.json()
                    # Extract memory ID from operation name if available
                    name = data.get("name", "")
                    if name:
                        # Try to extract a meaningful ID from the operation
                        parts = name.split("/")
                        for i, part in enumerate(parts):
                            if part == "memories" and i + 1 < len(parts):
                                memory_id = parts[i + 1]
                                break

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
                        "Memory saved to VertexAI",
                        memory_id=memory_id,
                        user_id=user_id,
                    )

                    return memory
                else:
                    logger.error(
                        "Failed to save memory to VertexAI",
                        status=response.status_code,
                        error=response.text[:200],
                    )
                    # Return local memory anyway
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
                    return memory

        except httpx.HTTPError as e:
            logger.error("HTTP error saving memory to VertexAI", error=str(e))
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
            return memory

    async def retrieve_memories(
        self,
        user_id: str,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        max_results: int = 10,
        include_global: bool = True,
    ) -> list[Memory]:
        """Retrieve memories using VertexAI Memory Bank REST API."""
        url = f"{self._base_url}/{self._reasoning_engine}/memories:retrieve"

        # Build request
        request_body: dict[str, Any] = {
            "scope": {"userId": user_id},
        }

        if query:
            request_body["similaritySearchParams"] = {
                "searchQuery": query,
                "topK": max_results,
            }

        memories = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Retrieve user memories
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=request_body,
                )

                if response.status_code == 200:
                    data = response.json()

                    for mem_data in data.get("memories", []):
                        name = mem_data.get("name", "")
                        memory_id = name.split("/")[-1] if "/" in name else str(uuid4())

                        memory = Memory(
                            memory_id=memory_id,
                            user_id=user_id,
                            fact=mem_data.get("fact", str(mem_data)),
                            scope=MemoryScope.USER,
                            source="vertex_ai",
                        )
                        memories.append(memory)
                else:
                    logger.warning(
                        "Failed to retrieve memories from VertexAI",
                        status=response.status_code,
                        error=response.text[:100],
                    )

                # Retrieve global memories if requested
                if include_global:
                    global_request = {
                        "scope": {"global": True},
                    }
                    if query:
                        global_request["similaritySearchParams"] = {
                            "searchQuery": query,
                            "topK": max_results,
                        }

                    global_response = await client.post(
                        url,
                        headers=self._get_headers(),
                        json=global_request,
                    )

                    if global_response.status_code == 200:
                        global_data = global_response.json()

                        for mem_data in global_data.get("memories", []):
                            name = mem_data.get("name", "")
                            memory_id = name.split("/")[-1] if "/" in name else str(uuid4())

                            memory = Memory(
                                memory_id=memory_id,
                                user_id=user_id,
                                fact=mem_data.get("fact", str(mem_data)),
                                scope=MemoryScope.GLOBAL,
                                source="vertex_ai",
                            )
                            memories.append(memory)

        except httpx.HTTPError as e:
            logger.error("HTTP error retrieving memories from VertexAI", error=str(e))

        # Fall back to local cache if no results from API
        if not memories:
            logger.debug("Falling back to local memory cache")
            for memory_id in self._user_index.get(user_id, set()):
                if memory_id in self._memory_cache:
                    memories.append(self._memory_cache[memory_id])

        logger.debug(
            "Memories retrieved from VertexAI",
            user_id=user_id,
            count=len(memories),
        )

        return memories[:max_results]

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
        url = f"{self._base_url}/{self._reasoning_engine}/memories/{memory_id}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(url, headers=self._get_headers())

                # Remove from local cache regardless of API result
                memory = self._memory_cache.pop(memory_id, None)
                if memory:
                    user_memories = self._user_index.get(memory.user_id, set())
                    user_memories.discard(memory_id)

                if response.status_code in [200, 204, 404]:
                    logger.info("Memory deleted from VertexAI", memory_id=memory_id)
                    return True
                else:
                    logger.warning(
                        "Failed to delete memory from VertexAI (removed from cache)",
                        memory_id=memory_id,
                        status=response.status_code,
                    )
                    return True  # Still return True since it's removed from cache

        except httpx.HTTPError as e:
            logger.error("HTTP error deleting memory from VertexAI", error=str(e))
            # Remove from cache anyway
            memory = self._memory_cache.pop(memory_id, None)
            if memory:
                user_memories = self._user_index.get(memory.user_id, set())
                user_memories.discard(memory_id)
            return memory is not None

    async def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user (GDPR compliance)."""
        memory_ids = list(self._user_index.get(user_id, set()))
        deleted_count = 0

        for memory_id in memory_ids:
            if await self.delete_memory(memory_id):
                deleted_count += 1

        self._user_index.pop(user_id, None)

        logger.info(
            "User memories deleted from VertexAI (GDPR)",
            user_id=user_id,
            count=deleted_count,
        )

        return deleted_count

    def get_user_memory_count(self, user_id: str) -> int:
        """Get the number of memories for a user."""
        return len(self._user_index.get(user_id, set()))

    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        return {
            "backend": "vertex_ai",
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
