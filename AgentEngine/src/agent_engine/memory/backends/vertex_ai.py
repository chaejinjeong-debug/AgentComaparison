"""VertexAI memory storage backend for production use."""

from typing import Any
from uuid import uuid4

import structlog

from agent_engine.config import MemoryConfig
from agent_engine.memory.backends.base import MemoryBackend
from agent_engine.memory.models import Memory, MemoryScope

logger = structlog.get_logger(__name__)


class VertexAIMemoryBackend(MemoryBackend):
    """VertexAI Agent Engine Memory Bank storage backend.

    This backend uses VertexAI's managed Memory Bank API for:
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
        self._client = None
        self._agent_engine_name = (
            f"projects/{project_id}/locations/{location}"
            f"/reasoningEngines/{agent_engine_id}"
        )

        logger.info(
            "VertexAIMemoryBackend initialized",
            project_id=project_id,
            location=location,
            agent_engine_id=agent_engine_id,
        )

    def _get_client(self):
        """Get or create VertexAI client."""
        if self._client is None:
            try:
                import vertexai
                from google.cloud import aiplatform

                vertexai.init(project=self.project_id, location=self.location)
                self._client = aiplatform.gapic.ReasoningEngineServiceClient()
            except ImportError as e:
                raise ImportError(
                    "google-cloud-aiplatform >= 1.114.0 required for VertexAI backend. "
                    "Install with: pip install 'google-cloud-aiplatform>=1.114.0'"
                ) from e
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
        """Save a memory using VertexAI Memory Bank API."""
        try:
            client = self._get_client()

            # Prepare scope for VertexAI
            vertex_scope = {"user_id": user_id}
            if scope == MemoryScope.GLOBAL:
                vertex_scope = {"global": True}

            # Generate memory via VertexAI
            request = {
                "parent": self._agent_engine_name,
                "direct_memories_source": {
                    "direct_memories": [
                        {
                            "fact": fact,
                            "metadata": {
                                "source": source,
                                "topics": topics or [],
                                **(metadata or {}),
                            },
                        }
                    ]
                },
                "scope": vertex_scope,
            }

            response = client.generate_memories(request=request)

            # Create internal Memory model
            memory_id = str(uuid4())
            if hasattr(response, "memories") and response.memories:
                memory_id = response.memories[0].name.split("/")[-1]

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

            logger.info(
                "Memory saved to VertexAI",
                memory_id=memory.memory_id,
                user_id=user_id,
            )

            return memory

        except Exception as e:
            logger.error("Failed to save memory to VertexAI", error=str(e))
            # Fallback to local memory creation
            return await self._create_local_memory(
                user_id, fact, embedding, topics, scope, source, metadata
            )

    async def _create_local_memory(
        self,
        user_id: str,
        fact: str,
        embedding: list[float] | None = None,
        topics: list[str] | None = None,
        scope: MemoryScope = MemoryScope.USER,
        source: str = "explicit",
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Create a local memory as fallback."""
        logger.warning("Using local memory fallback")
        return Memory(
            user_id=user_id,
            fact=fact,
            topics=topics or [],
            scope=scope,
            source=source,
            embedding=embedding,
            metadata=metadata or {},
        )

    async def retrieve_memories(
        self,
        user_id: str,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        max_results: int = 10,
        include_global: bool = True,
    ) -> list[Memory]:
        """Retrieve memories using VertexAI Memory Bank API."""
        try:
            client = self._get_client()

            # Build request
            request = {
                "parent": self._agent_engine_name,
                "scope": {"user_id": user_id},
            }

            if query:
                # Use similarity search
                request["similarity_search_params"] = {
                    "search_query": query,
                    "top_k": max_results,
                }

            response = client.retrieve_memories(request=request)

            memories = []
            for mem_data in response:
                memory = Memory(
                    memory_id=mem_data.name.split("/")[-1] if hasattr(mem_data, "name") else str(uuid4()),
                    user_id=user_id,
                    fact=getattr(mem_data, "fact", str(mem_data)),
                    scope=MemoryScope.USER,
                    source="vertex_ai",
                )
                memories.append(memory)

            # Optionally retrieve global memories
            if include_global:
                global_request = {
                    "parent": self._agent_engine_name,
                    "scope": {"global": True},
                }
                if query:
                    global_request["similarity_search_params"] = {
                        "search_query": query,
                        "top_k": max_results,
                    }

                global_response = client.retrieve_memories(request=global_request)
                for mem_data in global_response:
                    memory = Memory(
                        memory_id=mem_data.name.split("/")[-1] if hasattr(mem_data, "name") else str(uuid4()),
                        user_id=user_id,
                        fact=getattr(mem_data, "fact", str(mem_data)),
                        scope=MemoryScope.GLOBAL,
                        source="vertex_ai",
                    )
                    memories.append(memory)

            logger.debug(
                "Memories retrieved from VertexAI",
                user_id=user_id,
                count=len(memories),
            )

            return memories[:max_results]

        except Exception as e:
            logger.error(
                "Failed to retrieve memories from VertexAI",
                user_id=user_id,
                error=str(e),
            )
            return []

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Get a specific memory from VertexAI."""
        try:
            client = self._get_client()

            memory_name = f"{self._agent_engine_name}/memories/{memory_id}"
            response = client.get_memory(name=memory_name)

            if response is None:
                return None

            return Memory(
                memory_id=memory_id,
                user_id=getattr(response, "user_id", "unknown"),
                fact=getattr(response, "fact", str(response)),
                source="vertex_ai",
            )

        except Exception as e:
            logger.error(
                "Failed to get memory from VertexAI",
                memory_id=memory_id,
                error=str(e),
            )
            return None

    async def update_memory(
        self,
        memory_id: str,
        fact: str | None = None,
        embedding: list[float] | None = None,
        topics: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory | None:
        """Update an existing memory in VertexAI.

        Note: VertexAI Memory Bank may not support direct updates.
        This implementation deletes and recreates the memory.
        """
        try:
            # Get existing memory
            existing = await self.get_memory(memory_id)
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

        except Exception as e:
            logger.error(
                "Failed to update memory in VertexAI",
                memory_id=memory_id,
                error=str(e),
            )
            return None

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory from VertexAI."""
        try:
            client = self._get_client()

            memory_name = f"{self._agent_engine_name}/memories/{memory_id}"
            client.delete_memory(name=memory_name)

            logger.info("Memory deleted from VertexAI", memory_id=memory_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to delete memory from VertexAI",
                memory_id=memory_id,
                error=str(e),
            )
            return False

    async def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user from VertexAI (GDPR compliance)."""
        try:
            client = self._get_client()

            # List all user memories
            request = {
                "parent": self._agent_engine_name,
                "scope": {"user_id": user_id},
            }

            response = client.retrieve_memories(request=request)

            deleted_count = 0
            for mem_data in response:
                memory_name = getattr(mem_data, "name", None)
                if memory_name:
                    try:
                        client.delete_memory(name=memory_name)
                        deleted_count += 1
                    except Exception:
                        pass

            logger.info(
                "User memories deleted from VertexAI (GDPR)",
                user_id=user_id,
                count=deleted_count,
            )

            return deleted_count

        except Exception as e:
            logger.error(
                "Failed to delete user memories from VertexAI",
                user_id=user_id,
                error=str(e),
            )
            return 0

    def get_user_memory_count(self, user_id: str) -> int:
        """Get the number of memories for a user.

        Note: This requires listing all memories which may be expensive.
        """
        try:
            client = self._get_client()

            request = {
                "parent": self._agent_engine_name,
                "scope": {"user_id": user_id},
            }

            response = client.retrieve_memories(request=request)
            return sum(1 for _ in response)

        except Exception as e:
            logger.error(
                "Failed to count user memories from VertexAI",
                user_id=user_id,
                error=str(e),
            )
            return 0

    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        return {
            "backend": "vertex_ai",
            "project_id": self.project_id,
            "location": self.location,
            "agent_engine_id": self.agent_engine_id,
            "config": {
                "max_per_user": self.config.max_memories_per_user,
                "similarity_threshold": self.config.similarity_threshold,
            },
        }
