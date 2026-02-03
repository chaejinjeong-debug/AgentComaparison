"""Memory retriever with similarity search for the Agent Engine platform.

Implements MB-003: Similarity search implementation.
"""

import math
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from agent_engine.memory.models import Memory

from agent_engine.memory.models import MemorySearchResult

logger = structlog.get_logger(__name__)


class MemoryRetriever:
    """Retrieves memories using similarity search.

    This implementation uses a simple embedding-based approach.
    For production, consider using a vector database like Vertex AI
    Vector Search or Pinecone.

    Attributes:
        similarity_threshold: Minimum similarity score to return
    """

    def __init__(self, similarity_threshold: float = 0.7) -> None:
        """Initialize the memory retriever.

        Args:
            similarity_threshold: Minimum similarity score (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        logger.info(
            "MemoryRetriever initialized",
            similarity_threshold=similarity_threshold,
        )

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for text.

        This is a simplified implementation using character-based
        features. In production, use a proper embedding model like
        text-embedding-004 from Vertex AI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (normalized)
        """
        # Simplified embedding using character/word features
        # In production, use Vertex AI text-embedding-004
        text_lower = text.lower()
        features: list[float] = []

        # Character frequency features (26 letters)
        for c in "abcdefghijklmnopqrstuvwxyz":
            features.append(text_lower.count(c) / max(len(text), 1))

        # Word-based features
        words = text_lower.split()
        features.append(len(words) / 100.0)  # Word count (normalized)
        features.append(len(text) / 1000.0)  # Character count (normalized)

        # Common word presence features
        common_words = [
            "the", "is", "are", "was", "were", "have", "has",
            "like", "want", "need", "work", "live", "name",
        ]
        for word in common_words:
            features.append(1.0 if word in words else 0.0)

        # Normalize the vector
        magnitude = math.sqrt(sum(f * f for f in features))
        if magnitude > 0:
            features = [f / magnitude for f in features]

        return features

    async def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0.0-1.0)
        """
        if len(embedding1) != len(embedding2):
            # Pad shorter vector with zeros
            max_len = max(len(embedding1), len(embedding2))
            embedding1 = embedding1 + [0.0] * (max_len - len(embedding1))
            embedding2 = embedding2 + [0.0] * (max_len - len(embedding2))

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(b * b for b in embedding2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        similarity = dot_product / (magnitude1 * magnitude2)

        # Clamp to [0, 1] range
        return max(0.0, min(1.0, similarity))

    async def search(
        self,
        query: str,
        memories: list["Memory"],
        max_results: int = 5,
    ) -> list[MemorySearchResult]:
        """Search memories by similarity to query.

        MB-003: Similarity search implementation.

        Args:
            query: Search query
            memories: List of memories to search
            max_results: Maximum results to return

        Returns:
            List of MemorySearchResult sorted by score
        """
        if not memories:
            return []

        query_embedding = await self.generate_embedding(query)
        results: list[MemorySearchResult] = []

        for memory in memories:
            if memory.embedding is None:
                # Generate embedding if not present
                memory.embedding = await self.generate_embedding(memory.fact)

            score = await self.compute_similarity(query_embedding, memory.embedding)

            if score >= self.similarity_threshold:
                results.append(MemorySearchResult(memory=memory, score=score))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        logger.debug(
            "Memory search completed",
            query=query[:50],
            candidates=len(memories),
            results=len(results[:max_results]),
        )

        return results[:max_results]

    async def get_relevant_memories(
        self,
        user_id: str,
        query: str,
        memories: list["Memory"],
        max_results: int = 5,
    ) -> list[str]:
        """Get relevant memories as formatted strings.

        Convenience method for injecting memories into prompts.

        Args:
            user_id: User identifier (for logging)
            query: Search query
            memories: List of memories to search
            max_results: Maximum results to return

        Returns:
            List of memory fact strings
        """
        results = await self.search(
            query=query,
            memories=memories,
            max_results=max_results,
        )

        facts = [r.memory.fact for r in results]

        logger.debug(
            "Relevant memories retrieved",
            user_id=user_id,
            query=query[:50],
            count=len(facts),
        )

        return facts
