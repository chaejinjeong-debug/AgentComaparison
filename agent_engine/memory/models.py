"""Memory models for the Agent Engine platform."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryScope(str, Enum):
    """Scope of memory visibility."""

    USER = "user"  # Personal to a specific user
    SESSION = "session"  # Limited to a session
    GLOBAL = "global"  # Available across all users


class Memory(BaseModel):
    """Represents a single memory/fact.

    Attributes:
        memory_id: Unique memory identifier
        user_id: User who owns this memory
        fact: The memory content/fact
        topics: Related topics/tags
        scope: Memory visibility scope
        source: Where this memory came from
        created_at: When the memory was created
        accessed_at: Last access time
        access_count: Number of times accessed
        embedding: Vector embedding for similarity search
        metadata: Optional metadata
    """

    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    fact: str
    topics: list[str] = Field(default_factory=list)
    scope: MemoryScope = Field(default=MemoryScope.USER)
    source: str = Field(default="conversation")  # conversation, explicit, system
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0)
    embedding: list[float] | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def mark_accessed(self) -> None:
        """Mark the memory as accessed."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert memory to dictionary."""
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "fact": self.fact,
            "topics": self.topics,
            "scope": self.scope.value,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "metadata": self.metadata,
        }

    def to_context_string(self) -> str:
        """Convert memory to a context string for LLM."""
        topic_str = f" (topics: {', '.join(self.topics)})" if self.topics else ""
        return f"- {self.fact}{topic_str}"


class MemorySearchResult(BaseModel):
    """Result from memory similarity search."""

    memory: Memory
    score: float = Field(ge=0.0, le=1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
        }
