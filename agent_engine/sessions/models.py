"""Session and event models for the Agent Engine platform."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class EventAuthor(str, Enum):
    """Author type for session events."""

    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class SessionEvent(BaseModel):
    """Represents a single event in a session.

    Attributes:
        event_id: Unique event identifier
        author: Who created this event (user, agent, or system)
        content: Event content/payload
        timestamp: When the event occurred
        metadata: Optional metadata
    """

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    author: EventAuthor
    content: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "author": self.author.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class Session(BaseModel):
    """Represents a conversation session.

    Attributes:
        session_id: Unique session identifier
        user_id: User who owns this session
        created_at: Session creation time
        updated_at: Last update time
        expires_at: Session expiration time
        events: List of session events
        metadata: Optional session metadata
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    events: list[SessionEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.utcnow() > self.expires_at

    @property
    def event_count(self) -> int:
        """Get the number of events in the session."""
        return len(self.events)

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "events": [event.to_dict() for event in self.events],
            "metadata": self.metadata,
            "event_count": self.event_count,
            "is_expired": self.is_expired,
        }
