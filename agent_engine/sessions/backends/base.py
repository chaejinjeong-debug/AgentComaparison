"""Abstract base class for session storage backends."""

from abc import ABC, abstractmethod
from typing import Any

from agent_engine.sessions.models import EventAuthor, Session, SessionEvent


class SessionBackend(ABC):
    """Abstract base class for session storage backends.

    This defines the interface that all session backends must implement.
    Implementations can store sessions in memory, VertexAI, or other storage systems.
    """

    @abstractmethod
    async def create_session(
        self,
        user_id: str,
        ttl_seconds: int,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session for a user.

        Args:
            user_id: User identifier
            ttl_seconds: Session TTL in seconds
            metadata: Optional session metadata

        Returns:
            Newly created Session
        """
        pass

    @abstractmethod
    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session if found and not expired, None otherwise
        """
        pass

    @abstractmethod
    async def append_event(
        self,
        session_id: str,
        author: EventAuthor,
        content: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> SessionEvent | None:
        """Append an event to a session.

        Args:
            session_id: Session identifier
            author: Event author (user, agent, or system)
            content: Event content/payload
            metadata: Optional event metadata

        Returns:
            Created SessionEvent or None if session not found
        """
        pass

    @abstractmethod
    async def list_events(
        self,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[SessionEvent]:
        """List events from a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of events to return
            offset: Number of events to skip

        Returns:
            List of SessionEvents
        """
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        pass

    @abstractmethod
    async def cleanup_expired_sessions(self) -> int:
        """Remove all expired sessions.

        Returns:
            Number of sessions removed
        """
        pass

    @abstractmethod
    async def get_user_sessions(
        self,
        user_id: str,
        include_expired: bool = False,
    ) -> list[Session]:
        """Get all sessions for a user.

        Args:
            user_id: User identifier
            include_expired: Whether to include expired sessions

        Returns:
            List of user's sessions
        """
        pass

    @abstractmethod
    async def extend_session(
        self,
        session_id: str,
        additional_seconds: int,
    ) -> Session | None:
        """Extend a session's TTL.

        Args:
            session_id: Session identifier
            additional_seconds: Seconds to add to TTL

        Returns:
            Updated Session or None if not found
        """
        pass

    @abstractmethod
    def get_session_count(self) -> int:
        """Get the total number of active sessions."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        pass
