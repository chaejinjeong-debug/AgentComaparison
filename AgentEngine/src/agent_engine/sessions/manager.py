"""Session manager for the Agent Engine platform.

Implements SM-001~SM-005:
- SM-001: Session creation
- SM-002: Event recording (AppendEvent)
- SM-003: History retrieval (ListEvents)
- SM-004: TTL management (24 hours default)
- SM-005: Session deletion
"""

from datetime import datetime, timedelta
from typing import Any

import structlog

from agent_engine.config import SessionConfig
from agent_engine.sessions.models import EventAuthor, Session, SessionEvent

logger = structlog.get_logger(__name__)


class SessionManager:
    """Manages conversation sessions with TTL support.

    This manager handles session lifecycle including creation, event recording,
    history retrieval, and automatic expiration.

    Attributes:
        config: Session configuration
        _sessions: In-memory session storage (for local/testing use)
    """

    def __init__(self, config: SessionConfig | None = None) -> None:
        """Initialize the session manager.

        Args:
            config: Session configuration, uses defaults if not provided
        """
        self.config = config or SessionConfig()
        self._sessions: dict[str, Session] = {}
        logger.info(
            "SessionManager initialized",
            ttl_seconds=self.config.default_ttl_seconds,
            max_events=self.config.max_events_per_session,
        )

    async def create_session(
        self,
        user_id: str,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session for a user.

        SM-001: Session creation functionality.

        Args:
            user_id: User identifier
            ttl_seconds: Session TTL in seconds, defaults to config value
            metadata: Optional session metadata

        Returns:
            Newly created Session
        """
        ttl = ttl_seconds or self.config.default_ttl_seconds
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)

        session = Session(
            user_id=user_id,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        self._sessions[session.session_id] = session

        logger.info(
            "Session created",
            session_id=session.session_id,
            user_id=user_id,
            ttl_seconds=ttl,
            expires_at=expires_at.isoformat(),
        )

        return session

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session if found and not expired, None otherwise
        """
        session = self._sessions.get(session_id)

        if session is None:
            logger.debug("Session not found", session_id=session_id)
            return None

        if session.is_expired:
            logger.info("Session expired, removing", session_id=session_id)
            await self.delete_session(session_id)
            return None

        return session

    async def append_event(
        self,
        session_id: str,
        author: EventAuthor,
        content: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> SessionEvent | None:
        """Append an event to a session.

        SM-002: Session event recording (AppendEvent).

        Args:
            session_id: Session identifier
            author: Event author (user, agent, or system)
            content: Event content/payload
            metadata: Optional event metadata

        Returns:
            Created SessionEvent or None if session not found
        """
        session = await self.get_session(session_id)

        if session is None:
            logger.warning("Cannot append event: session not found", session_id=session_id)
            return None

        # Check max events limit
        if session.event_count >= self.config.max_events_per_session:
            logger.warning(
                "Session event limit reached",
                session_id=session_id,
                limit=self.config.max_events_per_session,
            )
            # Remove oldest event to make room
            session.events.pop(0)

        event = SessionEvent(
            author=author,
            content=content,
            metadata=metadata or {},
        )

        session.events.append(event)
        session.updated_at = datetime.utcnow()

        logger.debug(
            "Event appended",
            session_id=session_id,
            event_id=event.event_id,
            author=author.value,
        )

        return event

    async def list_events(
        self,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[SessionEvent]:
        """List events from a session.

        SM-003: Session history retrieval (ListEvents).

        Args:
            session_id: Session identifier
            limit: Maximum number of events to return
            offset: Number of events to skip

        Returns:
            List of SessionEvents
        """
        session = await self.get_session(session_id)

        if session is None:
            logger.debug("Cannot list events: session not found", session_id=session_id)
            return []

        events = session.events[offset:]

        if limit is not None:
            events = events[:limit]

        logger.debug(
            "Events listed",
            session_id=session_id,
            total_events=session.event_count,
            returned_events=len(events),
        )

        return events

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        SM-005: Session deletion functionality.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        if session_id not in self._sessions:
            logger.debug("Cannot delete: session not found", session_id=session_id)
            return False

        del self._sessions[session_id]
        logger.info("Session deleted", session_id=session_id)
        return True

    async def cleanup_expired_sessions(self) -> int:
        """Remove all expired sessions.

        SM-004: TTL management support.

        Returns:
            Number of sessions removed
        """
        expired_ids = [
            sid for sid, session in self._sessions.items() if session.is_expired
        ]

        for session_id in expired_ids:
            del self._sessions[session_id]

        if expired_ids:
            logger.info("Expired sessions cleaned up", count=len(expired_ids))

        return len(expired_ids)

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
        sessions = [
            session
            for session in self._sessions.values()
            if session.user_id == user_id
        ]

        if not include_expired:
            sessions = [s for s in sessions if not s.is_expired]

        logger.debug(
            "User sessions retrieved",
            user_id=user_id,
            count=len(sessions),
        )

        return sessions

    async def extend_session(
        self,
        session_id: str,
        additional_seconds: int | None = None,
    ) -> Session | None:
        """Extend a session's TTL.

        Args:
            session_id: Session identifier
            additional_seconds: Seconds to add, defaults to config TTL

        Returns:
            Updated Session or None if not found
        """
        session = await self.get_session(session_id)

        if session is None:
            return None

        extension = additional_seconds or self.config.default_ttl_seconds
        session.expires_at = datetime.utcnow() + timedelta(seconds=extension)
        session.updated_at = datetime.utcnow()

        logger.info(
            "Session extended",
            session_id=session_id,
            new_expires_at=session.expires_at.isoformat(),
        )

        return session

    def get_session_count(self) -> int:
        """Get the total number of active sessions."""
        return len(self._sessions)

    def get_stats(self) -> dict[str, Any]:
        """Get session manager statistics."""
        total_events = sum(s.event_count for s in self._sessions.values())
        expired_count = sum(1 for s in self._sessions.values() if s.is_expired)

        return {
            "total_sessions": len(self._sessions),
            "active_sessions": len(self._sessions) - expired_count,
            "expired_sessions": expired_count,
            "total_events": total_events,
            "config": {
                "default_ttl_seconds": self.config.default_ttl_seconds,
                "max_events_per_session": self.config.max_events_per_session,
            },
        }
