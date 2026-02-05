"""Session manager for the Agent Engine platform.

Implements SM-001~SM-005:
- SM-001: Session creation
- SM-002: Event recording (AppendEvent)
- SM-003: History retrieval (ListEvents)
- SM-004: TTL management (24 hours default)
- SM-005: Session deletion

Uses a pluggable backend system for storage:
- InMemoryBackend: Local development/testing
- VertexAIBackend: Production with persistent storage
"""

from typing import Any

import structlog

from agent_engine.config import SessionBackendType, SessionConfig
from agent_engine.sessions.backends.base import SessionBackend
from agent_engine.sessions.backends.in_memory import InMemorySessionBackend
from agent_engine.sessions.backends.vertex_ai import VertexAISessionBackend
from agent_engine.sessions.models import EventAuthor, Session, SessionEvent

logger = structlog.get_logger(__name__)


class SessionManager:
    """Manages conversation sessions with pluggable backends.

    This manager handles session lifecycle including creation, event recording,
    history retrieval, and automatic expiration. It delegates storage to
    a backend implementation.

    Attributes:
        config: Session configuration
        _backend: Storage backend implementation
    """

    def __init__(
        self,
        config: SessionConfig | None = None,
        project_id: str | None = None,
        location: str | None = None,
    ) -> None:
        """Initialize the session manager.

        Args:
            config: Session configuration, uses defaults if not provided
            project_id: GCP project ID (required for VertexAI backend)
            location: GCP region (required for VertexAI backend)
        """
        self.config = config or SessionConfig()
        self._project_id = project_id
        self._location = location
        self._backend = self._create_backend()

        logger.info(
            "SessionManager initialized",
            backend=self.config.backend.value,
            ttl_seconds=self.config.default_ttl_seconds,
            max_events=self.config.max_events_per_session,
        )

    def _create_backend(self) -> SessionBackend:
        """Create the appropriate backend based on configuration."""
        if self.config.backend == SessionBackendType.VERTEX_AI:
            if not self.config.agent_engine_id:
                raise ValueError(
                    "agent_engine_id is required for VertexAI backend. "
                    "Set SESSION_AGENT_ENGINE_ID environment variable."
                )
            if not self._project_id or not self._location:
                raise ValueError("project_id and location are required for VertexAI backend.")

            return VertexAISessionBackend(
                project_id=self._project_id,
                location=self._location,
                agent_engine_id=self.config.agent_engine_id,
                config=self.config,
            )

        # Default to in-memory backend
        return InMemorySessionBackend(config=self.config)

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
        return await self._backend.create_session(
            user_id=user_id,
            ttl_seconds=ttl,
            metadata=metadata,
        )

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session if found and not expired, None otherwise
        """
        return await self._backend.get_session(session_id)

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
        return await self._backend.append_event(
            session_id=session_id,
            author=author,
            content=content,
            metadata=metadata,
        )

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
        return await self._backend.list_events(
            session_id=session_id,
            limit=limit,
            offset=offset,
        )

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        SM-005: Session deletion functionality.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        return await self._backend.delete_session(session_id)

    async def cleanup_expired_sessions(self) -> int:
        """Remove all expired sessions.

        SM-004: TTL management support.

        Returns:
            Number of sessions removed
        """
        return await self._backend.cleanup_expired_sessions()

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
        return await self._backend.get_user_sessions(
            user_id=user_id,
            include_expired=include_expired,
        )

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
        extension = additional_seconds or self.config.default_ttl_seconds
        return await self._backend.extend_session(
            session_id=session_id,
            additional_seconds=extension,
        )

    def get_session_count(self) -> int:
        """Get the total number of active sessions."""
        return self._backend.get_session_count()

    def get_stats(self) -> dict[str, Any]:
        """Get session manager statistics."""
        return self._backend.get_stats()

    @property
    def backend_type(self) -> SessionBackendType:
        """Get the current backend type."""
        return self.config.backend
