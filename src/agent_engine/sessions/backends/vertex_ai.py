"""VertexAI session storage backend using official SDK for production use."""

import datetime
from typing import Any
from uuid import uuid4

import structlog
import vertexai

from agent_engine.config import SessionConfig
from agent_engine.sessions.backends.base import SessionBackend
from agent_engine.sessions.models import EventAuthor, Session, SessionEvent

logger = structlog.get_logger(__name__)


class VertexAISessionBackend(SessionBackend):
    """VertexAI Agent Engine session storage backend using official SDK.

    This backend uses VertexAI's managed Sessions SDK for:
    - Persistent session storage
    - Automatic replication and backup
    - Multi-instance support
    - Production deployments

    Sessions survive server restarts and network disconnections.
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        agent_engine_id: str,
        config: SessionConfig | None = None,
    ) -> None:
        """Initialize the VertexAI backend.

        Args:
            project_id: GCP project ID
            location: GCP region
            agent_engine_id: Deployed Agent Engine ID
            config: Session configuration
        """
        self.project_id = project_id
        self.location = location
        self.agent_engine_id = agent_engine_id
        self.config = config or SessionConfig()

        # SDK client (lazy init)
        self._client: vertexai.Client | None = None
        self._agent_engine_name = (
            f"projects/{project_id}/locations/{location}"
            f"/reasoningEngines/{agent_engine_id}"
        )

        # Local cache for session metadata
        self._session_cache: dict[str, Session] = {}

        logger.info(
            "VertexAISessionBackend initialized (SDK)",
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

    async def create_session(
        self,
        user_id: str,
        ttl_seconds: int,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session using VertexAI Sessions SDK."""
        try:
            client = self._get_client()
            vertex_session = client.agent_engines.sessions.create(
                name=self._agent_engine_name,
                user_id=user_id,
                config={"expire_time": f"{ttl_seconds}s"},
            )

            # Extract session ID from the full resource name
            session_id = vertex_session.name.split("/")[-1]

            session = Session(
                session_id=session_id,
                user_id=user_id,
                expires_at=datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(seconds=ttl_seconds),
                metadata=metadata or {},
            )

            # Cache locally
            self._session_cache[session_id] = session

            logger.info(
                "VertexAI session created via SDK",
                session_id=session_id,
                user_id=user_id,
            )

            return session

        except Exception as e:
            logger.error("Failed to create VertexAI session via SDK", error=str(e))
            raise

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session from VertexAI."""
        # Check local cache first
        if session_id in self._session_cache:
            session = self._session_cache[session_id]
            if not session.is_expired:
                return session
            else:
                del self._session_cache[session_id]

        try:
            client = self._get_client()
            session_name = f"{self._agent_engine_name}/sessions/{session_id}"
            vertex_session = client.agent_engines.sessions.get(name=session_name)

            session = Session(
                session_id=session_id,
                user_id=getattr(vertex_session, "user_id", "unknown"),
                expires_at=datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(hours=24),
            )
            self._session_cache[session_id] = session
            return session

        except Exception as e:
            logger.debug("Session not found or error", session_id=session_id, error=str(e))
            return None

    async def append_event(
        self,
        session_id: str,
        author: EventAuthor,
        content: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> SessionEvent | None:
        """Append an event to a VertexAI session."""
        try:
            client = self._get_client()
            session_name = f"{self._agent_engine_name}/sessions/{session_id}"

            # Prepare event content
            text_content = content.get("text", str(content))
            invocation_id = str(uuid4())

            client.agent_engines.sessions.events.append(
                name=session_name,
                author=author.value,
                invocation_id=invocation_id,
                timestamp=datetime.datetime.now(tz=datetime.timezone.utc),
                config={
                    "content": {
                        "role": author.value,
                        "parts": [{"text": text_content}],
                    }
                },
            )

            event = SessionEvent(
                author=author,
                content=content,
                metadata=metadata or {},
            )

            # Update local cache
            if session_id in self._session_cache:
                self._session_cache[session_id].events.append(event)
                self._session_cache[session_id].updated_at = datetime.datetime.now(datetime.timezone.utc)

            logger.debug(
                "Event appended to VertexAI session via SDK",
                session_id=session_id,
                event_id=event.event_id,
            )
            return event

        except Exception as e:
            logger.error("Failed to append event via SDK", session_id=session_id, error=str(e))
            return None

    async def list_events(
        self,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[SessionEvent]:
        """List events from a VertexAI session."""
        try:
            client = self._get_client()
            session_name = f"{self._agent_engine_name}/sessions/{session_id}"

            events = []
            for event_data in client.agent_engines.list_session_events(name=session_name):
                author_str = getattr(event_data, "author", "user")
                try:
                    author = EventAuthor(author_str)
                except ValueError:
                    author = EventAuthor.USER

                # Extract text content from event
                content_data = getattr(event_data, "content", {})
                parts = content_data.get("parts", []) if isinstance(content_data, dict) else []
                text = parts[0].get("text", "") if parts else ""

                event = SessionEvent(
                    author=author,
                    content={"text": text},
                )
                events.append(event)

            # Apply offset and limit
            events = events[offset:]
            if limit:
                events = events[:limit]

            logger.debug(
                "Events listed from VertexAI session via SDK",
                session_id=session_id,
                count=len(events),
            )
            return events

        except Exception as e:
            logger.error("Failed to list events via SDK", session_id=session_id, error=str(e))
            return []

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from VertexAI."""
        try:
            client = self._get_client()
            session_name = f"{self._agent_engine_name}/sessions/{session_id}"

            client.agent_engines.sessions.delete(name=session_name)
            self._session_cache.pop(session_id, None)

            logger.info("VertexAI session deleted via SDK", session_id=session_id)
            return True

        except Exception as e:
            logger.error("Failed to delete session via SDK", session_id=session_id, error=str(e))
            return False

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.

        Note: VertexAI automatically handles session expiration.
        This cleans up the local cache only.
        """
        expired_ids = [
            sid for sid, session in self._session_cache.items() if session.is_expired
        ]

        for session_id in expired_ids:
            del self._session_cache[session_id]

        if expired_ids:
            logger.info("Expired sessions cleaned from cache", count=len(expired_ids))

        return len(expired_ids)

    async def get_user_sessions(
        self,
        user_id: str,
        include_expired: bool = False,
    ) -> list[Session]:
        """Get all sessions for a user from VertexAI."""
        try:
            client = self._get_client()
            sessions = []

            for vertex_session in client.agent_engines.sessions.list(
                name=self._agent_engine_name
            ):
                session_user_id = getattr(vertex_session, "user_id", None)
                if session_user_id == user_id:
                    session_id = vertex_session.name.split("/")[-1]

                    session = Session(
                        session_id=session_id,
                        user_id=user_id,
                        expires_at=datetime.datetime.now(datetime.timezone.utc)
                        + datetime.timedelta(hours=24),
                    )

                    if include_expired or not session.is_expired:
                        sessions.append(session)

            logger.debug(
                "User sessions retrieved from VertexAI via SDK",
                user_id=user_id,
                count=len(sessions),
            )
            return sessions

        except Exception as e:
            logger.error("Failed to get user sessions via SDK", user_id=user_id, error=str(e))
            return []

    async def extend_session(
        self,
        session_id: str,
        additional_seconds: int,
    ) -> Session | None:
        """Extend a session's TTL.

        Note: VertexAI may not support direct TTL extension.
        This updates the local cache only.
        """
        session = await self.get_session(session_id)
        if session:
            session.expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
                seconds=additional_seconds
            )
            session.updated_at = datetime.datetime.now(datetime.timezone.utc)
            self._session_cache[session_id] = session
            logger.info(
                "Session TTL extended in cache",
                session_id=session_id,
                expires_at=session.expires_at.isoformat(),
            )
            return session
        return None

    def get_session_count(self) -> int:
        """Get the total number of cached sessions."""
        return len(self._session_cache)

    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        return {
            "backend": "vertex_ai_sdk",
            "project_id": self.project_id,
            "location": self.location,
            "agent_engine_id": self.agent_engine_id,
            "cached_sessions": len(self._session_cache),
            "config": {
                "default_ttl_seconds": self.config.default_ttl_seconds,
                "max_events_per_session": self.config.max_events_per_session,
            },
        }
