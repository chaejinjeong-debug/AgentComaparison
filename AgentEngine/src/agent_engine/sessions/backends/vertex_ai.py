"""VertexAI session storage backend using REST API for production use."""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import google.auth
import google.auth.transport.requests
import httpx
import structlog

from agent_engine.config import SessionConfig
from agent_engine.sessions.backends.base import SessionBackend
from agent_engine.sessions.models import EventAuthor, Session, SessionEvent

logger = structlog.get_logger(__name__)


class VertexAISessionBackend(SessionBackend):
    """VertexAI Agent Engine session storage backend using REST API.

    This backend uses VertexAI's managed Sessions REST API for:
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

        # REST API configuration
        self._base_url = f"https://{location}-aiplatform.googleapis.com/v1"
        self._reasoning_engine = (
            f"projects/{project_id}/locations/{location}"
            f"/reasoningEngines/{agent_engine_id}"
        )
        self._credentials = None
        self._token_expiry = None

        # Local cache for session metadata
        self._session_cache: dict[str, Session] = {}

        logger.info(
            "VertexAISessionBackend initialized",
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

    async def create_session(
        self,
        user_id: str,
        ttl_seconds: int,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session using VertexAI Sessions REST API."""
        url = f"{self._base_url}/{self._reasoning_engine}/sessions"

        request_body = {
            "userId": user_id,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=request_body,
                )

                if response.status_code in [200, 201]:
                    data = response.json()
                    # Extract session ID from operation name
                    # Format: projects/.../sessions/{session_id}/operations/{op_id}
                    name = data.get("name", "")
                    parts = name.split("/")
                    session_id = ""
                    for i, part in enumerate(parts):
                        if part == "sessions" and i + 1 < len(parts):
                            session_id = parts[i + 1]
                            break

                    if not session_id:
                        session_id = str(uuid4())

                    session = Session(
                        session_id=session_id,
                        user_id=user_id,
                        expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
                        metadata=metadata or {},
                    )

                    # Cache locally
                    self._session_cache[session_id] = session

                    logger.info(
                        "VertexAI session created",
                        session_id=session_id,
                        user_id=user_id,
                    )

                    return session
                else:
                    logger.error(
                        "Failed to create VertexAI session",
                        status=response.status_code,
                        error=response.text[:200],
                    )
                    raise RuntimeError(f"Failed to create session: {response.text[:200]}")

        except httpx.HTTPError as e:
            logger.error("HTTP error creating VertexAI session", error=str(e))
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

        url = f"{self._base_url}/{self._reasoning_engine}/sessions/{session_id}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self._get_headers())

                if response.status_code == 200:
                    data = response.json()
                    session = Session(
                        session_id=session_id,
                        user_id=data.get("userId", "unknown"),
                        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
                    )
                    self._session_cache[session_id] = session
                    return session
                elif response.status_code == 404:
                    return None
                else:
                    logger.error(
                        "Failed to get VertexAI session",
                        session_id=session_id,
                        status=response.status_code,
                    )
                    return None

        except httpx.HTTPError as e:
            logger.error("HTTP error getting VertexAI session", error=str(e))
            return None

    async def append_event(
        self,
        session_id: str,
        author: EventAuthor,
        content: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> SessionEvent | None:
        """Append an event to a VertexAI session."""
        url = f"{self._base_url}/{self._reasoning_engine}/sessions/{session_id}:appendEvent"

        # Prepare event content
        text_content = content.get("text", str(content))

        request_body = {
            "event": {
                "author": author.value,
                "invocationId": str(uuid4()),
                "content": {
                    "role": author.value,
                    "parts": [{"text": text_content}],
                },
            },
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=request_body,
                )

                if response.status_code in [200, 201]:
                    event = SessionEvent(
                        author=author,
                        content=content,
                        metadata=metadata or {},
                    )

                    # Update local cache
                    if session_id in self._session_cache:
                        self._session_cache[session_id].events.append(event)
                        self._session_cache[session_id].updated_at = datetime.utcnow()

                    logger.debug(
                        "Event appended to VertexAI session",
                        session_id=session_id,
                        event_id=event.event_id,
                    )
                    return event
                else:
                    logger.error(
                        "Failed to append event",
                        session_id=session_id,
                        status=response.status_code,
                        error=response.text[:200],
                    )
                    return None

        except httpx.HTTPError as e:
            logger.error("HTTP error appending event", error=str(e))
            return None

    async def list_events(
        self,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[SessionEvent]:
        """List events from a VertexAI session."""
        url = f"{self._base_url}/{self._reasoning_engine}/sessions/{session_id}:listEvents"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self._get_headers())

                if response.status_code == 200:
                    data = response.json()
                    events = []

                    for event_data in data.get("sessionEvents", []):
                        author_str = event_data.get("author", "user")
                        try:
                            author = EventAuthor(author_str)
                        except ValueError:
                            author = EventAuthor.USER

                        content_data = event_data.get("content", {})
                        parts = content_data.get("parts", [])
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
                        "Events listed from VertexAI session",
                        session_id=session_id,
                        count=len(events),
                    )
                    return events
                else:
                    logger.error(
                        "Failed to list events",
                        session_id=session_id,
                        status=response.status_code,
                    )
                    return []

        except httpx.HTTPError as e:
            logger.error("HTTP error listing events", error=str(e))
            return []

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from VertexAI."""
        url = f"{self._base_url}/{self._reasoning_engine}/sessions/{session_id}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(url, headers=self._get_headers())

                if response.status_code in [200, 204]:
                    self._session_cache.pop(session_id, None)
                    logger.info("VertexAI session deleted", session_id=session_id)
                    return True
                else:
                    logger.error(
                        "Failed to delete session",
                        session_id=session_id,
                        status=response.status_code,
                    )
                    return False

        except httpx.HTTPError as e:
            logger.error("HTTP error deleting session", error=str(e))
            return False

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.

        Note: VertexAI automatically handles session expiration.
        This cleans up the local cache only.
        """
        expired_ids = [
            sid for sid, session in self._session_cache.items()
            if session.is_expired
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
        url = f"{self._base_url}/{self._reasoning_engine}/sessions"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self._get_headers())

                if response.status_code == 200:
                    data = response.json()
                    sessions = []

                    for session_data in data.get("sessions", []):
                        if session_data.get("userId") == user_id:
                            name = session_data.get("name", "")
                            session_id = name.split("/")[-1] if "/" in name else name

                            session = Session(
                                session_id=session_id,
                                user_id=user_id,
                                expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
                            )

                            if include_expired or not session.is_expired:
                                sessions.append(session)

                    logger.debug(
                        "User sessions retrieved from VertexAI",
                        user_id=user_id,
                        count=len(sessions),
                    )
                    return sessions
                else:
                    logger.error(
                        "Failed to get user sessions",
                        user_id=user_id,
                        status=response.status_code,
                    )
                    return []

        except httpx.HTTPError as e:
            logger.error("HTTP error getting user sessions", error=str(e))
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
            session.expires_at = datetime.now(timezone.utc) + timedelta(seconds=additional_seconds)
            session.updated_at = datetime.utcnow()
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
            "backend": "vertex_ai",
            "project_id": self.project_id,
            "location": self.location,
            "agent_engine_id": self.agent_engine_id,
            "cached_sessions": len(self._session_cache),
            "config": {
                "default_ttl_seconds": self.config.default_ttl_seconds,
                "max_events_per_session": self.config.max_events_per_session,
            },
        }
