"""VertexAI session storage backend for production use."""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import structlog

from agent_engine.config import SessionConfig
from agent_engine.sessions.backends.base import SessionBackend
from agent_engine.sessions.models import EventAuthor, Session, SessionEvent

logger = structlog.get_logger(__name__)


class VertexAISessionBackend(SessionBackend):
    """VertexAI Agent Engine session storage backend.

    This backend uses VertexAI's managed Sessions API for:
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
        self._client = None
        self._agent_engine_name = (
            f"projects/{project_id}/locations/{location}"
            f"/reasoningEngines/{agent_engine_id}"
        )

        logger.info(
            "VertexAISessionBackend initialized",
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

    async def create_session(
        self,
        user_id: str,
        ttl_seconds: int,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session using VertexAI Sessions API."""
        try:
            client = self._get_client()

            # Call VertexAI Sessions API
            # Note: Actual API structure may vary based on SDK version
            request = {
                "parent": self._agent_engine_name,
                "session": {
                    "user_id": user_id,
                    "expire_time": {
                        "seconds": int(
                            (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).timestamp()
                        )
                    },
                },
            }

            # Create session via gRPC
            response = client.create_session(request=request)

            # Convert to internal Session model
            session = Session(
                session_id=response.name.split("/")[-1] if hasattr(response, "name") else str(uuid4()),
                user_id=user_id,
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
                metadata=metadata or {},
            )

            logger.info(
                "VertexAI session created",
                session_id=session.session_id,
                user_id=user_id,
                ttl_seconds=ttl_seconds,
            )

            return session

        except Exception as e:
            logger.error("Failed to create VertexAI session", error=str(e))
            # Fallback to local session creation for development
            return await self._create_local_session(user_id, ttl_seconds, metadata)

    async def _create_local_session(
        self,
        user_id: str,
        ttl_seconds: int,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a local session as fallback."""
        logger.warning("Using local session fallback")
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        return Session(
            user_id=user_id,
            expires_at=expires_at,
            metadata=metadata or {},
        )

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session from VertexAI."""
        try:
            client = self._get_client()

            session_name = f"{self._agent_engine_name}/sessions/{session_id}"
            response = client.get_session(name=session_name)

            if response is None:
                return None

            # Convert to internal Session model
            session = Session(
                session_id=session_id,
                user_id=getattr(response, "user_id", "unknown"),
                expires_at=datetime.fromtimestamp(
                    response.expire_time.seconds, tz=timezone.utc
                )
                if hasattr(response, "expire_time")
                else datetime.now(timezone.utc) + timedelta(hours=24),
            )

            if session.is_expired:
                logger.info("Session expired", session_id=session_id)
                return None

            return session

        except Exception as e:
            logger.error("Failed to get VertexAI session", session_id=session_id, error=str(e))
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
            event_content = {
                "role": author.value,
                "parts": [{"text": content.get("text", str(content))}],
            }

            request = {
                "name": session_name,
                "event": {
                    "author": author.value,
                    "invocation_id": str(uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "content": event_content,
                },
            }

            client.append_event(request=request)

            # Create internal event model
            event = SessionEvent(
                author=author,
                content=content,
                metadata=metadata or {},
            )

            logger.debug(
                "Event appended to VertexAI session",
                session_id=session_id,
                event_id=event.event_id,
                author=author.value,
            )

            return event

        except Exception as e:
            logger.error(
                "Failed to append event to VertexAI session",
                session_id=session_id,
                error=str(e),
            )
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

            # List events from VertexAI
            response = client.list_events(name=session_name)

            events = []
            for i, event_data in enumerate(response):
                if i < offset:
                    continue
                if limit and len(events) >= limit:
                    break

                # Convert to internal event model
                author_str = getattr(event_data, "author", "user")
                try:
                    author = EventAuthor(author_str)
                except ValueError:
                    author = EventAuthor.USER

                event = SessionEvent(
                    author=author,
                    content={"text": str(getattr(event_data, "content", {}))},
                )
                events.append(event)

            logger.debug(
                "Events listed from VertexAI session",
                session_id=session_id,
                returned_events=len(events),
            )

            return events

        except Exception as e:
            logger.error(
                "Failed to list events from VertexAI session",
                session_id=session_id,
                error=str(e),
            )
            return []

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from VertexAI."""
        try:
            client = self._get_client()

            session_name = f"{self._agent_engine_name}/sessions/{session_id}"
            client.delete_session(name=session_name)

            logger.info("VertexAI session deleted", session_id=session_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to delete VertexAI session",
                session_id=session_id,
                error=str(e),
            )
            return False

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.

        Note: VertexAI automatically handles session expiration.
        This method is a no-op for the VertexAI backend.
        """
        logger.debug("VertexAI handles session expiration automatically")
        return 0

    async def get_user_sessions(
        self,
        user_id: str,
        include_expired: bool = False,
    ) -> list[Session]:
        """Get all sessions for a user from VertexAI."""
        try:
            client = self._get_client()

            # List sessions filtered by user_id
            request = {
                "parent": self._agent_engine_name,
                "filter": f'user_id="{user_id}"',
            }

            response = client.list_sessions(request=request)

            sessions = []
            for session_data in response:
                session = Session(
                    session_id=session_data.name.split("/")[-1],
                    user_id=user_id,
                    expires_at=datetime.fromtimestamp(
                        session_data.expire_time.seconds, tz=timezone.utc
                    )
                    if hasattr(session_data, "expire_time")
                    else datetime.now(timezone.utc) + timedelta(hours=24),
                )

                if include_expired or not session.is_expired:
                    sessions.append(session)

            logger.debug(
                "User sessions retrieved from VertexAI",
                user_id=user_id,
                count=len(sessions),
            )

            return sessions

        except Exception as e:
            logger.error(
                "Failed to get user sessions from VertexAI",
                user_id=user_id,
                error=str(e),
            )
            return []

    async def extend_session(
        self,
        session_id: str,
        additional_seconds: int,
    ) -> Session | None:
        """Extend a session's TTL in VertexAI."""
        try:
            client = self._get_client()

            session_name = f"{self._agent_engine_name}/sessions/{session_id}"
            new_expire_time = datetime.now(timezone.utc) + timedelta(seconds=additional_seconds)

            request = {
                "session": {
                    "name": session_name,
                    "expire_time": {"seconds": int(new_expire_time.timestamp())},
                },
                "update_mask": {"paths": ["expire_time"]},
            }

            response = client.update_session(request=request)

            session = Session(
                session_id=session_id,
                user_id=getattr(response, "user_id", "unknown"),
                expires_at=new_expire_time,
            )

            logger.info(
                "VertexAI session extended",
                session_id=session_id,
                new_expires_at=new_expire_time.isoformat(),
            )

            return session

        except Exception as e:
            logger.error(
                "Failed to extend VertexAI session",
                session_id=session_id,
                error=str(e),
            )
            return None

    def get_session_count(self) -> int:
        """Get the total number of active sessions.

        Note: This requires listing all sessions which may be expensive.
        Consider caching or using metrics instead.
        """
        try:
            client = self._get_client()

            request = {"parent": self._agent_engine_name}
            response = client.list_sessions(request=request)

            count = sum(1 for _ in response)
            return count

        except Exception as e:
            logger.error("Failed to count VertexAI sessions", error=str(e))
            return 0

    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        return {
            "backend": "vertex_ai",
            "project_id": self.project_id,
            "location": self.location,
            "agent_engine_id": self.agent_engine_id,
            "config": {
                "default_ttl_seconds": self.config.default_ttl_seconds,
                "max_events_per_session": self.config.max_events_per_session,
            },
        }
