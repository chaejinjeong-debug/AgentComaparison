"""Tests for Session Management module (SM-001~SM-005)."""

import asyncio
from datetime import datetime, timedelta

import pytest

from agent_engine.config import SessionConfig
from agent_engine.sessions import EventAuthor, Session, SessionEvent, SessionManager


class TestSessionModels:
    """Tests for session models."""

    def test_session_event_creation(self):
        """Test SessionEvent creation with defaults."""
        event = SessionEvent(
            author=EventAuthor.USER,
            content={"text": "Hello"},
        )

        assert event.event_id is not None
        assert event.author == EventAuthor.USER
        assert event.content == {"text": "Hello"}
        assert event.timestamp is not None
        assert event.metadata == {}

    def test_session_event_to_dict(self):
        """Test SessionEvent serialization."""
        event = SessionEvent(
            author=EventAuthor.AGENT,
            content={"text": "Hi there!"},
            metadata={"model": "gemini"},
        )

        data = event.to_dict()

        assert data["author"] == "agent"
        assert data["content"]["text"] == "Hi there!"
        assert data["metadata"]["model"] == "gemini"
        assert "timestamp" in data

    def test_session_creation(self):
        """Test Session creation."""
        expires_at = datetime.utcnow() + timedelta(hours=1)
        session = Session(
            user_id="user-123",
            expires_at=expires_at,
        )

        assert session.session_id is not None
        assert session.user_id == "user-123"
        assert session.events == []
        assert session.event_count == 0
        assert not session.is_expired

    def test_session_is_expired(self):
        """Test Session expiration check."""
        # Expired session
        expired_session = Session(
            user_id="user-123",
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        assert expired_session.is_expired

        # Valid session
        valid_session = Session(
            user_id="user-123",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        assert not valid_session.is_expired


class TestSessionManager:
    """Tests for SessionManager (SM-001~SM-005)."""

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager: SessionManager):
        """SM-001: Test session creation."""
        session = await session_manager.create_session(user_id="user-123")

        assert session is not None
        assert session.user_id == "user-123"
        assert session.session_id is not None
        assert not session.is_expired
        assert session_manager.get_session_count() == 1

    @pytest.mark.asyncio
    async def test_create_session_with_custom_ttl(self, session_manager: SessionManager):
        """SM-001: Test session creation with custom TTL."""
        session = await session_manager.create_session(
            user_id="user-123",
            ttl_seconds=60,  # 1 minute
        )

        assert session is not None
        # Check that expires_at is approximately 1 minute from now
        expected_expiry = datetime.utcnow() + timedelta(seconds=60)
        assert abs((session.expires_at - expected_expiry).total_seconds()) < 5

    @pytest.mark.asyncio
    async def test_append_event(self, session_manager: SessionManager):
        """SM-002: Test event recording."""
        session = await session_manager.create_session(user_id="user-123")

        event = await session_manager.append_event(
            session_id=session.session_id,
            author=EventAuthor.USER,
            content={"text": "Hello!"},
        )

        assert event is not None
        assert event.author == EventAuthor.USER
        assert event.content["text"] == "Hello!"

        # Verify event was added to session
        updated_session = await session_manager.get_session(session.session_id)
        assert updated_session is not None
        assert updated_session.event_count == 1

    @pytest.mark.asyncio
    async def test_append_event_to_nonexistent_session(self, session_manager: SessionManager):
        """SM-002: Test appending to non-existent session."""
        event = await session_manager.append_event(
            session_id="nonexistent-session",
            author=EventAuthor.USER,
            content={"text": "Hello!"},
        )

        assert event is None

    @pytest.mark.asyncio
    async def test_list_events(self, session_manager: SessionManager):
        """SM-003: Test event history retrieval."""
        session = await session_manager.create_session(user_id="user-123")

        # Add multiple events
        for i in range(5):
            await session_manager.append_event(
                session_id=session.session_id,
                author=EventAuthor.USER if i % 2 == 0 else EventAuthor.AGENT,
                content={"text": f"Message {i}"},
            )

        # List all events
        events = await session_manager.list_events(session.session_id)
        assert len(events) == 5

        # List with limit
        limited_events = await session_manager.list_events(
            session.session_id,
            limit=3,
        )
        assert len(limited_events) == 3

        # List with offset
        offset_events = await session_manager.list_events(
            session.session_id,
            offset=2,
        )
        assert len(offset_events) == 3

    @pytest.mark.asyncio
    async def test_session_ttl_expiration(self, session_config: SessionConfig):
        """SM-004: Test session TTL management."""
        # Create manager with minimum allowed TTL (60 seconds)
        # Test by manipulating session expiry directly
        manager = SessionManager(config=session_config)

        session = await manager.create_session(user_id="user-123")
        session_id = session.session_id

        # Session should be valid initially
        assert await manager.get_session(session_id) is not None

        # Manually expire the session for testing
        from datetime import datetime, timedelta
        session.expires_at = datetime.utcnow() - timedelta(seconds=1)

        # Session should be expired now
        assert await manager.get_session(session_id) is None

    @pytest.mark.asyncio
    async def test_delete_session(self, session_manager: SessionManager):
        """SM-005: Test session deletion."""
        session = await session_manager.create_session(user_id="user-123")
        session_id = session.session_id

        # Verify session exists
        assert await session_manager.get_session(session_id) is not None

        # Delete session
        result = await session_manager.delete_session(session_id)
        assert result is True

        # Verify session is gone
        assert await session_manager.get_session(session_id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, session_manager: SessionManager):
        """SM-005: Test deleting non-existent session."""
        result = await session_manager.delete_session("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, session_config: SessionConfig):
        """SM-004: Test cleanup of expired sessions."""
        manager = SessionManager(config=session_config)

        # Create multiple sessions
        s1 = await manager.create_session(user_id="user-1")
        s2 = await manager.create_session(user_id="user-2")
        s3 = await manager.create_session(user_id="user-3")

        assert manager.get_session_count() == 3

        # Manually expire the sessions for testing
        from datetime import datetime, timedelta
        for session in [s1, s2, s3]:
            session.expires_at = datetime.utcnow() - timedelta(seconds=1)

        # Cleanup expired sessions
        removed = await manager.cleanup_expired_sessions()
        assert removed == 3
        assert manager.get_session_count() == 0

    @pytest.mark.asyncio
    async def test_get_user_sessions(self, session_manager: SessionManager):
        """Test retrieving all sessions for a user."""
        # Create sessions for multiple users
        await session_manager.create_session(user_id="user-1")
        await session_manager.create_session(user_id="user-1")
        await session_manager.create_session(user_id="user-2")

        # Get user-1's sessions
        user1_sessions = await session_manager.get_user_sessions("user-1")
        assert len(user1_sessions) == 2

        # Get user-2's sessions
        user2_sessions = await session_manager.get_user_sessions("user-2")
        assert len(user2_sessions) == 1

    @pytest.mark.asyncio
    async def test_extend_session(self, session_manager: SessionManager):
        """Test session TTL extension."""
        session = await session_manager.create_session(user_id="user-123")
        original_expiry = session.expires_at

        # Extend the session
        extended_session = await session_manager.extend_session(session.session_id)

        assert extended_session is not None
        assert extended_session.expires_at > original_expiry

    @pytest.mark.asyncio
    async def test_max_events_limit(self, session_config: SessionConfig):
        """Test max events per session limit."""
        limited_config = SessionConfig(
            enabled=True,
            default_ttl_seconds=3600,
            max_events_per_session=10,  # Minimum allowed is 10
        )
        manager = SessionManager(config=limited_config)

        session = await manager.create_session(user_id="user-123")

        # Add more events than the limit
        for i in range(15):
            await manager.append_event(
                session_id=session.session_id,
                author=EventAuthor.USER,
                content={"text": f"Message {i}"},
            )

        # Should only have max_events_per_session events
        events = await manager.list_events(session.session_id)
        assert len(events) == 10

        # First event should be Message 5 (oldest ones removed)
        assert events[0].content["text"] == "Message 5"

    @pytest.mark.asyncio
    async def test_get_stats(self, session_manager: SessionManager):
        """Test session manager statistics."""
        # Create some sessions and events
        session = await session_manager.create_session(user_id="user-123")
        await session_manager.append_event(
            session_id=session.session_id,
            author=EventAuthor.USER,
            content={"text": "Hello"},
        )

        stats = session_manager.get_stats()

        assert stats["total_sessions"] == 1
        assert stats["active_sessions"] == 1
        assert stats["total_events"] == 1
        assert "config" in stats
