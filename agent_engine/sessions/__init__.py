"""Session management for the Agent Engine platform."""

from agent_engine.sessions.backends import (
    InMemorySessionBackend,
    SessionBackend,
    VertexAISessionBackend,
)
from agent_engine.sessions.manager import SessionManager
from agent_engine.sessions.models import EventAuthor, Session, SessionEvent

__all__ = [
    # Manager
    "SessionManager",
    # Models
    "Session",
    "SessionEvent",
    "EventAuthor",
    # Backends
    "SessionBackend",
    "InMemorySessionBackend",
    "VertexAISessionBackend",
]
