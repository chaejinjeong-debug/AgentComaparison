#!/usr/bin/env python3
"""Test script for VertexAI Session/Memory backends.

This script tests the VertexAI backend integration for Sessions and Memory Bank.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


from agent_engine.config import (
    MemoryBackendType,
    MemoryConfig,
    SessionBackendType,
    SessionConfig,
)
from agent_engine.memory import MemoryManager
from agent_engine.sessions import SessionManager
from agent_engine.sessions.models import EventAuthor

# Configuration
PROJECT_ID = os.getenv("AGENT_PROJECT_ID", "")
LOCATION = os.getenv("AGENT_LOCATION", "asia-northeast3")
AGENT_ENGINE_ID = os.getenv("AGENT_ENGINE_ID", "")


async def test_inmemory_backend():
    """Test In-Memory backend (default)."""
    print("\n" + "=" * 60)
    print("Testing In-Memory Backend")
    print("=" * 60)

    # Session Manager with In-Memory backend
    session_config = SessionConfig(
        backend=SessionBackendType.IN_MEMORY,
        default_ttl_seconds=3600,
    )
    session_manager = SessionManager(config=session_config)

    print(f"\nSession Backend: {session_manager.backend_type.value}")

    # Create session
    session = await session_manager.create_session(user_id="test-user-001")
    print(f"Created session: {session.session_id}")

    # Append events
    await session_manager.append_event(
        session_id=session.session_id,
        author=EventAuthor.USER,
        content={"text": "Hello, my name is Luke!"},
    )
    await session_manager.append_event(
        session_id=session.session_id,
        author=EventAuthor.AGENT,
        content={"text": "Hello Luke! How can I help you today?"},
    )

    # List events
    events = await session_manager.list_events(session.session_id)
    print(f"Session has {len(events)} events")

    # Stats
    stats = session_manager.get_stats()
    print(f"Stats: {stats}")

    print("\nIn-Memory Backend Test: PASSED")
    return True


async def test_vertex_session_backend():
    """Test VertexAI Session backend."""
    print("\n" + "=" * 60)
    print("Testing VertexAI Session Backend")
    print("=" * 60)

    try:
        # Session Manager with VertexAI backend
        session_config = SessionConfig(
            backend=SessionBackendType.VERTEX_AI,
            agent_engine_id=AGENT_ENGINE_ID,
            default_ttl_seconds=3600,
        )
        session_manager = SessionManager(
            config=session_config,
            project_id=PROJECT_ID,
            location=LOCATION,
        )

        print(f"\nSession Backend: {session_manager.backend_type.value}")
        print(f"Agent Engine ID: {AGENT_ENGINE_ID}")

        # Create session
        print("\nCreating VertexAI session...")
        session = await session_manager.create_session(user_id="test-user-vertex")
        print(f"Created session: {session.session_id}")

        # Append events
        print("Appending events...")
        await session_manager.append_event(
            session_id=session.session_id,
            author=EventAuthor.USER,
            content={"text": "Hello from VertexAI backend test!"},
        )

        # List events
        events = await session_manager.list_events(session.session_id)
        print(f"Session has {len(events)} events")

        # Stats
        stats = session_manager.get_stats()
        print(f"Stats: {stats}")

        print("\nVertexAI Session Backend Test: PASSED")
        return True

    except Exception as e:
        print("\nVertexAI Session Backend Test: FAILED")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_vertex_memory_backend():
    """Test VertexAI Memory backend."""
    print("\n" + "=" * 60)
    print("Testing VertexAI Memory Backend")
    print("=" * 60)

    try:
        # Memory Manager with VertexAI backend
        memory_config = MemoryConfig(
            backend=MemoryBackendType.VERTEX_AI,
            agent_engine_id=AGENT_ENGINE_ID,
        )
        memory_manager = MemoryManager(
            config=memory_config,
            project_id=PROJECT_ID,
            location=LOCATION,
        )

        print(f"\nMemory Backend: {memory_manager.backend_type.value}")
        print(f"Agent Engine ID: {AGENT_ENGINE_ID}")

        # Save memory
        print("\nSaving memory to VertexAI...")
        memory = await memory_manager.save_memory(
            user_id="test-user-vertex",
            fact="User's name is Luke and they are testing VertexAI Memory Bank",
            topics=["name", "testing"],
        )
        print(f"Saved memory: {memory.memory_id}")

        # Retrieve memories
        print("Retrieving memories...")
        memories = await memory_manager.retrieve_memories(
            user_id="test-user-vertex",
            query="What is the user's name?",
            max_results=5,
        )
        print(f"Retrieved {len(memories)} memories")

        # Stats
        stats = memory_manager.get_stats()
        print(f"Stats: {stats}")

        print("\nVertexAI Memory Backend Test: PASSED")
        return True

    except Exception as e:
        print("\nVertexAI Memory Backend Test: FAILED")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all backend tests."""
    print("=" * 60)
    print("VertexAI Backend Integration Tests")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print(f"Agent Engine: {AGENT_ENGINE_ID}")

    results = []

    # Test 1: In-Memory backend
    results.append(("In-Memory", await test_inmemory_backend()))

    # Test 2: VertexAI Session backend
    results.append(("VertexAI Session", await test_vertex_session_backend()))

    # Test 3: VertexAI Memory backend
    results.append(("VertexAI Memory", await test_vertex_memory_backend()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
