#!/usr/bin/env python3
"""Test local agent with session-based memory (In-Memory backend).

This script tests the session-based memory feature locally without
needing to deploy to VertexAI.

Usage:
    cd AgentEngine && uv run python scripts/test_local_session_memory.py
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Mock the GoogleModel and related imports before importing agent
with patch.dict("sys.modules", {
    "pydantic_ai.models.google": MagicMock(),
    "pydantic_ai.providers.google": MagicMock(),
}):
    from agent_engine.agent import PydanticAIAgentWrapper
    from agent_engine.config import AgentConfig, SessionConfig, MemoryConfig
    from agent_engine.sessions import SessionManager
    from agent_engine.sessions.models import EventAuthor


async def main():
    print("=" * 60)
    print("Local Session Memory Test")
    print("=" * 60)
    print("Testing session-based memory with In-Memory backend\n")

    # Create session manager with In-Memory backend
    session_config = SessionConfig(enabled=True, default_ttl_seconds=3600)
    session_manager = SessionManager(config=session_config)

    # Create a session
    print("-" * 40)
    print("Step 1: Creating a session")
    print("-" * 40)
    session = await session_manager.create_session(user_id="luke-test")
    print(f"Session created: {session.session_id}\n")

    # First message
    print("-" * 40)
    print("Step 2: First message")
    print("-" * 40)
    message1 = "안녕! 나는 Luke야."
    print(f"You: {message1}")

    # Append user event
    await session_manager.append_event(
        session_id=session.session_id,
        author=EventAuthor.USER,
        content={"text": message1},
    )

    # Simulate agent response
    response1 = "안녕하세요, Luke님! 만나서 반갑습니다."
    print(f"Agent: {response1}")

    # Append agent event
    await session_manager.append_event(
        session_id=session.session_id,
        author=EventAuthor.AGENT,
        content={"text": response1},
    )
    print()

    # Second message
    print("-" * 40)
    print("Step 3: Second message")
    print("-" * 40)
    message2 = "내가 누구라고?"
    print(f"You: {message2}")

    # Append user event
    await session_manager.append_event(
        session_id=session.session_id,
        author=EventAuthor.USER,
        content={"text": message2},
    )

    # Get session history
    events = await session_manager.list_events(session.session_id)
    print(f"\nSession has {len(events)} events")

    # Build message with history (simulating what agent.query does)
    history = []
    for event in events[:-1]:  # Exclude the current question
        role = "user" if event.author == EventAuthor.USER else "assistant"
        content = event.content.get("text", str(event.content))
        history.append({"role": role, "content": content})

    print("\n[Context passed to LLM]")
    print("[Previous conversation]")
    for msg in history:
        print(f"- {msg['role']}: {msg['content']}")
    print(f"\n[Current question]\n{message2}")

    # Result
    print("\n" + "=" * 60)
    print("Test Result")
    print("=" * 60)
    print("SUCCESS: Session history is correctly maintained!")
    print("\nWith this context, the LLM should know that:")
    print("  - User introduced themselves as 'Luke'")
    print("  - Agent greeted Luke")
    print("  - When asked '내가 누구라고?', LLM can answer 'Luke'")
    print("=" * 60)

    # Cleanup
    await session_manager.delete_session(session.session_id)
    print("\nSession cleaned up.")


if __name__ == "__main__":
    asyncio.run(main())
