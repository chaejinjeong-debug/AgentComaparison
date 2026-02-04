#!/usr/bin/env python3
"""Test VertexAI Sessions API directly.

This script tests the Sessions SDK to verify it's working correctly.

Usage:
    cd AgentEngine && uv run python scripts/test_sessions_api.py
"""

import datetime
import os
import time
from pathlib import Path

import vertexai
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Configuration
PROJECT_ID = os.getenv("AGENT_PROJECT_ID", "")
LOCATION = os.getenv("AGENT_LOCATION", "asia-northeast3")
AGENT_ENGINE_ID = os.getenv("AGENT_ENGINE_ID", "")


def main():
    print("=" * 60)
    print("VertexAI Sessions API Test")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print(f"Agent Engine ID: {AGENT_ENGINE_ID}")
    print("=" * 60 + "\n")

    # Initialize VertexAI SDK client
    print("Creating VertexAI SDK client...")
    client = vertexai.Client(project=PROJECT_ID, location=LOCATION)
    agent_engine_name = (
        f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{AGENT_ENGINE_ID}"
    )
    print(f"Agent Engine Name: {agent_engine_name}\n")

    # Test 1: Create a session
    print("-" * 40)
    print("Test 1: Creating a session")
    print("-" * 40)
    try:
        expire_time = datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(hours=25)
        result = client.agent_engines.sessions.create(
            name=agent_engine_name,
            user_id="luke-test-user",
            config={"expire_time": expire_time},
        )
        print(f"Create result: {result}")
        print(f"Result name: {result.name}")

        # Extract actual session name (remove /operations/... if present)
        session_name = result.name
        if "/operations/" in session_name:
            session_name = session_name.split("/operations/")[0]

        session_id = session_name.split("/sessions/")[-1]
        print(f"Session Name: {session_name}")
        print(f"Session ID: {session_id}\n")

        # Wait a bit for session to be ready
        print("Waiting for session to be ready...")
        time.sleep(2)

    except Exception as e:
        print(f"Failed to create session: {e}\n")
        return

    # Test 2: Append user event
    print("-" * 40)
    print("Test 2: Appending user event")
    print("-" * 40)
    try:
        client.agent_engines.sessions.events.append(
            name=session_name,
            author="user",
            invocation_id="1",
            timestamp=datetime.datetime.now(tz=datetime.UTC),
            config={
                "content": {
                    "role": "user",
                    "parts": [{"text": "안녕! 나는 Luke야."}],
                }
            },
        )
        print("User event appended: '안녕! 나는 Luke야.'\n")
    except Exception as e:
        print(f"Failed to append event: {e}\n")

    # Test 3: Append agent event
    print("-" * 40)
    print("Test 3: Appending agent event")
    print("-" * 40)
    try:
        client.agent_engines.sessions.events.append(
            name=session_name,
            author="agent",
            invocation_id="1",
            timestamp=datetime.datetime.now(tz=datetime.UTC),
            config={
                "content": {
                    "role": "model",
                    "parts": [{"text": "안녕하세요, Luke님! 만나서 반갑습니다."}],
                }
            },
        )
        print("Agent event appended: '안녕하세요, Luke님! 만나서 반갑습니다.'\n")
    except Exception as e:
        print(f"Failed to append event: {e}\n")

    # Test 4: List events using new API
    print("-" * 40)
    print("Test 4: Listing session events")
    print("-" * 40)
    try:
        events = list(client.agent_engines.sessions.events.list(name=session_name))
        print(f"Found {len(events)} events:")
        for i, event in enumerate(events):
            author = getattr(event, "author", "unknown")
            content = getattr(event, "content", {})
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text = parts[0].get("text", "") if parts else str(content)
            print(f"  {i + 1}. [{author}] {text[:50]}...")
        print()
    except Exception as e:
        print(f"Failed to list events: {e}\n")

    # Test 5: Get session
    print("-" * 40)
    print("Test 5: Getting session")
    print("-" * 40)
    try:
        retrieved_session = client.agent_engines.sessions.get(name=session_name)
        print(f"Session retrieved: {retrieved_session.name}")
        print(f"User ID: {getattr(retrieved_session, 'user_id', 'N/A')}\n")
    except Exception as e:
        print(f"Failed to get session: {e}\n")

    # Test 6: List all sessions for this user
    print("-" * 40)
    print("Test 6: Listing all sessions")
    print("-" * 40)
    try:
        sessions = list(client.agent_engines.sessions.list(name=agent_engine_name))
        print(f"Found {len(sessions)} session(s):")
        for s in sessions[:5]:  # Show first 5
            print(f"  - {s.name}")
        print()
    except Exception as e:
        print(f"Failed to list sessions: {e}\n")

    # Test 7: Delete session
    print("-" * 40)
    print("Test 7: Deleting session")
    print("-" * 40)
    try:
        client.agent_engines.sessions.delete(name=session_name)
        print("Session deleted successfully!\n")
    except Exception as e:
        print(f"Failed to delete session: {e}\n")

    print("=" * 60)
    print("Sessions API Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
