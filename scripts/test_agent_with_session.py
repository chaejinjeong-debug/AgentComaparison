#!/usr/bin/env python3
"""Test VertexAI Agent Engine with Sessions.

This script tests if the deployed agent can use session context for memory.

Usage:
    cd AgentEngine && uv run python scripts/test_agent_with_session.py
"""

import datetime
import os
import time
from pathlib import Path

import vertexai
from dotenv import load_dotenv
from vertexai.preview import reasoning_engines

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Configuration
PROJECT_ID = os.getenv("AGENT_PROJECT_ID", "")
LOCATION = os.getenv("AGENT_LOCATION", "asia-northeast3")
AGENT_ENGINE_ID = os.getenv("AGENT_ENGINE_ID", "")
USER_ID = "luke-test-user"


def main():
    print("=" * 60)
    print("VertexAI Agent Engine with Session Test")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print(f"Agent Engine ID: {AGENT_ENGINE_ID}")
    print(f"User ID: {USER_ID}")
    print("=" * 60 + "\n")

    # Initialize
    print("Initializing...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    client = vertexai.Client(project=PROJECT_ID, location=LOCATION)
    agent = reasoning_engines.ReasoningEngine(AGENT_ENGINE_ID)

    agent_engine_name = (
        f"projects/{PROJECT_ID}/locations/{LOCATION}"
        f"/reasoningEngines/{AGENT_ENGINE_ID}"
    )

    # Step 1: Create a session
    print("-" * 40)
    print("Step 1: Creating a session")
    print("-" * 40)
    expire_time = datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(hours=25)
    result = client.agent_engines.sessions.create(
        name=agent_engine_name,
        user_id=USER_ID,
        config={"expire_time": expire_time},
    )

    session_name = result.name
    if "/operations/" in session_name:
        session_name = session_name.split("/operations/")[0]

    session_id = session_name.split("/sessions/")[-1]
    print(f"Session created: {session_id}\n")
    time.sleep(1)

    # Step 2: First message - introduce myself
    print("-" * 40)
    print("Step 2: First message (with session)")
    print("-" * 40)
    message1 = "안녕! 나는 Luke야."
    print(f"You: {message1}")

    # Append user event to session
    client.agent_engines.sessions.events.append(
        name=session_name,
        author="user",
        invocation_id="1",
        timestamp=datetime.datetime.now(tz=datetime.UTC),
        config={"content": {"role": "user", "parts": [{"text": message1}]}},
    )

    # Query agent with session_id
    response1 = agent.query(message=message1, user_id=USER_ID, session_id=session_id)
    answer1 = response1.get("response", str(response1))
    print(f"Agent: {answer1}")

    # Append agent response to session
    client.agent_engines.sessions.events.append(
        name=session_name,
        author="agent",
        invocation_id="1",
        timestamp=datetime.datetime.now(tz=datetime.UTC),
        config={"content": {"role": "model", "parts": [{"text": answer1}]}},
    )
    print()

    # Step 3: Second message - ask who I am
    print("-" * 40)
    print("Step 3: Second message (with same session)")
    print("-" * 40)
    message2 = "내가 누구라고?"
    print(f"You: {message2}")

    # Append user event to session
    client.agent_engines.sessions.events.append(
        name=session_name,
        author="user",
        invocation_id="2",
        timestamp=datetime.datetime.now(tz=datetime.UTC),
        config={"content": {"role": "user", "parts": [{"text": message2}]}},
    )

    # Query agent with session_id
    response2 = agent.query(message=message2, user_id=USER_ID, session_id=session_id)
    answer2 = response2.get("response", str(response2))
    print(f"Agent: {answer2}\n")

    # Step 4: Check session events
    print("-" * 40)
    print("Step 4: Session events")
    print("-" * 40)
    events = list(client.agent_engines.sessions.events.list(name=session_name))
    print(f"Total events in session: {len(events)}")
    for i, event in enumerate(events):
        author = getattr(event, "author", "?")
        content = getattr(event, "content", {})
        if hasattr(content, "parts"):
            parts = content.parts
            text = parts[0].text if parts else ""
        elif isinstance(content, dict):
            parts = content.get("parts", [])
            text = parts[0].get("text", "") if parts else ""
        else:
            text = str(content)
        print(f"  {i+1}. [{author}] {text[:60]}...")
    print()

    # Result
    print("=" * 60)
    print("Test Result")
    print("=" * 60)
    if "luke" in answer2.lower() or "Luke" in answer2:
        print("SUCCESS: Agent remembered 'Luke'!")
    else:
        print("NOTICE: Agent may not have used session context.")
        print("\nThis could mean:")
        print("  1. Deployed agent doesn't read session history")
        print("  2. Need to update agent to use query_with_session")
    print("=" * 60)

    # Cleanup
    print("\nCleaning up session...")
    client.agent_engines.sessions.delete(name=session_name)
    print("Done!")


if __name__ == "__main__":
    main()
