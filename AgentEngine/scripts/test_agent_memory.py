#!/usr/bin/env python3
"""Test VertexAI Agent Engine memory/session capability.

This script tests if the deployed agent can remember context across messages.

Usage:
    cd AgentEngine && uv run python scripts/test_agent_memory.py
"""

import vertexai
from vertexai.preview import reasoning_engines

# Configuration
PROJECT_ID = "heum-alfred-evidence-clf-dev"
LOCATION = "asia-northeast3"
AGENT_ENGINE_ID = "6406440838678708224"
USER_ID = "luke-test-user"


def main():
    print("=" * 60)
    print("VertexAI Agent Engine Memory Test")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print(f"Agent Engine ID: {AGENT_ENGINE_ID}")
    print(f"User ID: {USER_ID}")
    print("=" * 60 + "\n")

    # Initialize VertexAI
    print("Connecting to VertexAI Agent Engine...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Get deployed agent
    agent = reasoning_engines.ReasoningEngine(AGENT_ENGINE_ID)
    print("Connected!\n")

    # Test 1: Introduce myself (with user_id for session)
    print("-" * 40)
    print("Test 1: Introducing myself")
    print("-" * 40)
    message1 = "안녕! 나는 Luke야."
    print(f"You: {message1}")

    response1 = agent.query(message=message1, user_id=USER_ID)
    if isinstance(response1, dict):
        answer1 = response1.get("response", response1.get("output", str(response1)))
        session_id = response1.get("session_id")
    else:
        answer1 = str(response1)
        session_id = None
    print(f"Agent: {answer1}")
    if session_id:
        print(f"[Session ID: {session_id}]")
    print()

    # Test 2: Ask who I am (with same user_id and session_id)
    print("-" * 40)
    print("Test 2: Asking who I am")
    print("-" * 40)
    message2 = "내가 누구라고?"
    print(f"You: {message2}")

    query_params = {"message": message2, "user_id": USER_ID}
    if session_id:
        query_params["session_id"] = session_id

    response2 = agent.query(**query_params)
    if isinstance(response2, dict):
        answer2 = response2.get("response", response2.get("output", str(response2)))
    else:
        answer2 = str(response2)
    print(f"Agent: {answer2}\n")

    # Result
    print("=" * 60)
    print("Test Result")
    print("=" * 60)
    if "luke" in answer2.lower() or "Luke" in answer2:
        print("SUCCESS: Agent remembered the name 'Luke'!")
    else:
        print("NOTICE: Agent may not have remembered the name.")
        print("This could be due to:")
        print("  - Session not being maintained between queries")
        print("  - Memory not being stored/retrieved properly")
        print("\nDebug info:")
        print(f"  - Response1: {response1}")
        print(f"  - Response2: {response2}")
    print("=" * 60)


if __name__ == "__main__":
    main()
