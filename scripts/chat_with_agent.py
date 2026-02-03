#!/usr/bin/env python3
"""Interactive chat with deployed VertexAI Agent Engine.

Usage:
    cd AgentEngine && uv run python scripts/chat_with_agent.py
"""

import vertexai
from vertexai.preview import reasoning_engines

# Configuration
PROJECT_ID = "heum-alfred-evidence-clf-dev"
LOCATION = "asia-northeast3"
AGENT_ENGINE_ID = "6406440838678708224"


def main():
    print("=" * 60)
    print("VertexAI Agent Engine Chat")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print(f"Agent Engine ID: {AGENT_ENGINE_ID}")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end the chat")
    print("=" * 60 + "\n")

    # Initialize VertexAI
    print("Connecting to VertexAI Agent Engine...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Get deployed agent
    agent = reasoning_engines.ReasoningEngine(AGENT_ENGINE_ID)
    print("Connected!\n")

    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("Bye!")
                break

            # Query the agent
            response = agent.query(message=user_input)

            # Handle response format
            if isinstance(response, dict):
                answer = response.get("response", response.get("output", str(response)))
            else:
                answer = str(response)

            print(f"Agent: {answer}\n")

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
