#!/usr/bin/env python3
"""Verify Agent Engine deployment.

This script lists deployed agent engines using the Vertex AI Python SDK,
replacing the unavailable `gcloud ai agent-engines list` command.

Usage:
    python scripts/ci/verify_deployment.py --project YOUR_PROJECT --location asia-northeast3
"""

import argparse
import sys


def list_agent_engines(project: str, location: str) -> list[dict]:
    """List all deployed agent engines.

    Args:
        project: GCP project ID
        location: GCP region

    Returns:
        List of agent engine information dictionaries
    """
    import vertexai

    client = vertexai.Client(project=project, location=location)

    agents = []
    for agent in client.agent_engines.list():
        resource = agent.api_resource
        agents.append(
            {
                "name": resource.name,
                "display_name": resource.display_name,
                "state": getattr(resource, "state", "UNKNOWN"),
            }
        )

    return agents


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify Agent Engine deployment")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--location", default="asia-northeast3", help="GCP region")

    args = parser.parse_args()

    try:
        agents = list_agent_engines(args.project, args.location)

        if not agents:
            print("No agent engines found.")
            return 0

        # Print table header
        print(f"{'NAME':<60} {'DISPLAY_NAME':<30} {'STATE':<15}")
        print("-" * 105)

        # Print agents
        for agent in agents:
            print(f"{agent['name']:<60} {agent['display_name']:<30} {agent['state']:<15}")

        print(f"\nTotal: {len(agents)} agent(s)")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
