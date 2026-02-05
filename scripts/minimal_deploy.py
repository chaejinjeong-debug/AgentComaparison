#!/usr/bin/env python3
"""Minimal deployment script for VertexAI Reasoning Engine.

This script creates and deploys a minimal agent that uses Gemini directly
without external module dependencies.

Configuration is loaded from .env file in the AgentEngine directory.
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv

from agent_engine.envs import Env

# Load .env file from AgentEngine directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Get defaults from environment (using Env singleton)
ENV_PROJECT = Env.AGENT_PROJECT_ID
ENV_LOCATION = Env.AGENT_LOCATION or "us-central1"
ENV_MODEL = Env.AGENT_MODEL or "gemini-2.0-flash"
ENV_DISPLAY_NAME = Env.AGENT_DISPLAY_NAME or "gemini-agent"
ENV_DESCRIPTION = Env.AGENT_DESCRIPTION or "Gemini Agent on Reasoning Engine"
ENV_SYSTEM_PROMPT = Env.AGENT_SYSTEM_PROMPT

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# Define agent class inline to avoid module reference issues with cloudpickle
class GeminiAgent:
    """Minimal Gemini agent for Reasoning Engine deployment."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        project: str = "",
        location: str = "us-central1",
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> None:
        self.model = model
        self.project = project
        self.location = location
        self.system_prompt = system_prompt
        self._client = None

    def set_up(self) -> None:
        """Initialize the Gemini client."""
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=self.project, location=self.location)
        self._client = GenerativeModel(
            model_name=self.model,
            system_instruction=self.system_prompt,
        )

    def query(self, *, message: str, **kwargs: Any) -> dict[str, Any]:
        """Query the agent."""
        if self._client is None:
            self.set_up()

        start_time = datetime.now(UTC)

        try:
            response = self._client.generate_content(message)
            response_text = response.text

            return {
                "response": response_text,
                "metadata": {
                    "model": self.model,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "latency_ms": (datetime.now(UTC) - start_time).total_seconds() * 1000,
                },
            }
        except Exception as e:
            return {
                "response": f"Error: {e}",
                "metadata": {"error": str(e)},
            }


def deploy_agent(
    project: str,
    location: str,
    display_name: str,
    description: str,
    model: str,
    system_prompt: str,
    staging_bucket: str | None = None,
) -> str:
    """Deploy the agent to VertexAI Reasoning Engine."""
    import vertexai
    from vertexai.preview.reasoning_engines import ReasoningEngine

    logger.info(
        "starting_deployment",
        project=project,
        location=location,
        display_name=display_name,
    )

    # Create staging bucket if not provided
    if not staging_bucket:
        staging_bucket = f"gs://{project}-agent-staging-{location.replace('-', '')}"

    # Initialize Vertex AI with staging bucket
    vertexai.init(project=project, location=location, staging_bucket=staging_bucket)

    # Create agent instance
    agent = GeminiAgent(
        model=model,
        project=project,
        location=location,
        system_prompt=system_prompt,
    )

    # Minimal requirements
    requirements = [
        "google-cloud-aiplatform>=1.78.0",
    ]

    # Deploy to Reasoning Engine
    deployed_agent = ReasoningEngine.create(
        reasoning_engine=agent,
        requirements=requirements,
        display_name=display_name,
        description=description,
    )

    agent_name = deployed_agent.resource_name

    logger.info(
        "deployment_complete",
        agent_name=agent_name,
    )

    return agent_name


def list_agents(project: str, location: str) -> list[dict]:
    """List all deployed agents."""
    import vertexai
    from vertexai.preview.reasoning_engines import ReasoningEngine

    vertexai.init(project=project, location=location)
    agents = ReasoningEngine.list()

    return [
        {
            "name": agent.resource_name,
            "display_name": getattr(agent, "display_name", "N/A"),
        }
        for agent in agents
    ]


def query_agent(agent_name: str, project: str, location: str, message: str) -> dict:
    """Query a deployed agent."""
    import vertexai
    from vertexai.preview.reasoning_engines import ReasoningEngine

    vertexai.init(project=project, location=location)
    agent = ReasoningEngine(agent_name)
    return agent.query(message=message)


def delete_agent(agent_name: str, project: str, location: str) -> bool:
    """Delete a deployed agent."""
    import vertexai
    from vertexai.preview.reasoning_engines import ReasoningEngine

    vertexai.init(project=project, location=location)
    agent = ReasoningEngine(agent_name)
    agent.delete()
    return True


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy minimal Gemini agent to VertexAI Reasoning Engine"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy a new agent")
    deploy_parser.add_argument(
        "--project", default=ENV_PROJECT, help=f"GCP project ID (default: {ENV_PROJECT})"
    )
    deploy_parser.add_argument(
        "--location", default=ENV_LOCATION, help=f"GCP region (default: {ENV_LOCATION})"
    )
    deploy_parser.add_argument(
        "--model", default=ENV_MODEL, help=f"Gemini model (default: {ENV_MODEL})"
    )
    deploy_parser.add_argument(
        "--display-name",
        default=ENV_DISPLAY_NAME,
        help=f"Display name (default: {ENV_DISPLAY_NAME})",
    )
    deploy_parser.add_argument("--description", default=ENV_DESCRIPTION, help="Description")
    deploy_parser.add_argument("--system-prompt", default=ENV_SYSTEM_PROMPT, help="System prompt")
    deploy_parser.add_argument("--staging-bucket", help="GCS staging bucket")

    # List command
    list_parser = subparsers.add_parser("list", help="List agents")
    list_parser.add_argument(
        "--project", default=ENV_PROJECT, help=f"GCP project ID (default: {ENV_PROJECT})"
    )
    list_parser.add_argument(
        "--location", default=ENV_LOCATION, help=f"GCP region (default: {ENV_LOCATION})"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query an agent")
    query_parser.add_argument(
        "--project", default=ENV_PROJECT, help=f"GCP project ID (default: {ENV_PROJECT})"
    )
    query_parser.add_argument(
        "--location", default=ENV_LOCATION, help=f"GCP region (default: {ENV_LOCATION})"
    )
    query_parser.add_argument("--agent-name", required=True, help="Agent resource name")
    query_parser.add_argument("--message", required=True, help="Message to send")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete an agent")
    delete_parser.add_argument(
        "--project", default=ENV_PROJECT, help=f"GCP project ID (default: {ENV_PROJECT})"
    )
    delete_parser.add_argument(
        "--location", default=ENV_LOCATION, help=f"GCP region (default: {ENV_LOCATION})"
    )
    delete_parser.add_argument("--agent-name", required=True, help="Agent resource name")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Validate project is set
    if not args.project:
        print("\nError: --project is required. Set AGENT_PROJECT_ID in .env or pass --project")
        sys.exit(1)

    try:
        if args.command == "deploy":
            agent_name = deploy_agent(
                project=args.project,
                location=args.location,
                display_name=args.display_name,
                description=args.description,
                model=args.model,
                system_prompt=args.system_prompt,
                staging_bucket=args.staging_bucket,
            )
            print("\nAgent deployed successfully!")
            print(f"Agent Name: {agent_name}")

        elif args.command == "list":
            agents = list_agents(project=args.project, location=args.location)
            print(f"\nDeployed Agents ({len(agents)}):")
            for agent in agents:
                print(f"  - {agent['display_name']}: {agent['name']}")

        elif args.command == "query":
            response = query_agent(
                agent_name=args.agent_name,
                project=args.project,
                location=args.location,
                message=args.message,
            )
            print("\nAgent Response:")
            if isinstance(response, dict):
                print(f"  {response.get('response', response)}")
            else:
                print(f"  {response}")

        elif args.command == "delete":
            if not args.force:
                confirm = input(f"Delete agent {args.agent_name}? [y/N]: ")
                if confirm.lower() != "y":
                    print("Deletion cancelled.")
                    sys.exit(0)

            delete_agent(
                agent_name=args.agent_name,
                project=args.project,
                location=args.location,
            )
            print("\nAgent deleted successfully!")

    except Exception as e:
        logger.error("command_failed", error=str(e), command=args.command)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
