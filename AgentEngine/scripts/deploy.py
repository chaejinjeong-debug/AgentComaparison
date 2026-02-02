#!/usr/bin/env python3
"""SDK-based deployment script for VertexAI Agent Engine.

This script deploys the Pydantic AI Agent to VertexAI Agent Engine
using the SDK-based deployment method.

Requirement: DM-001 - SDK-based deployment

Usage:
    python scripts/deploy.py --project YOUR_PROJECT --location asia-northeast3

    # With custom configuration
    python scripts/deploy.py \\
        --project YOUR_PROJECT \\
        --location asia-northeast3 \\
        --model gemini-2.5-pro \\
        --display-name my-agent
"""

import argparse
import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog

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


def create_agent_instance(
    model: str,
    project: str,
    location: str,
    system_prompt: str,
) -> "PydanticAIAgentWrapper":
    """Create an agent instance for deployment.

    Args:
        model: Gemini model name
        project: GCP project ID
        location: GCP region
        system_prompt: System prompt for the agent

    Returns:
        Configured PydanticAIAgentWrapper instance
    """
    from agent_engine.agent import PydanticAIAgentWrapper
    from agent_engine.tools import DEFAULT_TOOLS

    agent = PydanticAIAgentWrapper(
        model=model,
        project=project,
        location=location,
        system_prompt=system_prompt,
        tools=DEFAULT_TOOLS,
    )

    return agent


def deploy_agent(
    agent: "PydanticAIAgentWrapper",
    project: str,
    location: str,
    display_name: str,
    description: str,
) -> str:
    """Deploy the agent to VertexAI Agent Engine.

    Args:
        agent: Agent instance to deploy
        project: GCP project ID
        location: GCP region
        display_name: Agent display name
        description: Agent description

    Returns:
        Deployed agent resource name
    """
    from google.cloud import aiplatform

    logger.info(
        "starting_deployment",
        project=project,
        location=location,
        display_name=display_name,
    )

    # Initialize Vertex AI
    aiplatform.init(project=project, location=location)

    # Deploy to Agent Engine
    deployed_agent = aiplatform.agent_engines.create(
        agent=agent,
        config={
            "display_name": display_name,
            "description": description,
        },
    )

    agent_name = deployed_agent.resource_name

    logger.info(
        "deployment_complete",
        agent_name=agent_name,
        display_name=display_name,
    )

    return agent_name


def get_agent_status(agent_name: str, project: str, location: str) -> dict:
    """Get the status of a deployed agent.

    Args:
        agent_name: Agent resource name or ID
        project: GCP project ID
        location: GCP region

    Returns:
        Agent status dictionary
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)

    agent = aiplatform.agent_engines.get(agent_name)

    return {
        "name": agent.resource_name,
        "display_name": agent.display_name,
        "state": agent.state.name if hasattr(agent, "state") else "UNKNOWN",
        "create_time": str(agent.create_time) if hasattr(agent, "create_time") else None,
    }


def list_agents(project: str, location: str) -> list[dict]:
    """List all deployed agents.

    Args:
        project: GCP project ID
        location: GCP region

    Returns:
        List of agent information dictionaries
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)

    agents = aiplatform.agent_engines.list()

    return [
        {
            "name": agent.resource_name,
            "display_name": agent.display_name,
        }
        for agent in agents
    ]


def delete_agent(agent_name: str, project: str, location: str) -> bool:
    """Delete a deployed agent.

    Args:
        agent_name: Agent resource name or ID
        project: GCP project ID
        location: GCP region

    Returns:
        True if deletion was successful
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)

    agent = aiplatform.agent_engines.get(agent_name)
    agent.delete()

    logger.info("agent_deleted", agent_name=agent_name)
    return True


def main() -> None:
    """Main entry point for the deployment script."""
    parser = argparse.ArgumentParser(
        description="Deploy Pydantic AI Agent to VertexAI Agent Engine"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy a new agent")
    deploy_parser.add_argument("--project", required=True, help="GCP project ID")
    deploy_parser.add_argument(
        "--location", default="asia-northeast3", help="GCP region"
    )
    deploy_parser.add_argument(
        "--model", default="gemini-2.5-pro", help="Gemini model name"
    )
    deploy_parser.add_argument(
        "--display-name", default="pydantic-ai-agent", help="Agent display name"
    )
    deploy_parser.add_argument(
        "--description",
        default="Pydantic AI Agent on VertexAI Agent Engine",
        help="Agent description",
    )
    deploy_parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant.",
        help="System prompt",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Get agent status")
    status_parser.add_argument("--project", required=True, help="GCP project ID")
    status_parser.add_argument(
        "--location", default="asia-northeast3", help="GCP region"
    )
    status_parser.add_argument("--agent-name", required=True, help="Agent resource name")

    # List command
    list_parser = subparsers.add_parser("list", help="List all agents")
    list_parser.add_argument("--project", required=True, help="GCP project ID")
    list_parser.add_argument(
        "--location", default="asia-northeast3", help="GCP region"
    )

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete an agent")
    delete_parser.add_argument("--project", required=True, help="GCP project ID")
    delete_parser.add_argument(
        "--location", default="asia-northeast3", help="GCP region"
    )
    delete_parser.add_argument("--agent-name", required=True, help="Agent resource name")
    delete_parser.add_argument(
        "--force", action="store_true", help="Force deletion without confirmation"
    )

    args = parser.parse_args()

    if args.command is None:
        # Default to deploy if no command specified
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "deploy":
            agent = create_agent_instance(
                model=args.model,
                project=args.project,
                location=args.location,
                system_prompt=args.system_prompt,
            )

            agent_name = deploy_agent(
                agent=agent,
                project=args.project,
                location=args.location,
                display_name=args.display_name,
                description=args.description,
            )

            print(f"\nAgent deployed successfully!")
            print(f"Agent Name: {agent_name}")

        elif args.command == "status":
            status = get_agent_status(
                agent_name=args.agent_name,
                project=args.project,
                location=args.location,
            )
            print("\nAgent Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")

        elif args.command == "list":
            agents = list_agents(project=args.project, location=args.location)
            print(f"\nDeployed Agents ({len(agents)}):")
            for agent in agents:
                print(f"  - {agent['display_name']}: {agent['name']}")

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
            print(f"\nAgent {args.agent_name} deleted successfully!")

    except Exception as e:
        logger.error("command_failed", error=str(e), command=args.command)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
