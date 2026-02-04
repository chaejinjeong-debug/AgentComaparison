#!/usr/bin/env python3
"""Source-based deployment script for VertexAI Agent Engine.

This script deploys the Pydantic AI Agent to VertexAI Agent Engine
using the source-based deployment method, suitable for CI/CD pipelines.

Requirement: DM-002 - Source-based deployment (CI/CD)

Usage:
    python scripts/deploy_source.py --project YOUR_PROJECT --location asia-northeast3

    # With custom staging bucket
    python scripts/deploy_source.py \\
        --project YOUR_PROJECT \\
        --location asia-northeast3 \\
        --staging-bucket gs://your-bucket/staging
"""

import argparse
import sys
from pathlib import Path


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


# Source package configuration
SOURCE_PACKAGES = [
    "agent_engine",  # Main package
]

# Entrypoint configuration
ENTRYPOINT_MODULE = "agent_engine.agent"
ENTRYPOINT_OBJECT = "PydanticAIAgentWrapper"

# Class methods exposed to Agent Engine
CLASS_METHODS = [
    {
        "name": "query",
        "api_mode": "SYNC",
        "parameters": [
            {"name": "message", "type": "STRING", "required": True},
            {"name": "user_id", "type": "STRING", "required": False},
            {"name": "session_id", "type": "STRING", "required": False},
        ],
    },
    {
        "name": "aquery",
        "api_mode": "ASYNC",
        "parameters": [
            {"name": "message", "type": "STRING", "required": True},
            {"name": "user_id", "type": "STRING", "required": False},
            {"name": "session_id", "type": "STRING", "required": False},
        ],
    },
]


def deploy_from_source(
    project: str,
    location: str,
    display_name: str,
    description: str,
    staging_bucket: str | None = None,
    model: str = "gemini-2.5-pro",
    system_prompt: str = "You are a helpful AI assistant.",
) -> str:
    """Deploy the agent from source files.

    Args:
        project: GCP project ID
        location: GCP region
        display_name: Agent display name
        description: Agent description
        staging_bucket: Optional GCS bucket for staging files
        model: Gemini model name
        system_prompt: System prompt for the agent

    Returns:
        Deployed agent resource name
    """
    import vertexai

    logger.info(
        "starting_source_deployment",
        project=project,
        location=location,
        display_name=display_name,
        staging_bucket=staging_bucket,
    )

    # Create Vertex AI client
    client = vertexai.Client(project=project, location=location)

    # Get absolute paths for source packages
    project_root = Path(__file__).parent.parent
    source_packages = [str(project_root / pkg) for pkg in SOURCE_PACKAGES]

    # Verify source packages exist
    for pkg in source_packages:
        if not Path(pkg).exists():
            raise FileNotFoundError(f"Source package not found: {pkg}")

    logger.info("source_packages_verified", packages=source_packages)

    # Create temporary requirements file for deployment
    # Include all core dependencies from pyproject.toml
    requirements_content = """pydantic-ai-slim[google]>=1.51.0
google-cloud-aiplatform[agent_engines]>=1.78.0
structlog>=24.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-gcp-trace>=1.6.0
opentelemetry-exporter-gcp-monitoring>=1.6.0
google-cloud-trace>=1.11.0
google-cloud-logging>=3.8.0
google-cloud-monitoring>=2.18.0
pyyaml>=6.0.3
"""
    requirements_file = project_root / ".agent_requirements.txt"
    requirements_file.write_text(requirements_content)
    logger.info("requirements_file_created", path=str(requirements_file))

    # Deploy from source using client.agent_engines.create()
    deployed_agent = client.agent_engines.create(
        config={
            "source_packages": source_packages,
            "entrypoint_module": ENTRYPOINT_MODULE,
            "entrypoint_object": ENTRYPOINT_OBJECT,
            "class_methods": CLASS_METHODS,
            "display_name": display_name,
            "description": description,
            "requirements_file": str(requirements_file),
            "env_vars": {
                "AGENT_LOCATION": location,
            },
        }
    )

    agent_name = deployed_agent.resource_name

    logger.info(
        "source_deployment_complete",
        agent_name=agent_name,
        display_name=display_name,
    )

    return agent_name


def update_agent_from_source(
    agent_name: str,
    project: str,
    location: str,
    staging_bucket: str | None = None,
) -> str:
    """Update an existing agent from source files.

    Args:
        agent_name: Agent resource name or ID
        project: GCP project ID
        location: GCP region
        staging_bucket: Optional GCS bucket for staging files

    Returns:
        Updated agent resource name
    """
    import vertexai

    logger.info(
        "starting_source_update",
        agent_name=agent_name,
        project=project,
        location=location,
    )

    # Create Vertex AI client
    client = vertexai.Client(project=project, location=location)

    # Get absolute paths for source packages
    project_root = Path(__file__).parent.parent
    source_packages = [str(project_root / pkg) for pkg in SOURCE_PACKAGES]

    # Get existing agent
    agent = client.agent_engines.get(agent_name)

    # Update from source
    agent.update(
        config={
            "source_packages": source_packages,
            "entrypoint_module": ENTRYPOINT_MODULE,
            "entrypoint_object": ENTRYPOINT_OBJECT,
        }
    )

    logger.info(
        "source_update_complete",
        agent_name=agent.resource_name,
    )

    return agent.resource_name


def verify_deployment(agent_name: str, project: str, location: str) -> dict:
    """Verify the deployment by running a test query.

    Args:
        agent_name: Agent resource name or ID
        project: GCP project ID
        location: GCP region

    Returns:
        Verification result dictionary
    """
    import vertexai

    # Create Vertex AI client
    client = vertexai.Client(project=project, location=location)

    agent = client.agent_engines.get(agent_name)

    # Run a test query
    test_message = "Hello! Can you confirm you are working correctly?"
    response = agent.query(message=test_message)

    return {
        "agent_name": agent_name,
        "test_message": test_message,
        "response": response,
        "status": "OK" if response else "FAILED",
    }


def main() -> None:
    """Main entry point for the source deployment script."""
    parser = argparse.ArgumentParser(
        description="Deploy Pydantic AI Agent from source to VertexAI Agent Engine"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy a new agent from source")
    deploy_parser.add_argument("--project", required=True, help="GCP project ID")
    deploy_parser.add_argument("--location", default="asia-northeast3", help="GCP region")
    deploy_parser.add_argument("--staging-bucket", help="GCS bucket for staging (gs://...)")
    deploy_parser.add_argument(
        "--display-name", default="pydantic-ai-agent", help="Agent display name"
    )
    deploy_parser.add_argument(
        "--description",
        default="Pydantic AI Agent on VertexAI Agent Engine",
        help="Agent description",
    )
    deploy_parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name")
    deploy_parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant.",
        help="System prompt",
    )
    deploy_parser.add_argument(
        "--verify", action="store_true", help="Verify deployment with test query"
    )

    # Update command
    update_parser = subparsers.add_parser("update", help="Update an existing agent")
    update_parser.add_argument("--project", required=True, help="GCP project ID")
    update_parser.add_argument("--location", default="asia-northeast3", help="GCP region")
    update_parser.add_argument("--agent-name", required=True, help="Agent resource name")
    update_parser.add_argument("--staging-bucket", help="GCS bucket for staging (gs://...)")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify agent deployment")
    verify_parser.add_argument("--project", required=True, help="GCP project ID")
    verify_parser.add_argument("--location", default="asia-northeast3", help="GCP region")
    verify_parser.add_argument("--agent-name", required=True, help="Agent resource name")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "deploy":
            agent_name = deploy_from_source(
                project=args.project,
                location=args.location,
                display_name=args.display_name,
                description=args.description,
                staging_bucket=args.staging_bucket,
                model=args.model,
                system_prompt=args.system_prompt,
            )

            print("\nAgent deployed successfully from source!")
            print(f"Agent Name: {agent_name}")

            if args.verify:
                print("\nVerifying deployment...")
                result = verify_deployment(
                    agent_name=agent_name,
                    project=args.project,
                    location=args.location,
                )
                print(f"Verification Status: {result['status']}")
                if result["response"]:
                    print(f"Test Response: {result['response']}")

        elif args.command == "update":
            agent_name = update_agent_from_source(
                agent_name=args.agent_name,
                project=args.project,
                location=args.location,
                staging_bucket=args.staging_bucket,
            )

            print("\nAgent updated successfully from source!")
            print(f"Agent Name: {agent_name}")

        elif args.command == "verify":
            result = verify_deployment(
                agent_name=args.agent_name,
                project=args.project,
                location=args.location,
            )

            print("\nVerification Result:")
            print(f"  Agent: {result['agent_name']}")
            print(f"  Status: {result['status']}")
            print(f"  Test Message: {result['test_message']}")
            if result["response"]:
                print(f"  Response: {result['response']}")

    except FileNotFoundError as e:
        logger.error("source_not_found", error=str(e))
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error("command_failed", error=str(e), command=args.command)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
