#!/usr/bin/env python3
"""Source-based deployment script for VertexAI Agent Engine.

This script deploys the Pydantic AI Agent to VertexAI Agent Engine
using the source-based deployment method, suitable for CI/CD pipelines.

Requirement: DM-002 - Source-based deployment (CI/CD)

Usage:
    python scripts/deploy_source.py deploy --project YOUR_PROJECT --location asia-northeast3

    # With labels for versioning
    python scripts/deploy_source.py deploy \\
        --project YOUR_PROJECT \\
        --location asia-northeast3 \\
        --labels "environment=staging,version=v1.2.0"

    # Upsert mode (default): update if exists, create if not
    python scripts/deploy_source.py deploy \\
        --project YOUR_PROJECT \\
        --display-name "pydantic-ai-agent-staging" \\
        --upsert
"""

import argparse
import sys
from pathlib import Path

import google.auth
import google.auth.transport.requests
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
# agent_engine/ is at project root, so it uploads to /code/agent_engine/
SOURCE_PACKAGES = ["agent_engine"]

# Entrypoint configuration
ENTRYPOINT_MODULE = "agent_engine.agent"
ENTRYPOINT_OBJECT = "agent"

# Class methods exposed to Agent Engine
# Supported api_mode values: "", "async", "stream", "async_stream", "a2a_extension"
# Using api_mode="async" for query to run it asynchronously in Agent Engine
CLASS_METHODS = [
    {
        "name": "query",
        "api_mode": "async",  # Use async mode to avoid event loop conflicts
        "parameters": [
            {"name": "message", "type": "STRING", "required": True},
            {"name": "user_id", "type": "STRING", "required": False},
            {"name": "session_id", "type": "STRING", "required": False},
        ],
    },
]


def parse_labels(labels_str: str) -> dict[str, str]:
    """Parse labels string into dictionary.

    Args:
        labels_str: Comma-separated key=value pairs (e.g., "environment=staging,version=v1.0.0")

    Returns:
        Dictionary of labels
    """
    if not labels_str:
        return {}

    labels = {}
    for pair in labels_str.split(","):
        pair = pair.strip()
        if "=" in pair:
            key, value = pair.split("=", 1)
            labels[key.strip()] = value.strip()
    return labels


def find_agent_by_display_name(client, display_name: str) -> str | None:
    """Find an existing agent by display_name.

    Args:
        client: Vertex AI client
        display_name: Agent display name to search for

    Returns:
        Agent resource name if found, None otherwise
    """
    try:
        agents = client.agent_engines.list()
        for agent in agents:
            if agent.display_name == display_name:
                logger.info(
                    "agent_found_by_display_name",
                    display_name=display_name,
                    agent_name=agent.api_resource.name,
                )
                return agent.api_resource.name
    except Exception as e:
        logger.warning("agent_search_failed", error=str(e))

    logger.info("agent_not_found", display_name=display_name)
    return None


def update_agent_labels(
    project: str, location: str, agent_name: str, labels: dict[str, str]
) -> None:
    """Update agent labels using REST API.

    The Python SDK doesn't support labels, so we use the REST API directly.

    Args:
        project: GCP project ID
        location: GCP region
        agent_name: Full agent resource name
        labels: Dictionary of labels to set
    """
    import json
    import urllib.request

    if not labels:
        logger.info("no_labels_to_update")
        return

    # Get credentials
    credentials, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)

    # Build REST API URL
    # Agent name format: projects/{project}/locations/{location}/reasoningEngines/{id}
    api_url = f"https://{location}-aiplatform.googleapis.com/v1/{agent_name}"

    # Prepare PATCH request with labels
    patch_data = {"labels": labels}
    update_mask = "labels"

    url_with_mask = f"{api_url}?updateMask={update_mask}"

    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }

    request = urllib.request.Request(
        url_with_mask,
        data=json.dumps(patch_data).encode("utf-8"),
        headers=headers,
        method="PATCH",
    )

    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))
            logger.info(
                "labels_updated_successfully",
                agent_name=agent_name,
                labels=labels,
            )
            return result
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        logger.error(
            "labels_update_failed",
            status=e.code,
            error=error_body,
        )
        raise


def deploy_from_source(
    project: str,
    location: str,
    display_name: str,
    description: str,
    staging_bucket: str | None = None,
    model: str = "gemini-2.5-pro",
    system_prompt: str = "You are a helpful AI assistant.",
    labels: dict[str, str] | None = None,
    upsert: bool = True,
) -> str:
    """Deploy the agent from source files with upsert support.

    Args:
        project: GCP project ID
        location: GCP region
        display_name: Agent display name
        description: Agent description
        staging_bucket: Optional GCS bucket for staging files
        model: Gemini model name
        system_prompt: System prompt for the agent
        labels: Optional labels dict (environment, version, etc.)
        upsert: If True, update existing agent instead of creating new

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
        upsert=upsert,
        labels=labels,
    )

    # Create Vertex AI client
    client = vertexai.Client(project=project, location=location)

    # Change to project root directory for relative path resolution
    import os

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    logger.info("working_directory_changed", cwd=str(project_root))

    # Use relative paths for source packages (SDK expects relative paths)
    source_packages = SOURCE_PACKAGES

    # Verify source packages exist
    for pkg in source_packages:
        if not Path(pkg).exists():
            raise FileNotFoundError(f"Source package not found: {pkg}")

    logger.info("source_packages_verified", packages=source_packages)

    # Check for existing agent if upsert mode
    existing_agent_name = None
    if upsert:
        existing_agent_name = find_agent_by_display_name(client, display_name)

    if existing_agent_name:
        # Update existing agent
        logger.info("updating_existing_agent", agent_name=existing_agent_name)
        agent = client.agent_engines.get(name=existing_agent_name)
        agent.update(
            config={
                "source_packages": source_packages,
                "entrypoint_module": ENTRYPOINT_MODULE,
                "entrypoint_object": ENTRYPOINT_OBJECT,
                "class_methods": CLASS_METHODS,
                "description": description,
                "requirements_file": "agent_engine/requirements.txt",
                "env_vars": {
                    "AGENT_LOCATION": location,
                    "AGENT_MODEL": model,
                },
            }
        )
        agent_name = agent.api_resource.name
        logger.info("agent_updated", agent_name=agent_name)
    else:
        # Create new agent
        logger.info("creating_new_agent", display_name=display_name)
        deployed_agent = client.agent_engines.create(
            config={
                "source_packages": source_packages,
                "entrypoint_module": ENTRYPOINT_MODULE,
                "entrypoint_object": ENTRYPOINT_OBJECT,
                "class_methods": CLASS_METHODS,
                "display_name": display_name,
                "description": description,
                "requirements_file": "agent_engine/requirements.txt",
                "env_vars": {
                    "AGENT_LOCATION": location,
                    "AGENT_MODEL": model,
                },
            }
        )
        agent_name = deployed_agent.api_resource.name

    # Update labels via REST API (SDK doesn't support labels)
    if labels:
        update_agent_labels(project, location, agent_name, labels)

    logger.info(
        "source_deployment_complete",
        agent_name=agent_name,
        display_name=display_name,
        labels=labels,
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

    # Change to project root directory for relative path resolution
    import os

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Use relative paths for source packages
    source_packages = SOURCE_PACKAGES

    # Get existing agent
    agent = client.agent_engines.get(name=agent_name)

    # Update from source
    agent.update(
        config={
            "source_packages": source_packages,
            "entrypoint_module": ENTRYPOINT_MODULE,
            "entrypoint_object": ENTRYPOINT_OBJECT,
            "requirements_file": "agent_engine/requirements.txt",
        }
    )

    logger.info(
        "source_update_complete",
        agent_name=agent.api_resource.name,
    )

    return agent.api_resource.name


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

    agent = client.agent_engines.get(name=agent_name)

    # Run a test query (using async method)
    import asyncio

    test_message = "Hello! Can you confirm you are working correctly?"

    # The agent.query returns a coroutine for async mode, need to await it
    async def run_query():
        result = agent.query(message=test_message)
        # If result is a coroutine, await it
        if asyncio.iscoroutine(result):
            return await result
        return result

    response = asyncio.run(run_query())

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
        "--labels",
        default="",
        help="Comma-separated labels (e.g., 'environment=staging,version=v1.0.0')",
    )
    deploy_parser.add_argument(
        "--upsert",
        action="store_true",
        default=True,
        help="Update existing agent if found (default: True)",
    )
    deploy_parser.add_argument(
        "--no-upsert",
        action="store_false",
        dest="upsert",
        help="Always create new agent",
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
            # Parse labels from command line
            labels = parse_labels(args.labels) if args.labels else None

            agent_name = deploy_from_source(
                project=args.project,
                location=args.location,
                display_name=args.display_name,
                description=args.description,
                staging_bucket=args.staging_bucket,
                model=args.model,
                system_prompt=args.system_prompt,
                labels=labels,
                upsert=args.upsert,
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
