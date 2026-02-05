#!/usr/bin/env python3
"""Promote a version from staging to production.

Usage:
    python scripts/version/promote.py
    python scripts/version/promote.py --version v1.0.0
    python scripts/version/promote.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

from agent_engine.version import VersionRegistry
from agent_engine.version.models import Environment


def deploy_to_production(agent_engine_id: str, project_id: str | None = None) -> bool:
    """Deploy/update agent engine for production.

    In a real implementation, this would:
    1. Update production load balancer to point to this deployment
    2. Run smoke tests
    3. Update DNS/routing
    """
    print(f"  Deploying {agent_engine_id} to production...")

    # Verify deployment exists
    if project_id:
        cmd = [
            "gcloud",
            "ai",
            "agent-engines",
            "describe",
            agent_engine_id,
            f"--project={project_id}",
            "--format=value(name)",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)  # noqa: S603
            if result.returncode != 0:
                print(f"  Error: Agent engine not found: {result.stderr}")
                return False
        except Exception as e:
            print(f"  Warning: Could not verify deployment: {e}")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote staging version to production")
    parser.add_argument(
        "--version",
        "-v",
        help="Specific version to promote (defaults to current staging)",
    )
    parser.add_argument(
        "--project-id",
        help="GCP project ID for verification",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate promotion without changes",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--registry-path",
        help="Path to registry.yaml",
    )

    args = parser.parse_args()

    # Initialize registry
    registry_path = (
        args.registry_path or Path(__file__).parent.parent.parent / "versions" / "registry.yaml"
    )
    registry = VersionRegistry(registry_path=registry_path)

    # Get staging version to promote
    if args.version:
        staging_version = registry.get_version(args.version, Environment.STAGING)
        if not staging_version:
            print(f"Error: Version {args.version} not found in staging")
            return 1
    else:
        staging_version = registry.get_current_version(Environment.STAGING)
        if not staging_version:
            print("Error: No current staging version found")
            return 1

    # Check current production version
    prod_version = registry.get_current_version(Environment.PRODUCTION)

    print("=" * 60)
    print("VERSION PROMOTION: Staging -> Production")
    print("=" * 60)
    print(f"\nStaging version to promote: {staging_version.version}")
    print(f"  Agent Engine ID: {staging_version.agent_engine_id or 'Not set'}")
    print(f"  Deployed at: {staging_version.deployed_at.isoformat()}")
    print(f"  Commit SHA: {staging_version.commit_sha or 'Unknown'}")

    if prod_version:
        print(f"\nCurrent production version: {prod_version.version}")
        print(f"  Agent Engine ID: {prod_version.agent_engine_id or 'Not set'}")
    else:
        print("\nNo current production version")

    print("-" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would promote version - no changes made")
        return 0

    # Confirm promotion
    if not args.force:
        response = input("\nProceed with promotion? [y/N]: ")
        if response.lower() != "y":
            print("Promotion cancelled")
            return 0

    # Perform promotion
    print("\nPromoting to production...")

    if staging_version.agent_engine_id:
        success = deploy_to_production(
            staging_version.agent_engine_id,
            args.project_id,
        )
        if not success:
            print("Error: Deployment verification failed")
            return 1

    # Register production version
    prod_registered = registry.register_version(
        version=staging_version.version,
        environment=Environment.PRODUCTION,
        agent_engine_id=staging_version.agent_engine_id,
        commit_sha=staging_version.commit_sha,
        metadata={
            **staging_version.metadata,
            "promoted_from": "staging",
        },
    )

    print("\nPromotion successful!")
    print(f"  Production version: {prod_registered.version}")
    print(f"  Status: {prod_registered.status.value}")
    print(f"  Promoted at: {prod_registered.deployed_at.isoformat()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
