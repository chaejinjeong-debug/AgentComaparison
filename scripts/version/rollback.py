#!/usr/bin/env python3
"""Execute a rollback to a previous version.

Usage:
    python scripts/version/rollback.py --env staging
    python scripts/version/rollback.py --env production --target v1.0.0
    python scripts/version/rollback.py --env staging --dry-run
"""

import argparse
import sys
from pathlib import Path


from agent_engine.version import RollbackManager, VersionRegistry
from agent_engine.version.models import Environment


def main() -> int:
    parser = argparse.ArgumentParser(description="Rollback to a previous version")
    parser.add_argument(
        "--env",
        "-e",
        choices=["staging", "production"],
        required=True,
        help="Target environment",
    )
    parser.add_argument(
        "--target",
        "-t",
        help="Specific version to rollback to (defaults to previous)",
    )
    parser.add_argument(
        "--reason",
        "-r",
        default="",
        help="Reason for rollback",
    )
    parser.add_argument(
        "--project-id",
        help="GCP project ID",
    )
    parser.add_argument(
        "--location",
        default="asia-northeast3",
        help="GCP region",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate rollback without changes",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List rollback candidates",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show rollback history",
    )
    parser.add_argument(
        "--registry-path",
        help="Path to registry.yaml",
    )

    args = parser.parse_args()

    # Initialize components
    registry_path = (
        args.registry_path or Path(__file__).parent.parent.parent / "versions" / "registry.yaml"
    )
    registry = VersionRegistry(registry_path=registry_path)
    rollback_manager = RollbackManager(
        registry=registry,
        project_id=args.project_id,
        location=args.location,
    )

    environment = Environment.STAGING if args.env == "staging" else Environment.PRODUCTION

    # Handle --list flag
    if args.list:
        print(f"Rollback candidates for {environment.value}:")
        candidates = rollback_manager.list_rollback_candidates(environment)
        if not candidates:
            print("  No candidates available")
        for v in candidates:
            print(f"  - {v.version} (deployed: {v.deployed_at.strftime('%Y-%m-%d %H:%M')})")
        return 0

    # Handle --history flag
    if args.history:
        print(f"Rollback history for {environment.value}:")
        history = rollback_manager.get_rollback_history(environment)
        if not history:
            print("  No rollback history")
        for r in history:
            print(f"  - {r.from_version} -> {r.to_version}")
            print(f"    Time: {r.executed_at.strftime('%Y-%m-%d %H:%M')}")
            print(f"    By: {r.executed_by}")
            if r.reason:
                print(f"    Reason: {r.reason}")
        return 0

    # Check if rollback is possible
    can_rollback, message = rollback_manager.can_rollback(environment)
    if not can_rollback:
        print(f"Error: Cannot rollback - {message}")
        return 1

    # Get current and target versions
    current = registry.get_current_version(environment)
    if args.target:
        target = registry.get_version(args.target, environment)
        if not target:
            print(f"Error: Target version {args.target} not found")
            return 1
    else:
        target = rollback_manager.get_rollback_target(environment)

    if not current or not target:
        print("Error: Could not determine versions for rollback")
        return 1

    print("=" * 60)
    print(f"ROLLBACK: {environment.value.upper()}")
    print("=" * 60)
    print(f"\nCurrent version: {current.version}")
    print(f"  Agent Engine ID: {current.agent_engine_id or 'Not set'}")
    print(f"  Deployed at: {current.deployed_at.isoformat()}")
    print(f"\nRollback target: {target.version}")
    print(f"  Agent Engine ID: {target.agent_engine_id or 'Not set'}")
    print(f"  Deployed at: {target.deployed_at.isoformat()}")
    if args.reason:
        print(f"\nReason: {args.reason}")
    print("-" * 60)

    if args.dry_run:
        result = rollback_manager.execute_rollback(
            environment=environment,
            target_version=args.target,
            reason=args.reason,
            dry_run=True,
        )
        print(f"\n{result.message}")
        return 0

    # Confirm rollback
    if not args.force:
        response = input("\nProceed with rollback? [y/N]: ")
        if response.lower() != "y":
            print("Rollback cancelled")
            return 0

    # Execute rollback
    print("\nExecuting rollback...")
    result = rollback_manager.execute_rollback(
        environment=environment,
        target_version=args.target,
        reason=args.reason,
        dry_run=False,
    )

    if result.success:
        print(f"\n{result.message}")
        return 0
    else:
        print(f"\nError: {result.message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
