#!/usr/bin/env python3
"""Register a new version in the version registry.

Usage:
    python scripts/version/register.py --version v1.0.0 --env staging
    python scripts/version/register.py --version v1.0.0 --env production --agent-id projects/xxx/...
    python scripts/version/register.py --version v1.0.0-test --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent_engine.version import VersionRegistry
from agent_engine.version.models import Environment


def get_git_info() -> tuple[str | None, str | None]:
    """Get current git commit SHA and branch."""
    try:
        sha = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        branch = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return (
            sha.stdout.strip() if sha.returncode == 0 else None,
            branch.stdout.strip() if branch.returncode == 0 else None,
        )
    except Exception:
        return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Register a new version")
    parser.add_argument(
        "--version",
        "-v",
        required=True,
        help="Version string (e.g., v1.0.0)",
    )
    parser.add_argument(
        "--env",
        "-e",
        choices=["staging", "production"],
        default="staging",
        help="Target environment",
    )
    parser.add_argument(
        "--agent-id",
        help="Agent Engine resource ID",
    )
    parser.add_argument(
        "--deployed-by",
        help="Deployer identity (defaults to current user)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate registration without writing",
    )
    parser.add_argument(
        "--registry-path",
        help="Path to registry.yaml (defaults to versions/registry.yaml)",
    )

    args = parser.parse_args()

    # Get git info
    commit_sha, branch = get_git_info()

    # Determine environment
    environment = Environment.STAGING if args.env == "staging" else Environment.PRODUCTION

    print(f"Registering version: {args.version}")
    print(f"  Environment: {environment.value}")
    print(f"  Agent Engine ID: {args.agent_id or 'Not specified'}")
    print(f"  Commit SHA: {commit_sha or 'Unknown'}")
    print(f"  Git Branch: {branch or 'Unknown'}")
    print(f"  Deployed By: {args.deployed_by or 'Current user'}")

    if args.dry_run:
        print("\n[DRY RUN] Would register version - no changes made")
        return 0

    # Initialize registry
    registry_path = args.registry_path or Path(__file__).parent.parent.parent / "versions" / "registry.yaml"
    registry = VersionRegistry(registry_path=registry_path)

    # Register version
    version = registry.register_version(
        version=args.version,
        environment=environment,
        agent_engine_id=args.agent_id,
        deployed_by=args.deployed_by,
        commit_sha=commit_sha,
        metadata={"git_branch": branch} if branch else {},
    )

    print(f"\nVersion registered successfully!")
    print(f"  Status: {version.status.value}")
    print(f"  Deployed at: {version.deployed_at.isoformat()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
