#!/usr/bin/env python3
"""Setup VPC Service Controls for AgentEngine.

This script creates access policies, access levels, and service perimeters
based on the definitions in infra/vpc-sc/*.yaml.

Usage:
    python scripts/infra/setup_vpc_sc.py --organization 123456789 --policy my-policy
    python scripts/infra/setup_vpc_sc.py --organization 123456789 --policy my-policy --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    """Load YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_gcloud(cmd: list[str], dry_run: bool = False) -> tuple[int, str, str]:
    """Run a gcloud command."""
    full_cmd = ["gcloud"] + cmd
    print(f"  {'[DRY RUN] ' if dry_run else ''}Running: {' '.join(full_cmd)}")

    if dry_run:
        return 0, "", ""

    result = subprocess.run(full_cmd, capture_output=True, text=True)  # noqa: S603
    return result.returncode, result.stdout, result.stderr


def create_access_policy(
    org_id: str,
    title: str,
    dry_run: bool = False,
) -> str | None:
    """Create or get access policy."""
    print(f"\n=== Access Policy: {title} ===")

    # List existing policies
    code, out, err = run_gcloud(
        [
            "access-context-manager",
            "policies",
            "list",
            f"--organization={org_id}",
            "--format=value(name,title)",
        ],
        dry_run=False,
    )

    if code == 0 and out:
        for line in out.strip().split("\n"):
            if line:
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1] == title:
                    policy_name = parts[0].split("/")[-1]
                    print(f"  Access policy already exists: {policy_name}")
                    return policy_name

    # Create new policy
    code, out, err = run_gcloud(
        [
            "access-context-manager",
            "policies",
            "create",
            f"--organization={org_id}",
            f"--title={title}",
        ],
        dry_run=dry_run,
    )

    if code != 0 and not dry_run:
        print(f"  Error creating access policy: {err}")
        return None

    if dry_run:
        return "dry-run-policy-id"

    # Extract policy name from output
    policy_name = out.strip().split("/")[-1] if out else None
    print(f"  Created access policy: {policy_name}")
    return policy_name


def create_access_level(
    policy_id: str,
    name: str,
    title: str,
    conditions: dict,
    dry_run: bool = False,
) -> bool:
    """Create an access level."""
    print(f"\n  Creating access level: {name}")

    # Build basic level specification
    basic_spec = []

    for condition in conditions.get("conditions", []):
        if "ip_subnetworks" in condition:
            for subnet in condition["ip_subnetworks"]:
                basic_spec.append(f"--basic-level-spec=ipSubnetworks={subnet}")

        if "members" in condition:
            for member in condition["members"]:
                basic_spec.append(f"--basic-level-spec=members={member}")

    cmd = [
        "access-context-manager",
        "levels",
        "create",
        name,
        f"--policy={policy_id}",
        f"--title={title}",
    ] + basic_spec

    code, out, err = run_gcloud(cmd, dry_run=dry_run)

    if code != 0 and not dry_run:
        if "already exists" in err:
            print(f"    Access level {name} already exists")
            return True
        print(f"    Error creating access level: {err}")
        return False

    return True


def create_perimeter(
    policy_id: str,
    name: str,
    title: str,
    config: dict,
    dry_run: bool = False,
) -> bool:
    """Create a service perimeter."""
    print(f"\n=== Creating Perimeter: {name} ===")

    # Build perimeter command
    cmd = [
        "access-context-manager",
        "perimeters",
        "create",
        name,
        f"--policy={policy_id}",
        f"--title={title}",
        "--perimeter-type=regular",
    ]

    # Add resources
    resources = config.get("resources", [])
    if resources:
        cmd.append(f"--resources={','.join(resources)}")

    # Add restricted services
    services = config.get("restricted_services", [])
    if services:
        cmd.append(f"--restricted-services={','.join(services)}")

    # Add access levels
    levels = config.get("access_levels", [])
    if levels:
        cmd.append(f"--access-levels={','.join(levels)}")

    code, out, err = run_gcloud(cmd, dry_run=dry_run)

    if code != 0 and not dry_run:
        if "already exists" in err:
            print(f"  Perimeter {name} already exists")
            return True
        print(f"  Error creating perimeter: {err}")
        return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup VPC Service Controls")
    parser.add_argument(
        "--organization",
        "-o",
        required=True,
        help="GCP organization ID",
    )
    parser.add_argument(
        "--policy",
        "-p",
        default="agent-engine-policy",
        help="Access policy name",
    )
    parser.add_argument(
        "--config-dir",
        default="infra/vpc-sc",
        help="Path to VPC-SC config directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without making changes",
    )

    args = parser.parse_args()

    # Resolve config path
    config_dir = Path(__file__).parent.parent.parent / args.config_dir
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        return 1

    print(f"Organization: {args.organization}")
    print(f"Config directory: {config_dir}")
    if args.dry_run:
        print("Mode: DRY RUN")

    # Load configuration
    config = load_yaml(config_dir / "perimeter.yaml")

    # Create or get access policy
    policy_id = create_access_policy(
        args.organization,
        config.get("access_policy", {}).get("title", "Agent Engine Access Policy"),
        args.dry_run,
    )

    if not policy_id:
        return 1

    # Create access levels
    print("\n=== Setting up Access Levels ===")
    for level in config.get("access_levels", []):
        if not create_access_level(
            policy_id,
            level["name"],
            level["title"],
            level.get("basic", {}),
            args.dry_run,
        ):
            return 1

    # Create service perimeter
    perimeter_config = config.get("service_perimeter", {})
    if perimeter_config:
        if not create_perimeter(
            policy_id,
            perimeter_config["name"],
            perimeter_config["title"],
            perimeter_config,
            args.dry_run,
        ):
            return 1

    print("\n=== VPC-SC Setup Complete ===")
    print("\nTo verify the setup, run:")
    print(f"  gcloud access-context-manager perimeters list --policy={policy_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
