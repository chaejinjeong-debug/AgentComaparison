#!/usr/bin/env python3
"""Setup IAM resources for AgentEngine.

This script creates service accounts and configures IAM bindings
based on the definitions in infra/iam/*.yaml.

Usage:
    python scripts/infra/setup_iam.py --project my-project
    python scripts/infra/setup_iam.py --project my-project --dry-run
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


def create_service_account(
    project_id: str,
    name: str,
    display_name: str,
    description: str,
    dry_run: bool = False,
) -> bool:
    """Create a service account."""
    email = f"{name}@{project_id}.iam.gserviceaccount.com"

    # Check if exists
    code, out, err = run_gcloud(
        [
            "iam",
            "service-accounts",
            "describe",
            email,
            f"--project={project_id}",
            "--format=value(email)",
        ],
        dry_run=False,  # Always check
    )

    if code == 0:
        print(f"  Service account {name} already exists")
        return True

    # Create service account
    code, out, err = run_gcloud(
        [
            "iam",
            "service-accounts",
            "create",
            name,
            f"--project={project_id}",
            f"--display-name={display_name}",
            f"--description={description}",
        ],
        dry_run=dry_run,
    )

    if code != 0 and not dry_run:
        print(f"  Error creating service account: {err}")
        return False

    print(f"  Created service account: {email}")
    return True


def bind_role(
    project_id: str,
    role: str,
    member: str,
    condition: dict | None = None,
    dry_run: bool = False,
) -> bool:
    """Bind a role to a member."""
    cmd = [
        "projects",
        "add-iam-policy-binding",
        project_id,
        f"--member={member}",
        f"--role={role}",
    ]

    if condition:
        cmd.extend(
            [
                f"--condition=title={condition['title']},expression={condition['expression']}",
            ]
        )

    code, out, err = run_gcloud(cmd, dry_run=dry_run)

    if code != 0 and not dry_run:
        print(f"  Error binding role: {err}")
        return False

    return True


def setup_service_accounts(config: dict, project_id: str, dry_run: bool = False) -> bool:
    """Setup all service accounts."""
    print("\n=== Setting up Service Accounts ===")

    for sa in config.get("service_accounts", []):
        name = sa["name"]
        print(f"\nCreating service account: {name}")

        success = create_service_account(
            project_id=project_id,
            name=name,
            display_name=sa["display_name"],
            description=sa["description"],
            dry_run=dry_run,
        )

        if not success:
            return False

        # Bind roles
        email = f"{name}@{project_id}.iam.gserviceaccount.com"
        member = f"serviceAccount:{email}"

        for role in sa.get("roles", []):
            print(f"  Binding role: {role}")
            bind_role(project_id, role, member, dry_run=dry_run)

    return True


def setup_custom_roles(config: dict, project_id: str, dry_run: bool = False) -> bool:
    """Setup custom IAM roles."""
    print("\n=== Setting up Custom Roles ===")

    for role in config.get("custom_roles", []):
        role_id = role["id"]
        print(f"\nCreating custom role: {role_id}")

        # Check if exists
        code, out, err = run_gcloud(
            [
                "iam",
                "roles",
                "describe",
                role_id,
                f"--project={project_id}",
            ],
            dry_run=False,
        )

        if code == 0:
            print(f"  Custom role {role_id} already exists, updating...")
            action = "update"
        else:
            action = "create"

        # Create/update role
        permissions = ",".join(role["permissions"])
        cmd = [
            "iam",
            "roles",
            action,
            role_id,
            f"--project={project_id}",
            f"--title={role['title']}",
            f"--description={role['description']}",
            f"--permissions={permissions}",
        ]

        code, out, err = run_gcloud(cmd, dry_run=dry_run)

        if code != 0 and not dry_run:
            print(f"  Error {action}ing custom role: {err}")
            return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup IAM resources")
    parser.add_argument(
        "--project",
        "-p",
        required=True,
        help="GCP project ID",
    )
    parser.add_argument(
        "--config-dir",
        default="infra/iam",
        help="Path to IAM config directory",
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

    print(f"Project: {args.project}")
    print(f"Config directory: {config_dir}")
    if args.dry_run:
        print("Mode: DRY RUN")

    # Load configurations
    sa_config = load_yaml(config_dir / "service-accounts.yaml")
    bindings_config = load_yaml(config_dir / "bindings.yaml")

    # Setup service accounts
    if not setup_service_accounts(sa_config, args.project, args.dry_run):
        return 1

    # Setup custom roles from bindings config
    if not setup_custom_roles(bindings_config, args.project, args.dry_run):
        return 1

    print("\n=== IAM Setup Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
