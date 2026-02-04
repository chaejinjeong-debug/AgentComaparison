#!/usr/bin/env python3
"""Setup Cloud Monitoring alerts for AgentEngine.

This script creates alert policies and notification channels
based on the definitions in monitoring/alerts.yaml.

Usage:
    python scripts/infra/setup_alerts.py --project my-project
    python scripts/infra/setup_alerts.py --project my-project --dry-run
"""

import argparse
import json
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
    print(f"  {'[DRY RUN] ' if dry_run else ''}Running: gcloud {' '.join(cmd[:3])}...")

    if dry_run:
        return 0, "", ""

    result = subprocess.run(full_cmd, capture_output=True, text=True)  # noqa: S603
    return result.returncode, result.stdout, result.stderr


def create_notification_channel(
    project_id: str,
    channel_config: dict,
    dry_run: bool = False,
) -> str | None:
    """Create a notification channel."""
    channel_type = channel_config["type"]
    display_name = channel_config["display_name"]

    print(f"\n  Creating notification channel: {display_name}")

    # Check if exists
    code, out, err = run_gcloud(
        [
            "alpha",
            "monitoring",
            "channels",
            "list",
            f"--project={project_id}",
            f"--filter=displayName='{display_name}'",
            "--format=value(name)",
        ],
        dry_run=False,
    )

    if code == 0 and out.strip():
        channel_name = out.strip()
        print(f"    Channel already exists: {channel_name}")
        return channel_name

    # Build channel configuration
    channel_json = {
        "type": channel_type,
        "displayName": display_name,
        "labels": channel_config.get("labels", {}),
    }

    # Create channel
    cmd = [
        "alpha",
        "monitoring",
        "channels",
        "create",
        f"--project={project_id}",
        f"--channel-content={json.dumps(channel_json)}",
    ]

    code, out, err = run_gcloud(cmd, dry_run=dry_run)

    if code != 0 and not dry_run:
        print(f"    Error creating channel: {err}")
        return None

    if dry_run:
        return f"projects/{project_id}/notificationChannels/dry-run"

    # Extract channel name from output
    channel_name = out.strip() if out else None
    print(f"    Created: {channel_name}")
    return channel_name


def create_alert_policy(
    project_id: str,
    policy_config: dict,
    notification_channels: list[str],
    dry_run: bool = False,
) -> bool:
    """Create an alert policy."""
    display_name = policy_config["display_name"]

    print(f"\n  Creating alert policy: {display_name}")

    # Check if exists
    code, out, err = run_gcloud(
        [
            "alpha",
            "monitoring",
            "policies",
            "list",
            f"--project={project_id}",
            f"--filter=displayName='{display_name}'",
            "--format=value(name)",
        ],
        dry_run=False,
    )

    if code == 0 and out.strip():
        print(f"    Policy already exists: {out.strip()}")
        return True

    # Build policy configuration
    policy_json = {
        "displayName": display_name,
        "documentation": policy_config.get("documentation", {}),
        "conditions": [],
        "combiner": policy_config.get("combiner", "OR"),
        "notificationChannels": notification_channels,
    }

    # Add conditions
    for condition in policy_config.get("conditions", []):
        cond_obj = {
            "displayName": condition["display_name"],
        }

        if "condition_threshold" in condition:
            threshold = condition["condition_threshold"]
            cond_obj["conditionThreshold"] = {
                "filter": threshold.get("filter", ""),
                "comparison": threshold.get("comparison", "COMPARISON_GT"),
                "thresholdValue": threshold.get("threshold_value", 0),
                "duration": threshold.get("duration", "0s"),
            }

            if "aggregations" in threshold:
                cond_obj["conditionThreshold"]["aggregations"] = threshold["aggregations"]

        if "condition_absent" in condition:
            absent = condition["condition_absent"]
            cond_obj["conditionAbsent"] = {
                "filter": absent.get("filter", ""),
                "duration": absent.get("duration", "300s"),
            }

            if "aggregations" in absent:
                cond_obj["conditionAbsent"]["aggregations"] = absent["aggregations"]

        policy_json["conditions"].append(cond_obj)

    # Create policy
    cmd = [
        "alpha",
        "monitoring",
        "policies",
        "create",
        f"--project={project_id}",
        f"--policy={json.dumps(policy_json)}",
    ]

    code, out, err = run_gcloud(cmd, dry_run=dry_run)

    if code != 0 and not dry_run:
        print(f"    Error creating policy: {err}")
        return False

    return True


def create_uptime_check(
    project_id: str,
    check_config: dict,
    dry_run: bool = False,
) -> bool:
    """Create an uptime check."""
    display_name = check_config["display_name"]

    print(f"\n  Creating uptime check: {display_name}")

    # Check if exists
    code, out, err = run_gcloud(
        [
            "monitoring",
            "uptime",
            "list-configs",
            f"--project={project_id}",
            f"--filter=displayName='{display_name}'",
            "--format=value(name)",
        ],
        dry_run=False,
    )

    if code == 0 and out.strip():
        print(f"    Uptime check already exists: {out.strip()}")
        return True

    # Create uptime check
    http_check = check_config.get("http_check", {})
    resource = check_config.get("monitored_resource", {})

    cmd = [
        "monitoring",
        "uptime",
        "create",
        display_name,
        f"--project={project_id}",
        f"--resource-type={resource.get('type', 'uptime_url')}",
    ]

    if "labels" in resource:
        labels = resource["labels"]
        if "host" in labels:
            cmd.append(f"--resource-labels=host={labels['host']}")

    if http_check:
        if "path" in http_check:
            cmd.append(f"--path={http_check['path']}")
        if "port" in http_check:
            cmd.append(f"--port={http_check['port']}")

    code, out, err = run_gcloud(cmd, dry_run=dry_run)

    if code != 0 and not dry_run:
        print(f"    Error creating uptime check: {err}")
        return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup Cloud Monitoring alerts")
    parser.add_argument(
        "--project",
        "-p",
        required=True,
        help="GCP project ID",
    )
    parser.add_argument(
        "--config-file",
        default="monitoring/alerts.yaml",
        help="Path to alerts config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without making changes",
    )

    args = parser.parse_args()

    # Resolve config path
    config_path = Path(__file__).parent.parent.parent / args.config_file
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    print(f"Project: {args.project}")
    print(f"Config file: {config_path}")
    if args.dry_run:
        print("Mode: DRY RUN")

    # Load configuration
    config = load_yaml(config_path)

    # Create notification channels
    print("\n=== Setting up Notification Channels ===")
    channel_names = []
    for channel_config in config.get("notification_channels", []):
        channel_name = create_notification_channel(
            args.project,
            channel_config,
            args.dry_run,
        )
        if channel_name:
            channel_names.append(channel_name)

    # Create alert policies
    print("\n=== Setting up Alert Policies ===")
    for policy_config in config.get("alert_policies", []):
        if not create_alert_policy(
            args.project,
            policy_config,
            channel_names,
            args.dry_run,
        ):
            print(f"  Warning: Failed to create policy: {policy_config['display_name']}")

    # Create uptime checks
    print("\n=== Setting up Uptime Checks ===")
    for check_config in config.get("uptime_checks", []):
        create_uptime_check(args.project, check_config, args.dry_run)

    print("\n=== Alerts Setup Complete ===")
    print("\nTo view alerts, run:")
    print(f"  gcloud alpha monitoring policies list --project={args.project}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
