#!/usr/bin/env python3
"""Quality gate checks for CI/CD pipeline.

This script runs comprehensive quality checks and generates a summary report.
It should be run after all individual checks (ruff, mypy, bandit, pytest) pass.
"""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class QualityReport:
    """Quality check report."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    passed: bool = True
    checks: dict[str, dict] = field(default_factory=dict)
    summary: str = ""


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    return result.returncode, result.stdout, result.stderr


def check_coverage(min_coverage: float = 80.0) -> dict:
    """Check test coverage meets minimum threshold."""
    coverage_file = Path("coverage.xml")
    if not coverage_file.exists():
        return {
            "status": "skip",
            "message": "coverage.xml not found",
            "coverage": 0,
        }

    try:
        tree = ET.parse(coverage_file)  # noqa: S314
        root = tree.getroot()
        line_rate = float(root.get("line-rate", 0))
        coverage_pct = line_rate * 100

        passed = coverage_pct >= min_coverage
        return {
            "status": "pass" if passed else "fail",
            "message": f"Coverage: {coverage_pct:.1f}% (minimum: {min_coverage}%)",
            "coverage": coverage_pct,
            "threshold": min_coverage,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to parse coverage: {e}",
            "coverage": 0,
        }


def check_lint() -> dict:
    """Check ruff lint status."""
    exit_code, stdout, stderr = run_command(["ruff", "check", "src/", "--output-format=json"])

    if exit_code == 0:
        return {
            "status": "pass",
            "message": "No lint issues found",
            "issues": 0,
        }

    try:
        issues = json.loads(stdout) if stdout else []
        return {
            "status": "fail",
            "message": f"Found {len(issues)} lint issues",
            "issues": len(issues),
        }
    except json.JSONDecodeError:
        return {
            "status": "fail",
            "message": f"Lint check failed: {stderr}",
            "issues": -1,
        }


def check_type_safety() -> dict:
    """Check mypy type safety."""
    exit_code, stdout, stderr = run_command(
        ["mypy", "src/", "--ignore-missing-imports", "--no-error-summary"]
    )

    if exit_code == 0:
        return {
            "status": "pass",
            "message": "No type errors found",
            "errors": 0,
        }

    # Count error lines
    error_lines = [line for line in stdout.split("\n") if ": error:" in line]
    return {
        "status": "fail",
        "message": f"Found {len(error_lines)} type errors",
        "errors": len(error_lines),
    }


def check_security() -> dict:
    """Check bandit security scan results."""
    exit_code, stdout, stderr = run_command(
        ["bandit", "-r", "src/", "-f", "json", "-ll", "-ii"]
    )

    if exit_code == 0:
        return {
            "status": "pass",
            "message": "No security issues found",
            "issues": 0,
        }

    try:
        report = json.loads(stdout) if stdout else {}
        issues = report.get("results", [])
        high_severity = sum(1 for i in issues if i.get("issue_severity") == "HIGH")
        medium_severity = sum(1 for i in issues if i.get("issue_severity") == "MEDIUM")

        return {
            "status": "fail",
            "message": f"Found {len(issues)} security issues (High: {high_severity}, Medium: {medium_severity})",
            "issues": len(issues),
            "high": high_severity,
            "medium": medium_severity,
        }
    except json.JSONDecodeError:
        return {
            "status": "fail" if exit_code != 0 else "pass",
            "message": f"Security scan completed with code {exit_code}",
            "issues": 0,
        }


def check_complexity() -> dict:
    """Check code complexity using radon (if available)."""
    exit_code, stdout, stderr = run_command(
        ["radon", "cc", "src/", "-a", "-nc", "--total-average"]
    )

    if exit_code != 0 and "not found" in stderr.lower():
        return {
            "status": "skip",
            "message": "radon not installed",
            "average": 0,
        }

    # Parse average complexity
    for line in stdout.split("\n"):
        if "Average complexity:" in line:
            try:
                avg = float(line.split()[-1].strip("()"))
                passed = avg <= 10  # A or B complexity is acceptable
                return {
                    "status": "pass" if passed else "warn",
                    "message": f"Average complexity: {avg:.1f}",
                    "average": avg,
                }
            except (ValueError, IndexError):
                pass

    return {
        "status": "skip",
        "message": "Could not determine complexity",
        "average": 0,
    }


def generate_report(checks: dict[str, dict]) -> QualityReport:
    """Generate quality report from check results."""
    report = QualityReport()
    report.checks = checks

    # Determine overall pass/fail
    failed_checks = [
        name for name, result in checks.items()
        if result.get("status") == "fail"
    ]

    if failed_checks:
        report.passed = False
        report.summary = f"Quality gate FAILED: {', '.join(failed_checks)}"
    else:
        report.passed = True
        report.summary = "Quality gate PASSED"

    return report


def print_report(report: QualityReport) -> None:
    """Print quality report to console."""
    print("\n" + "=" * 60)
    print("QUALITY GATE REPORT")
    print("=" * 60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Status: {'PASSED' if report.passed else 'FAILED'}")
    print("-" * 60)

    for name, result in report.checks.items():
        status = result.get("status", "unknown")
        message = result.get("message", "")
        icon = {"pass": "[OK]", "fail": "[X]", "warn": "[!]", "skip": "[-]"}.get(
            status, "[?]"
        )
        print(f"{icon} {name}: {message}")

    print("-" * 60)
    print(report.summary)
    print("=" * 60 + "\n")


def main() -> int:
    """Run all quality checks and generate report."""
    print("Running quality gate checks...")

    checks: dict[str, dict] = {}

    # Run all checks
    print("  - Checking test coverage...")
    checks["coverage"] = check_coverage(min_coverage=80.0)

    print("  - Checking lint status...")
    checks["lint"] = check_lint()

    print("  - Checking type safety...")
    checks["type_safety"] = check_type_safety()

    print("  - Checking security...")
    checks["security"] = check_security()

    print("  - Checking complexity...")
    checks["complexity"] = check_complexity()

    # Generate and print report
    report = generate_report(checks)
    print_report(report)

    # Save report as JSON
    report_path = Path("quality_report.json")
    report_data = {
        "timestamp": report.timestamp,
        "passed": report.passed,
        "checks": report.checks,
        "summary": report.summary,
    }
    report_path.write_text(json.dumps(report_data, indent=2))
    print(f"Report saved to: {report_path}")

    # Return exit code based on pass/fail
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
