"""Version registry management.

Provides functionality to track, register, and query deployed versions.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent_engine.version.models import Environment, RollbackRecord, Version, VersionStatus


class VersionRegistry:
    """Registry for tracking deployed agent versions.

    This class manages the version registry YAML file and provides
    methods for registering, querying, and updating versions.

    Attributes:
        registry_path: Path to the registry YAML file
    """

    def __init__(self, registry_path: str | Path | None = None) -> None:
        """Initialize the version registry.

        Args:
            registry_path: Path to the registry YAML file.
                         Defaults to versions/registry.yaml
        """
        if registry_path is None:
            registry_path = Path(__file__).parent.parent.parent.parent / "versions" / "registry.yaml"
        self.registry_path = Path(registry_path)
        self._ensure_registry_exists()

    def _ensure_registry_exists(self) -> None:
        """Ensure the registry file exists with default structure."""
        if not self.registry_path.exists():
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            default_registry = {
                "schema_version": "1.0",
                "current": {"staging": None, "production": None},
                "versions": [],
                "rollbacks": [],
            }
            self._save_registry(default_registry)

    def _load_registry(self) -> dict[str, Any]:
        """Load registry from YAML file."""
        with open(self.registry_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _save_registry(self, data: dict[str, Any]) -> None:
        """Save registry to YAML file."""
        with open(self.registry_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def register_version(
        self,
        version: str,
        environment: Environment,
        agent_engine_id: str | None = None,
        deployed_by: str | None = None,
        commit_sha: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Version:
        """Register a new version in the registry.

        Args:
            version: Semantic version string (e.g., v1.0.0)
            environment: Target environment (staging/production)
            agent_engine_id: Deployed Agent Engine resource ID
            deployed_by: User or service account name
            commit_sha: Git commit SHA
            metadata: Additional metadata

        Returns:
            Registered Version object
        """
        registry = self._load_registry()

        # Create version record
        version_obj = Version(
            version=version,
            environment=environment,
            agent_engine_id=agent_engine_id,
            deployed_at=datetime.now(timezone.utc),
            deployed_by=deployed_by or os.environ.get("USER", "unknown"),
            commit_sha=commit_sha,
            status=VersionStatus.ACTIVE,
            metadata=metadata or {},
        )

        # Add to versions list
        version_dict = version_obj.model_dump(mode="json")
        version_dict["deployed_at"] = version_obj.deployed_at.isoformat()
        registry["versions"].insert(0, version_dict)

        # Update current version for environment
        env_key = environment.value
        old_version = registry["current"].get(env_key)

        # Mark old version as inactive
        if old_version:
            for v in registry["versions"]:
                if v["version"] == old_version and v["environment"] == env_key:
                    v["status"] = VersionStatus.INACTIVE.value
                    break

        registry["current"][env_key] = version

        self._save_registry(registry)
        return version_obj

    def get_current_version(self, environment: Environment) -> Version | None:
        """Get the current active version for an environment.

        Args:
            environment: Target environment

        Returns:
            Current Version or None if no active version
        """
        registry = self._load_registry()
        current_version = registry["current"].get(environment.value)

        if not current_version:
            return None

        for v in registry["versions"]:
            if v["version"] == current_version and v["environment"] == environment.value:
                return Version(**v)

        return None

    def get_version(self, version: str, environment: Environment) -> Version | None:
        """Get a specific version.

        Args:
            version: Version string
            environment: Environment

        Returns:
            Version object or None if not found
        """
        registry = self._load_registry()

        for v in registry["versions"]:
            if v["version"] == version and v["environment"] == environment.value:
                return Version(**v)

        return None

    def list_versions(
        self,
        environment: Environment | None = None,
        status: VersionStatus | None = None,
        limit: int = 10,
    ) -> list[Version]:
        """List versions with optional filters.

        Args:
            environment: Filter by environment
            status: Filter by status
            limit: Maximum number of versions to return

        Returns:
            List of Version objects
        """
        registry = self._load_registry()
        versions = []

        for v in registry["versions"]:
            if environment and v["environment"] != environment.value:
                continue
            if status and v["status"] != status.value:
                continue
            versions.append(Version(**v))
            if len(versions) >= limit:
                break

        return versions

    def get_previous_version(self, environment: Environment) -> Version | None:
        """Get the previous active version (for rollback).

        Args:
            environment: Target environment

        Returns:
            Previous Version or None
        """
        registry = self._load_registry()
        current = registry["current"].get(environment.value)
        found_current = False

        for v in registry["versions"]:
            if v["environment"] != environment.value:
                continue

            if v["version"] == current:
                found_current = True
                continue

            if found_current and v["status"] in [
                VersionStatus.INACTIVE.value,
                VersionStatus.ACTIVE.value,
            ]:
                return Version(**v)

        return None

    def record_rollback(
        self,
        from_version: str,
        to_version: str,
        environment: Environment,
        executed_by: str | None = None,
        reason: str = "",
        success: bool = True,
    ) -> RollbackRecord:
        """Record a rollback operation.

        Args:
            from_version: Version rolled back from
            to_version: Version rolled back to
            environment: Environment
            executed_by: User who executed rollback
            reason: Reason for rollback
            success: Whether rollback succeeded

        Returns:
            RollbackRecord object
        """
        registry = self._load_registry()

        record = RollbackRecord(
            from_version=from_version,
            to_version=to_version,
            environment=environment,
            executed_at=datetime.now(timezone.utc),
            executed_by=executed_by or os.environ.get("USER", "unknown"),
            reason=reason,
            success=success,
        )

        record_dict = record.model_dump(mode="json")
        record_dict["executed_at"] = record.executed_at.isoformat()
        registry["rollbacks"].insert(0, record_dict)

        # Update version statuses
        for v in registry["versions"]:
            if v["environment"] != environment.value:
                continue
            if v["version"] == from_version:
                v["status"] = VersionStatus.ROLLBACK.value
            elif v["version"] == to_version:
                v["status"] = VersionStatus.ACTIVE.value

        # Update current version
        registry["current"][environment.value] = to_version

        self._save_registry(registry)
        return record

    def get_rollback_history(
        self,
        environment: Environment | None = None,
        limit: int = 10,
    ) -> list[RollbackRecord]:
        """Get rollback history.

        Args:
            environment: Filter by environment
            limit: Maximum records to return

        Returns:
            List of RollbackRecord objects
        """
        registry = self._load_registry()
        records = []

        for r in registry.get("rollbacks", []):
            if environment and r["environment"] != environment.value:
                continue
            records.append(RollbackRecord(**r))
            if len(records) >= limit:
                break

        return records
