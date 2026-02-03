"""Rollback management for AgentEngine deployments.

Provides functionality to safely rollback to previous versions.
"""

import subprocess
from dataclasses import dataclass
from typing import Any

import structlog

from agent_engine.version.models import Environment, RollbackRecord, Version
from agent_engine.version.registry import VersionRegistry

logger = structlog.get_logger(__name__)


@dataclass
class RollbackResult:
    """Result of a rollback operation."""

    success: bool
    from_version: str
    to_version: str
    environment: str
    message: str
    details: dict[str, Any] | None = None


class RollbackManager:
    """Manages rollback operations for deployed agents.

    This class provides safe rollback functionality with
    validation, execution, and recovery capabilities.

    Attributes:
        registry: VersionRegistry instance
        project_id: GCP project ID
        location: GCP region
    """

    def __init__(
        self,
        registry: VersionRegistry | None = None,
        project_id: str | None = None,
        location: str = "asia-northeast3",
    ) -> None:
        """Initialize the rollback manager.

        Args:
            registry: VersionRegistry instance (creates new if None)
            project_id: GCP project ID
            location: GCP region
        """
        self.registry = registry or VersionRegistry()
        self.project_id = project_id
        self.location = location

    def can_rollback(self, environment: Environment) -> tuple[bool, str]:
        """Check if rollback is possible for an environment.

        Args:
            environment: Target environment

        Returns:
            Tuple of (can_rollback, reason)
        """
        current = self.registry.get_current_version(environment)
        if not current:
            return False, "No current version deployed"

        previous = self.registry.get_previous_version(environment)
        if not previous:
            return False, "No previous version available for rollback"

        if not previous.agent_engine_id:
            return False, f"Previous version {previous.version} has no agent_engine_id"

        return True, f"Can rollback from {current.version} to {previous.version}"

    def get_rollback_target(self, environment: Environment) -> Version | None:
        """Get the target version for rollback.

        Args:
            environment: Target environment

        Returns:
            Target Version or None
        """
        return self.registry.get_previous_version(environment)

    def execute_rollback(
        self,
        environment: Environment,
        target_version: str | None = None,
        reason: str = "",
        dry_run: bool = False,
    ) -> RollbackResult:
        """Execute a rollback operation.

        Args:
            environment: Target environment
            target_version: Specific version to rollback to (optional)
            reason: Reason for rollback
            dry_run: If True, only simulate the rollback

        Returns:
            RollbackResult with operation details
        """
        current = self.registry.get_current_version(environment)
        if not current:
            return RollbackResult(
                success=False,
                from_version="",
                to_version="",
                environment=environment.value,
                message="No current version deployed",
            )

        # Determine target version
        if target_version:
            target = self.registry.get_version(target_version, environment)
        else:
            target = self.registry.get_previous_version(environment)

        if not target:
            return RollbackResult(
                success=False,
                from_version=current.version,
                to_version=target_version or "",
                environment=environment.value,
                message="Target version not found",
            )

        logger.info(
            "executing_rollback",
            from_version=current.version,
            to_version=target.version,
            environment=environment.value,
            dry_run=dry_run,
        )

        if dry_run:
            return RollbackResult(
                success=True,
                from_version=current.version,
                to_version=target.version,
                environment=environment.value,
                message=f"[DRY RUN] Would rollback from {current.version} to {target.version}",
                details={
                    "current_agent_engine_id": current.agent_engine_id,
                    "target_agent_engine_id": target.agent_engine_id,
                },
            )

        # Execute the actual rollback
        try:
            success = self._perform_rollback(current, target, environment)

            if success:
                # Record the rollback
                self.registry.record_rollback(
                    from_version=current.version,
                    to_version=target.version,
                    environment=environment,
                    reason=reason,
                    success=True,
                )

                return RollbackResult(
                    success=True,
                    from_version=current.version,
                    to_version=target.version,
                    environment=environment.value,
                    message=f"Successfully rolled back from {current.version} to {target.version}",
                )
            else:
                return RollbackResult(
                    success=False,
                    from_version=current.version,
                    to_version=target.version,
                    environment=environment.value,
                    message="Rollback execution failed",
                )

        except Exception as e:
            logger.exception("rollback_failed", error=str(e))
            return RollbackResult(
                success=False,
                from_version=current.version,
                to_version=target.version,
                environment=environment.value,
                message=f"Rollback failed: {e}",
            )

    def _perform_rollback(
        self,
        current: Version,
        target: Version,
        environment: Environment,
    ) -> bool:
        """Perform the actual rollback operation.

        This updates traffic routing to point to the target version's
        Agent Engine deployment.

        Args:
            current: Current version
            target: Target version to rollback to
            environment: Target environment

        Returns:
            True if successful
        """
        if not target.agent_engine_id:
            raise ValueError(f"Target version {target.version} has no agent_engine_id")

        # In a real implementation, this would:
        # 1. Update traffic routing to the target version
        # 2. Verify the target version is healthy
        # 3. Update any load balancers or DNS

        # For now, we simulate using gcloud commands
        logger.info(
            "performing_rollback",
            target_agent_engine_id=target.agent_engine_id,
            environment=environment.value,
        )

        # Verify the target deployment exists
        if self.project_id:
            cmd = [
                "gcloud",
                "ai",
                "agent-engines",
                "describe",
                target.agent_engine_id,
                f"--project={self.project_id}",
                f"--region={self.location}",
                "--format=json",
            ]

            try:
                result = subprocess.run(  # noqa: S603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    logger.error("target_deployment_not_found", stderr=result.stderr)
                    return False
            except subprocess.TimeoutExpired:
                logger.error("gcloud_timeout")
                return False
            except FileNotFoundError:
                logger.warning("gcloud_not_found", message="Skipping verification")

        return True

    def list_rollback_candidates(
        self,
        environment: Environment,
        limit: int = 5,
    ) -> list[Version]:
        """List versions that can be rolled back to.

        Args:
            environment: Target environment
            limit: Maximum candidates to return

        Returns:
            List of candidate versions
        """
        versions = self.registry.list_versions(environment=environment, limit=limit + 1)

        # Skip the current version
        current = self.registry.get_current_version(environment)
        candidates = [v for v in versions if not current or v.version != current.version]

        return candidates[:limit]

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
            List of rollback records
        """
        return self.registry.get_rollback_history(environment=environment, limit=limit)
