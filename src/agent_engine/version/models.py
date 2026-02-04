"""Version management models.

Defines data models for version tracking and deployment information.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VersionStatus(str, Enum):
    """Status of a deployed version."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ROLLBACK = "rollback"
    DEPLOYING = "deploying"
    FAILED = "failed"


class Environment(str, Enum):
    """Deployment environment."""

    STAGING = "staging"
    PRODUCTION = "production"


class Version(BaseModel):
    """Version information model.

    Attributes:
        version: Semantic version string (e.g., v1.0.0)
        environment: Deployment environment
        agent_engine_id: Deployed Agent Engine resource ID
        deployed_at: Deployment timestamp
        deployed_by: User or service account that deployed
        commit_sha: Git commit SHA
        status: Version status
        metadata: Additional metadata
    """

    version: str = Field(..., description="Semantic version (e.g., v1.0.0)")
    environment: Environment = Field(..., description="Deployment environment")
    agent_engine_id: str | None = Field(default=None, description="Agent Engine resource ID")
    deployed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Deployment timestamp",
    )
    deployed_by: str = Field(default="unknown", description="Deployer identity")
    commit_sha: str | None = Field(default=None, description="Git commit SHA")
    status: VersionStatus = Field(default=VersionStatus.DEPLOYING, description="Version status")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def is_active(self) -> bool:
        """Check if version is active."""
        return self.status == VersionStatus.ACTIVE

    def mark_active(self) -> None:
        """Mark version as active."""
        self.status = VersionStatus.ACTIVE

    def mark_inactive(self) -> None:
        """Mark version as inactive."""
        self.status = VersionStatus.INACTIVE

    def mark_rollback(self) -> None:
        """Mark version as rolled back."""
        self.status = VersionStatus.ROLLBACK


class DeploymentInfo(BaseModel):
    """Deployment information for tracking.

    Attributes:
        version: Version being deployed
        source_version: Previous version (for rollbacks)
        target_environment: Target environment
        started_at: Deployment start time
        completed_at: Deployment completion time
        success: Whether deployment succeeded
        error_message: Error message if failed
    """

    version: str = Field(..., description="Version being deployed")
    source_version: str | None = Field(default=None, description="Previous version")
    target_environment: Environment = Field(..., description="Target environment")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Deployment start time",
    )
    completed_at: datetime | None = Field(default=None, description="Completion time")
    success: bool | None = Field(default=None, description="Deployment success")
    error_message: str | None = Field(default=None, description="Error message")

    def complete(self, success: bool, error_message: str | None = None) -> None:
        """Mark deployment as complete."""
        self.completed_at = datetime.now(UTC)
        self.success = success
        self.error_message = error_message


class RollbackRecord(BaseModel):
    """Record of a rollback operation.

    Attributes:
        from_version: Version being rolled back from
        to_version: Version being rolled back to
        environment: Environment where rollback occurred
        executed_at: Rollback execution time
        executed_by: User who executed rollback
        reason: Reason for rollback
        success: Whether rollback succeeded
    """

    from_version: str = Field(..., description="Version rolled back from")
    to_version: str = Field(..., description="Version rolled back to")
    environment: Environment = Field(..., description="Environment")
    executed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Execution time",
    )
    executed_by: str = Field(default="unknown", description="Executor identity")
    reason: str = Field(default="", description="Rollback reason")
    success: bool = Field(default=True, description="Rollback success")
