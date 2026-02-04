"""Tests for rollback management functionality."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_engine.version import RollbackManager, VersionRegistry
from agent_engine.version.models import Environment


@pytest.fixture
def temp_registry() -> Path:
    """Create a temporary registry file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")
        return Path(f.name)


@pytest.fixture
def registry(temp_registry: Path) -> VersionRegistry:
    """Create a version registry with temp file."""
    return VersionRegistry(registry_path=temp_registry)


@pytest.fixture
def rollback_manager(registry: VersionRegistry) -> RollbackManager:
    """Create a rollback manager."""
    return RollbackManager(
        registry=registry,
        project_id="test-project",
        location="asia-northeast3",
    )


class TestRollbackManager:
    """Tests for RollbackManager class."""

    def test_can_rollback_no_current_version(
        self,
        rollback_manager: RollbackManager,
    ) -> None:
        """Test can_rollback when no version is deployed."""
        can_rollback, reason = rollback_manager.can_rollback(Environment.STAGING)
        assert can_rollback is False
        assert "No current version" in reason

    def test_can_rollback_no_previous_version(
        self,
        rollback_manager: RollbackManager,
        registry: VersionRegistry,
    ) -> None:
        """Test can_rollback when only one version exists."""
        registry.register_version(
            version="v1.0.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-123",
        )

        can_rollback, reason = rollback_manager.can_rollback(Environment.STAGING)
        assert can_rollback is False
        assert "No previous version" in reason

    def test_can_rollback_success(
        self,
        rollback_manager: RollbackManager,
        registry: VersionRegistry,
    ) -> None:
        """Test can_rollback when rollback is possible."""
        registry.register_version(
            version="v1.0.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-123",
        )
        registry.register_version(
            version="v1.1.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-124",
        )

        can_rollback, reason = rollback_manager.can_rollback(Environment.STAGING)
        assert can_rollback is True
        assert "v1.1.0 to v1.0.0" in reason

    def test_get_rollback_target(
        self,
        rollback_manager: RollbackManager,
        registry: VersionRegistry,
    ) -> None:
        """Test getting rollback target version."""
        registry.register_version(
            version="v1.0.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-123",
        )
        registry.register_version(
            version="v1.1.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-124",
        )

        target = rollback_manager.get_rollback_target(Environment.STAGING)
        assert target is not None
        assert target.version == "v1.0.0"

    def test_execute_rollback_dry_run(
        self,
        rollback_manager: RollbackManager,
        registry: VersionRegistry,
    ) -> None:
        """Test dry run rollback."""
        registry.register_version(
            version="v1.0.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-123",
        )
        registry.register_version(
            version="v1.1.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-124",
        )

        result = rollback_manager.execute_rollback(
            environment=Environment.STAGING,
            reason="Test rollback",
            dry_run=True,
        )

        assert result.success is True
        assert "[DRY RUN]" in result.message
        assert result.from_version == "v1.1.0"
        assert result.to_version == "v1.0.0"

        # Verify no changes were made
        current = registry.get_current_version(Environment.STAGING)
        assert current is not None
        assert current.version == "v1.1.0"

    def test_execute_rollback_no_current(
        self,
        rollback_manager: RollbackManager,
    ) -> None:
        """Test rollback fails when no current version."""
        result = rollback_manager.execute_rollback(
            environment=Environment.STAGING,
            dry_run=False,
        )

        assert result.success is False
        assert "No current version" in result.message

    def test_execute_rollback_target_not_found(
        self,
        rollback_manager: RollbackManager,
        registry: VersionRegistry,
    ) -> None:
        """Test rollback fails when target not found."""
        registry.register_version(
            version="v1.0.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-123",
        )

        result = rollback_manager.execute_rollback(
            environment=Environment.STAGING,
            target_version="v0.9.0",
            dry_run=False,
        )

        assert result.success is False
        assert "not found" in result.message

    @patch("subprocess.run")
    def test_execute_rollback_success(
        self,
        mock_run: MagicMock,
        rollback_manager: RollbackManager,
        registry: VersionRegistry,
    ) -> None:
        """Test successful rollback execution."""
        # Mock gcloud command to succeed
        mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")

        registry.register_version(
            version="v1.0.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-123",
        )
        registry.register_version(
            version="v1.1.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-124",
        )

        result = rollback_manager.execute_rollback(
            environment=Environment.STAGING,
            reason="Bug found",
            dry_run=False,
        )

        assert result.success is True
        assert "Successfully rolled back" in result.message

        # Verify changes
        current = registry.get_current_version(Environment.STAGING)
        assert current is not None
        assert current.version == "v1.0.0"

    @patch("subprocess.run")
    def test_execute_rollback_to_specific_version(
        self,
        mock_run: MagicMock,
        rollback_manager: RollbackManager,
        registry: VersionRegistry,
    ) -> None:
        """Test rollback to a specific version."""
        # Mock gcloud command to succeed
        mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")

        registry.register_version(
            version="v1.0.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-123",
        )
        registry.register_version(
            version="v1.1.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-124",
        )
        registry.register_version(
            version="v1.2.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-125",
        )

        result = rollback_manager.execute_rollback(
            environment=Environment.STAGING,
            target_version="v1.0.0",
            dry_run=False,
        )

        assert result.success is True
        assert result.to_version == "v1.0.0"

    def test_list_rollback_candidates(
        self,
        rollback_manager: RollbackManager,
        registry: VersionRegistry,
    ) -> None:
        """Test listing rollback candidates."""
        registry.register_version(
            version="v1.0.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-123",
        )
        registry.register_version(
            version="v1.1.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-124",
        )
        registry.register_version(
            version="v1.2.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-125",
        )

        candidates = rollback_manager.list_rollback_candidates(Environment.STAGING)

        # Should not include current version (v1.2.0)
        assert len(candidates) == 2
        versions = [c.version for c in candidates]
        assert "v1.2.0" not in versions
        assert "v1.1.0" in versions
        assert "v1.0.0" in versions

    @patch("subprocess.run")
    def test_get_rollback_history(
        self,
        mock_run: MagicMock,
        rollback_manager: RollbackManager,
        registry: VersionRegistry,
    ) -> None:
        """Test getting rollback history."""
        # Mock gcloud command to succeed
        mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")

        registry.register_version(
            version="v1.0.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-123",
        )
        registry.register_version(
            version="v1.1.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-124",
        )
        rollback_manager.execute_rollback(
            environment=Environment.STAGING,
            reason="First rollback",
            dry_run=False,
        )

        history = rollback_manager.get_rollback_history(Environment.STAGING)

        assert len(history) == 1
        assert history[0].from_version == "v1.1.0"
        assert history[0].to_version == "v1.0.0"
        assert history[0].reason == "First rollback"

    @patch("subprocess.run")
    def test_rollback_records_history(
        self,
        mock_run: MagicMock,
        rollback_manager: RollbackManager,
        registry: VersionRegistry,
    ) -> None:
        """Test that rollback creates history record."""
        # Mock gcloud command to succeed
        mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")

        registry.register_version(
            version="v1.0.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-123",
        )
        registry.register_version(
            version="v1.1.0",
            environment=Environment.STAGING,
            agent_engine_id="ae-124",
        )

        rollback_manager.execute_rollback(
            environment=Environment.STAGING,
            reason="Test rollback",
            dry_run=False,
        )

        history = registry.get_rollback_history(Environment.STAGING)
        assert len(history) == 1
        assert history[0].success is True
