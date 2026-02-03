"""Tests for version registry functionality."""

import tempfile
from pathlib import Path

import pytest

from agent_engine.version import VersionRegistry
from agent_engine.version.models import Environment, VersionStatus


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


class TestVersionRegistry:
    """Tests for VersionRegistry class."""

    def test_registry_initialization(self, registry: VersionRegistry) -> None:
        """Test registry initializes with default structure."""
        assert registry.registry_path.exists()
        data = registry._load_registry()
        assert data["schema_version"] == "1.0"
        assert data["current"]["staging"] is None
        assert data["current"]["production"] is None
        assert data["versions"] == []

    def test_register_version_staging(self, registry: VersionRegistry) -> None:
        """Test registering a version in staging."""
        version = registry.register_version(
            version="v1.0.0",
            environment=Environment.STAGING,
            agent_engine_id="projects/test/locations/us/agentEngines/ae-123",
            deployed_by="test-user",
            commit_sha="abc123",
        )

        assert version.version == "v1.0.0"
        assert version.environment == Environment.STAGING
        assert version.status == VersionStatus.ACTIVE
        assert version.agent_engine_id == "projects/test/locations/us/agentEngines/ae-123"
        assert version.deployed_by == "test-user"
        assert version.commit_sha == "abc123"

    def test_register_version_updates_current(self, registry: VersionRegistry) -> None:
        """Test that registering a new version updates current."""
        registry.register_version(version="v1.0.0", environment=Environment.STAGING)
        registry.register_version(version="v1.1.0", environment=Environment.STAGING)

        current = registry.get_current_version(Environment.STAGING)
        assert current is not None
        assert current.version == "v1.1.0"

    def test_previous_version_becomes_inactive(self, registry: VersionRegistry) -> None:
        """Test that previous version is marked inactive."""
        registry.register_version(version="v1.0.0", environment=Environment.STAGING)
        registry.register_version(version="v1.1.0", environment=Environment.STAGING)

        v1 = registry.get_version("v1.0.0", Environment.STAGING)
        assert v1 is not None
        assert v1.status == VersionStatus.INACTIVE

    def test_get_current_version_none(self, registry: VersionRegistry) -> None:
        """Test getting current version when none exists."""
        current = registry.get_current_version(Environment.STAGING)
        assert current is None

    def test_get_current_version(self, registry: VersionRegistry) -> None:
        """Test getting current version."""
        registry.register_version(version="v1.0.0", environment=Environment.STAGING)
        current = registry.get_current_version(Environment.STAGING)
        assert current is not None
        assert current.version == "v1.0.0"

    def test_get_version(self, registry: VersionRegistry) -> None:
        """Test getting a specific version."""
        registry.register_version(version="v1.0.0", environment=Environment.STAGING)
        version = registry.get_version("v1.0.0", Environment.STAGING)
        assert version is not None
        assert version.version == "v1.0.0"

    def test_get_version_not_found(self, registry: VersionRegistry) -> None:
        """Test getting a version that doesn't exist."""
        version = registry.get_version("v1.0.0", Environment.STAGING)
        assert version is None

    def test_list_versions(self, registry: VersionRegistry) -> None:
        """Test listing versions."""
        registry.register_version(version="v1.0.0", environment=Environment.STAGING)
        registry.register_version(version="v1.1.0", environment=Environment.STAGING)
        registry.register_version(version="v1.0.0", environment=Environment.PRODUCTION)

        # List all
        all_versions = registry.list_versions(limit=10)
        assert len(all_versions) == 3

        # List staging only
        staging_versions = registry.list_versions(environment=Environment.STAGING)
        assert len(staging_versions) == 2

        # List with limit
        limited = registry.list_versions(limit=1)
        assert len(limited) == 1

    def test_list_versions_by_status(self, registry: VersionRegistry) -> None:
        """Test listing versions filtered by status."""
        registry.register_version(version="v1.0.0", environment=Environment.STAGING)
        registry.register_version(version="v1.1.0", environment=Environment.STAGING)

        active = registry.list_versions(status=VersionStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].version == "v1.1.0"

        inactive = registry.list_versions(status=VersionStatus.INACTIVE)
        assert len(inactive) == 1
        assert inactive[0].version == "v1.0.0"

    def test_get_previous_version(self, registry: VersionRegistry) -> None:
        """Test getting previous version for rollback."""
        registry.register_version(version="v1.0.0", environment=Environment.STAGING)
        registry.register_version(version="v1.1.0", environment=Environment.STAGING)

        previous = registry.get_previous_version(Environment.STAGING)
        assert previous is not None
        assert previous.version == "v1.0.0"

    def test_get_previous_version_none(self, registry: VersionRegistry) -> None:
        """Test getting previous version when only one exists."""
        registry.register_version(version="v1.0.0", environment=Environment.STAGING)

        previous = registry.get_previous_version(Environment.STAGING)
        assert previous is None

    def test_record_rollback(self, registry: VersionRegistry) -> None:
        """Test recording a rollback."""
        registry.register_version(version="v1.0.0", environment=Environment.STAGING)
        registry.register_version(version="v1.1.0", environment=Environment.STAGING)

        record = registry.record_rollback(
            from_version="v1.1.0",
            to_version="v1.0.0",
            environment=Environment.STAGING,
            executed_by="test-user",
            reason="Bug in v1.1.0",
        )

        assert record.from_version == "v1.1.0"
        assert record.to_version == "v1.0.0"
        assert record.reason == "Bug in v1.1.0"
        assert record.success is True

        # Check current is updated
        current = registry.get_current_version(Environment.STAGING)
        assert current is not None
        assert current.version == "v1.0.0"

    def test_get_rollback_history(self, registry: VersionRegistry) -> None:
        """Test getting rollback history."""
        registry.register_version(version="v1.0.0", environment=Environment.STAGING)
        registry.register_version(version="v1.1.0", environment=Environment.STAGING)
        registry.record_rollback(
            from_version="v1.1.0",
            to_version="v1.0.0",
            environment=Environment.STAGING,
            reason="Test rollback",
        )

        history = registry.get_rollback_history(Environment.STAGING)
        assert len(history) == 1
        assert history[0].from_version == "v1.1.0"
        assert history[0].to_version == "v1.0.0"

    def test_environments_are_separate(self, registry: VersionRegistry) -> None:
        """Test that staging and production are tracked separately."""
        registry.register_version(version="v1.0.0", environment=Environment.STAGING)
        registry.register_version(version="v0.9.0", environment=Environment.PRODUCTION)

        staging_current = registry.get_current_version(Environment.STAGING)
        prod_current = registry.get_current_version(Environment.PRODUCTION)

        assert staging_current is not None
        assert staging_current.version == "v1.0.0"
        assert prod_current is not None
        assert prod_current.version == "v0.9.0"

    def test_version_with_metadata(self, registry: VersionRegistry) -> None:
        """Test registering version with metadata."""
        version = registry.register_version(
            version="v1.0.0",
            environment=Environment.STAGING,
            metadata={"git_branch": "main", "build_number": "123"},
        )

        assert version.metadata["git_branch"] == "main"
        assert version.metadata["build_number"] == "123"
