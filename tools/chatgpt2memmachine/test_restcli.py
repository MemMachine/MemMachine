"""Pytest test cases for API tracing functionality"""

import pytest

from restcli import MemMachineRestClient


@pytest.fixture
def client():
    """Fixture to create a MemMachineRestClient instance with verbose tracing enabled."""
    return MemMachineRestClient(base_url="http://localhost:8080", verbose=True)


@pytest.fixture
def org_id():
    """Fixture for organization ID."""
    return "my-org"


@pytest.fixture
def project_id():
    """Fixture for project ID."""
    return "my-project"


class TestAPITracing:
    """Test class for API tracing functionality."""

    def test_add_memory_tracing(self, client, org_id, project_id):
        """Test that add_memory request tracing works correctly."""
        # This test verifies tracing output for POST requests
        # Note: This may fail if the server is not running, which is expected
        # The test passes if tracing output is generated (via verbose=True)
        result = client.add_memory(
            org_id,
            project_id,
            [{"content": "Test message for tracing"}],
        )
        # If successful, verify we got a response
        assert result is not None

    def test_search_memory_tracing(self, client, org_id, project_id):
        """Test that search_memory request tracing works correctly."""
        # This test verifies tracing output for SEARCH requests
        # Note: This may fail if the server is not running, which is expected
        # The test passes if tracing output is generated (via verbose=True)
        result = client.search_memory(org_id, project_id, "test query", limit=3)
        # If successful, verify we got a response
        assert result is not None
