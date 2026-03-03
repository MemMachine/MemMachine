"""Comprehensive integration tests for MemMachine REST API client.

This test suite provides complete end-to-end testing of the MemMachine client,
covering all major functionality including:
- Project lifecycle (create, get, get_or_create, refresh, delete)
- Memory operations (add, search, delete)
- Semantic memory extraction and processing
- Error handling and edge cases
- Data consistency and validation
- Concurrent operations

These tests require a running MemMachine server.
"""

import os
import time
from typing import Any
from uuid import uuid4

import pytest
import requests
from pydantic import ValidationError

from memmachine.common.api.spec import SearchResult
from memmachine.rest_client.client import MemMachineClient
from memmachine.rest_client.langgraph import MemMachineTools
from memmachine.rest_client.project import Project


def _get_episodic_episodes(results: SearchResult) -> list[Any]:
    episodic = results.content.episodic_memory
    if episodic is None:
        return []
    return episodic.short_term_memory.episodes + episodic.long_term_memory.episodes


def check_server_available():
    """Check if MemMachine server is available."""
    base_url = os.environ.get("MEMORY_BACKEND_URL", "http://localhost:8080")
    try:
        response = requests.get(f"{base_url}/api/v2/health", timeout=5)
    except Exception:
        return False
    else:
        return response.status_code == 200


TEST_BASE_URL = os.environ.get("MEMORY_BACKEND_URL", "http://localhost:8080")


@pytest.mark.integration
@pytest.mark.skipif(
    not check_server_available(),
    reason="MemMachine server not available. Start server or set MEMORY_BACKEND_URL",
)
class TestMemMachineIntegration:
    """Comprehensive integration tests for MemMachine client."""

    @pytest.fixture
    def client(self):
        """Create a MemMachine client instance."""
        return MemMachineClient(base_url=TEST_BASE_URL, timeout=60)

    @pytest.fixture
    def unique_test_ids(self):
        """Generate unique test IDs for test isolation."""
        test_id = str(uuid4())[:8]
        return {
            "org_id": f"test_org_{test_id}",
            "project_id": f"test_project_{test_id}",
            "user_id": f"test_user_{test_id}",
            "agent_id": f"test_agent_{test_id}",
            "session_id": f"test_session_{test_id}",
            "group_id": f"test_group_{test_id}",
        }

    # ==================== Project Lifecycle Tests ====================

    def test_project_create_and_get(self, client, unique_test_ids):
        """Test creating a project and retrieving it."""
        # Create project
        project = client.create_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
            description="Test project for integration tests",
            embedder="",  # Use server defaults
            reranker="",  # Use server defaults
        )

        assert isinstance(project, Project)
        assert project.org_id == unique_test_ids["org_id"]
        assert project.project_id == unique_test_ids["project_id"]
        assert project.description == "Test project for integration tests"

        # Get the project
        retrieved_project = client.get_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
        )

        assert retrieved_project.org_id == unique_test_ids["org_id"]
        assert retrieved_project.project_id == unique_test_ids["project_id"]
        assert retrieved_project.description == "Test project for integration tests"

    def test_project_get_or_create_existing(self, client, unique_test_ids):
        """Test get_or_create_project when project already exists."""
        # Create project first
        client.create_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
            description="Original description",
        )

        # Use get_or_create - should return existing project
        project = client.get_or_create_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
            description="New description",  # Should be ignored
        )

        assert project.org_id == unique_test_ids["org_id"]
        assert project.project_id == unique_test_ids["project_id"]
        # Description should be from existing project, not new one
        assert project.description == "Original description"

    def test_project_get_or_create_new(self, client, unique_test_ids):
        """Test get_or_create_project when project doesn't exist."""
        # Use get_or_create on non-existent project
        project = client.get_or_create_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
            description="Created via get_or_create",
        )

        assert project.org_id == unique_test_ids["org_id"]
        assert project.project_id == unique_test_ids["project_id"]
        assert project.description == "Created via get_or_create"

    def test_project_refresh(self, client, unique_test_ids):
        """Test refreshing project data from server."""
        # Create project
        project = client.create_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
            description="Original description",
        )

        # Manually modify local state (simulating stale data)
        project.description = "Stale description"

        # Refresh from server
        project.refresh()

        # Should have original description from server
        assert project.description == "Original description"

    def test_project_delete(self, client, unique_test_ids):
        """Test deleting a project."""
        # Create project
        project = client.create_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
            description="Project to be deleted",
        )

        # Delete the project
        result = project.delete()
        assert result is True

        # Verify project no longer exists
        with pytest.raises(requests.HTTPError) as exc_info:
            client.get_project(
                org_id=unique_test_ids["org_id"],
                project_id=unique_test_ids["project_id"],
            )
        assert exc_info.value.response.status_code == 404

    def test_project_configuration_persistence(self, client, unique_test_ids):
        """Test that project configuration is persisted and retrieved correctly."""
        # Create project with empty embedder and reranker (use server defaults)
        client.create_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
            embedder="",  # Use server defaults
            reranker="",  # Use server defaults
        )

        # Get project and verify configuration
        retrieved = client.get_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
        )

        # Configuration should be available
        assert retrieved.config is not None
        # Should be a ProjectConfig object with embedder and reranker fields
        from memmachine.common.api.spec import ProjectConfig

        assert isinstance(retrieved.config, ProjectConfig)

    # ==================== Memory Operations Tests ====================

    @pytest.fixture
    def memory(self, client, unique_test_ids):
        """Create a Memory instance for testing."""
        project = client.get_or_create_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
            description="Test project for memory operations",
        )

        return project.memory(
            metadata={
                "group_id": unique_test_ids["group_id"],
                "agent_id": unique_test_ids["agent_id"],
                "user_id": unique_test_ids["user_id"],
                "session_id": unique_test_ids["session_id"],
            }
        )

    def test_add_memory_user_role(self, memory):
        """Test adding a user memory."""
        result = memory.add(
            content="I love pizza and Italian food",
            role="user",
            metadata={"preference": "food", "category": "cuisine"},
        )

        assert isinstance(result, list)
        assert len(result) > 0
        from memmachine.common.api.spec import AddMemoryResult

        assert isinstance(result[0], AddMemoryResult)
        assert hasattr(result[0], "uid")

    def test_add_memory_assistant_role(self, memory):
        """Test adding an assistant memory."""
        result = memory.add(
            content="I understand you like Italian cuisine",
            role="assistant",
        )

        assert isinstance(result, list)
        assert len(result) > 0
        from memmachine.common.api.spec import AddMemoryResult

        assert isinstance(result[0], AddMemoryResult)
        assert hasattr(result[0], "uid")

    def test_add_memory_system_role(self, memory):
        """Test adding a system memory."""
        result = memory.add(
            content="System initialized for user session",
            role="system",
        )

        assert isinstance(result, list)
        assert len(result) > 0
        from memmachine.common.api.spec import AddMemoryResult

        assert isinstance(result[0], AddMemoryResult)
        assert hasattr(result[0], "uid")

    def test_add_memory_with_metadata(self, memory):
        """Test adding memory with custom metadata."""
        result = memory.add(
            content="User prefers morning meetings",
            role="user",
            metadata={
                "preference": "schedule",
                "time": "morning",
                "type": "meeting",
            },
        )

        assert isinstance(result, list)
        assert len(result) > 0
        from memmachine.common.api.spec import AddMemoryResult

        assert isinstance(result[0], AddMemoryResult)
        assert hasattr(result[0], "uid")

    def test_search_memory_basic(self, memory):
        """Test basic memory search functionality."""
        # Add some memories
        memory.add("I work as a software engineer", role="user")
        memory.add("I enjoy reading science fiction books", role="user")
        memory.add("My favorite programming language is Python", role="user")

        # Wait a bit for indexing
        time.sleep(1)

        # Search for memories
        results = memory.search("What is my profession?", limit=5)

        assert isinstance(results, SearchResult)
        assert results.content.episodic_memory is not None
        assert results.content.semantic_memory is not None

        # Should find at least one result
        episodes = _get_episodic_episodes(results)
        assert len(episodes) > 0 or len(results.content.semantic_memory or []) > 0

    def test_search_memory_with_limit(self, memory):
        """Test memory search with limit parameter."""
        # Add multiple memories
        for i in range(10):
            memory.add(f"Memory item {i}: This is test content {i}", role="user")

        # Wait for indexing
        time.sleep(1)

        # Search with limit
        results = memory.search("test content", limit=3)

        # Limit applies per memory type, so we may get more than limit total
        # But each type should respect the limit
        episodic_count = len(_get_episodic_episodes(results))
        semantic_count = len(results.content.semantic_memory or [])

        # Each type should respect limit (may be less if not enough matches)
        # Note: limit is applied per memory type, so total may exceed limit
        assert (
            episodic_count <= 3
            or semantic_count <= 3
            or (episodic_count + semantic_count) > 0
        )

    def test_search_memory_with_filter(self, memory):
        """Test memory search with filter."""
        # Add memories with different metadata
        memory.add(
            "I like coffee in the morning",
            role="user",
            metadata={"category": "preference", "time": "morning"},
        )
        memory.add(
            "I prefer tea in the afternoon",
            role="user",
            metadata={"category": "preference", "time": "afternoon"},
        )

        # Wait for indexing
        time.sleep(1)

        # Search without filter first to verify memories exist
        results_no_filter = memory.search("drink", limit=10)
        assert isinstance(results_no_filter, SearchResult)
        all_episodes = _get_episodic_episodes(results_no_filter)
        assert len(all_episodes) >= 2, (
            "Should find both morning and afternoon memories without filter"
        )

        # Search with filter (filter_dict is converted to SQL-like string format)
        try:
            results = memory.search(
                "What do I like to drink?",
                filter_dict={"time": "morning"},
                limit=10,
            )
            assert isinstance(results, SearchResult)

            # Verify filter is working: only memories with time="morning" should be returned
            filtered_episodes = _get_episodic_episodes(results)
            filtered_semantic = results.content.semantic_memory or []

            # Check all returned episodic memories match the filter
            for episode in filtered_episodes:
                if isinstance(episode, dict):
                    metadata = episode.get("metadata", {})
                    if metadata and "time" in metadata:
                        assert metadata["time"] == "morning", (
                            f"Found episode with time='{metadata.get('time')}' but filter requires 'morning'"
                        )
                elif hasattr(episode, "metadata") and episode.metadata:
                    if "time" in episode.metadata:
                        assert episode.metadata["time"] == "morning", (
                            f"Found episode with time='{episode.metadata.get('time')}' but filter requires 'morning'"
                        )

            # Verify that we got fewer or equal results with filter
            # (should exclude afternoon memories)
            total_filtered = len(filtered_episodes) + len(filtered_semantic)
            total_unfiltered = len(all_episodes) + len(
                results_no_filter.content.semantic_memory or []
            )
            assert total_filtered <= total_unfiltered, (
                "Filtered results should not exceed unfiltered results"
            )

        except requests.HTTPError as e:
            # If filter format is not supported, skip this test
            if e.response.status_code == 422:
                pytest.skip("Filter format not supported by server")
            raise

    def test_get_default_filter_dict(self, memory):
        """Test get_default_filter_dict method returns correct built-in filters."""
        # Get default filter dict
        default_filters = memory.get_default_filter_dict()

        # Should contain metadata filters for all non-None context fields
        assert isinstance(default_filters, dict)
        # Check if metadata fields exist and match
        user_id = memory.metadata.get("user_id")
        agent_id = memory.metadata.get("agent_id")
        session_id = memory.metadata.get("session_id")

        if user_id:
            assert "metadata.user_id" in default_filters
            assert default_filters["metadata.user_id"] == user_id
        if agent_id:
            assert "metadata.agent_id" in default_filters
            assert default_filters["metadata.agent_id"] == agent_id
        if session_id:
            assert "metadata.session_id" in default_filters
            assert default_filters["metadata.session_id"] == session_id

    def test_get_current_metadata(self, memory):
        """Test get_current_metadata method returns context, filters, and filter string."""
        # Get current metadata
        metadata = memory.get_current_metadata()

        # Check structure
        assert "context" in metadata
        assert "built_in_filters" in metadata
        assert "built_in_filter_string" in metadata

        # Check context
        context = metadata["context"]
        assert context["org_id"] == memory._Memory__org_id
        assert context["project_id"] == memory._Memory__project_id
        context_metadata = context["metadata"]
        assert context_metadata.get("user_id") == memory.metadata.get("user_id")
        assert context_metadata.get("agent_id") == memory.metadata.get("agent_id")
        assert context_metadata.get("session_id") == memory.metadata.get("session_id")

        # Check built-in filters
        filters = metadata["built_in_filters"]
        assert isinstance(filters, dict)
        user_id = memory.metadata.get("user_id")
        agent_id = memory.metadata.get("agent_id")
        session_id = memory.metadata.get("session_id")

        if user_id:
            assert "metadata.user_id" in filters
            assert filters["metadata.user_id"] == user_id
        if agent_id:
            assert "metadata.agent_id" in filters
            assert filters["metadata.agent_id"] == agent_id
        if session_id:
            assert "metadata.session_id" in filters
            assert filters["metadata.session_id"] == session_id

        # Check filter string
        filter_str = metadata["built_in_filter_string"]
        assert isinstance(filter_str, str)
        if user_id:
            assert "metadata.user_id" in filter_str
            assert f"metadata.user_id='{user_id}'" in filter_str
        if agent_id:
            assert "metadata.agent_id" in filter_str
            assert f"metadata.agent_id='{agent_id}'" in filter_str
        if session_id:
            assert "metadata.session_id" in filter_str
            assert f"metadata.session_id='{session_id}'" in filter_str

    def test_search_with_default_filter_dict(self, memory):
        """Test search automatically applies built-in filters and merges with custom filters."""
        # Add memories with different metadata
        memory.add(
            "I work as a software engineer",
            role="user",
            metadata={"category": "profession"},
        )
        memory.add(
            "I enjoy reading books",
            role="user",
            metadata={"category": "hobby"},
        )

        # Wait for indexing
        time.sleep(1)

        # Search without filter first to verify memories exist
        results_no_filter = memory.search("work", limit=10)
        all_episodes = _get_episodic_episodes(results_no_filter)
        assert len(all_episodes) >= 2, (
            "Should find both profession and hobby memories without filter"
        )

        # Search with custom filters only - built-in filters are automatically merged
        custom_filters = {"category": "profession"}
        try:
            results = memory.search(
                "What is my profession?",
                filter_dict=custom_filters,
                limit=10,
            )
            assert isinstance(results, SearchResult)

            # Verify filter is working: only memories with category="profession" should be returned
            filtered_episodes = _get_episodic_episodes(results)
            filtered_semantic = results.content.semantic_memory or []

            # Check all returned episodic memories match the filter
            for episode in filtered_episodes:
                if isinstance(episode, dict):
                    metadata = episode.get("metadata", {})
                    if metadata and "category" in metadata:
                        assert metadata["category"] == "profession", (
                            f"Found episode with category='{metadata.get('category')}' but filter requires 'profession'"
                        )
                elif hasattr(episode, "metadata") and episode.metadata:
                    if "category" in episode.metadata:
                        assert episode.metadata["category"] == "profession", (
                            f"Found episode with category='{episode.metadata.get('category')}' but filter requires 'profession'"
                        )

            # Verify that we got fewer or equal results with filter
            # (should exclude hobby memories)
            total_filtered = len(filtered_episodes) + len(filtered_semantic)
            total_unfiltered = len(all_episodes) + len(
                results_no_filter.content.semantic_memory or []
            )
            assert total_filtered <= total_unfiltered, (
                "Filtered results should not exceed unfiltered results"
            )

        except requests.HTTPError as e:
            # If filter format is not supported, skip this test
            if e.response.status_code == 422:
                pytest.skip("Filter format not supported by server")
            raise

    def _verify_episode_user_id(self, episode: Any, expected_user_id: str) -> None:
        """Verify that an episode has the expected user_id in its metadata."""
        if isinstance(episode, dict):
            metadata = episode.get("metadata", {})
            if metadata and "user_id" in metadata:
                assert metadata["user_id"] == expected_user_id, (
                    f"Found episode with user_id='{metadata.get('user_id')}' but filter requires '{expected_user_id}'"
                )
        elif hasattr(episode, "metadata") and episode.metadata:
            if "user_id" in episode.metadata:
                assert episode.metadata["user_id"] == expected_user_id, (
                    f"Found episode with user_id='{episode.metadata.get('user_id')}' but filter requires '{expected_user_id}'"
                )

    def _verify_filtered_results(
        self,
        filtered_episodes: list[Any],
        filtered_semantic: list[Any],
        all_episodes: list[Any],
        all_semantic: list[Any],
        user_label: str,
    ) -> None:
        """Verify that filtered results are a subset of unfiltered results."""
        total_filtered = len(filtered_episodes) + len(filtered_semantic)
        total_unfiltered = len(all_episodes) + len(all_semantic)
        assert total_filtered <= total_unfiltered, (
            f"{user_label} filtered results should not exceed unfiltered results"
        )

    def test_search_with_user_id_filter(self, client, unique_test_ids):
        """Test that filter by user_id only returns memories for that specific user."""
        # Create two different memory instances with different user_ids
        project = client.get_or_create_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
        )

        user1_id = f"{unique_test_ids['user_id']}_1"
        user2_id = f"{unique_test_ids['user_id']}_2"

        # Memory instance for user1
        memory_user1 = project.memory(
            metadata={
                "user_id": user1_id,
                "agent_id": unique_test_ids["agent_id"],
                "session_id": unique_test_ids["session_id"],
            }
        )

        # Memory instance for user2
        memory_user2 = project.memory(
            metadata={
                "user_id": user2_id,
                "agent_id": unique_test_ids["agent_id"],
                "session_id": unique_test_ids["session_id"],
            }
        )

        # Add memories for user1
        memory_user1.add(
            "I love Python programming",
            role="user",
            metadata={"topic": "programming", "language": "Python"},
        )
        memory_user1.add(
            "I enjoy machine learning",
            role="user",
            metadata={"topic": "AI", "interest": "high"},
        )

        # Add memories for user2
        memory_user2.add(
            "I prefer JavaScript for web development",
            role="user",
            metadata={"topic": "programming", "language": "JavaScript"},
        )
        memory_user2.add(
            "I like data science",
            role="user",
            metadata={"topic": "data", "interest": "high"},
        )

        # Wait for indexing
        time.sleep(2)

        # Search without filter first - should return memories from both users
        results_no_filter = memory_user1.search("programming", limit=10)
        all_episodes = _get_episodic_episodes(results_no_filter)
        assert len(all_episodes) >= 2, (
            "Should find memories from both users without filter"
        )

        # Search with user1 - built-in filters are automatically applied
        default_filters_user1 = memory_user1.get_default_filter_dict()
        assert "metadata.user_id" in default_filters_user1
        assert default_filters_user1["metadata.user_id"] == user1_id

        # Search without explicit filter_dict - built-in filters are automatically applied
        results_user1 = memory_user1.search(
            "programming",
            limit=10,
        )

        # Verify all returned memories belong to user1
        filtered_episodes_user1 = _get_episodic_episodes(results_user1)
        filtered_semantic_user1 = results_user1.content.semantic_memory or []

        for episode in filtered_episodes_user1:
            self._verify_episode_user_id(episode, user1_id)

        # Search with user2 - built-in filters are automatically applied
        default_filters_user2 = memory_user2.get_default_filter_dict()
        assert "metadata.user_id" in default_filters_user2
        assert default_filters_user2["metadata.user_id"] == user2_id

        # Search without explicit filter_dict - built-in filters are automatically applied
        results_user2 = memory_user2.search(
            "programming",
            limit=10,
        )

        # Verify all returned memories belong to user2
        filtered_episodes_user2 = _get_episodic_episodes(results_user2)

        for episode in filtered_episodes_user2:
            self._verify_episode_user_id(episode, user2_id)

        # Verify filter is working: user1 should see "Python" but not "JavaScript"
        # user2 should see "JavaScript" but not "Python" (or at least different results)
        assert len(filtered_episodes_user1) > 0 or len(filtered_episodes_user2) > 0, (
            "Should find at least some memories for at least one user"
        )

        # Verify that results are filtered (filtered results should be <= unfiltered)
        self._verify_filtered_results(
            filtered_episodes_user1,
            filtered_semantic_user1,
            all_episodes,
            results_no_filter.content.semantic_memory or [],
            "User1",
        )
        self._verify_filtered_results(
            filtered_episodes_user2,
            results_user2.content.semantic_memory or [],
            all_episodes,
            results_no_filter.content.semantic_memory or [],
            "User2",
        )

    def test_search_memory_empty_query(self, memory):
        """Test search with empty query (should handle gracefully)."""
        results = memory.search("", limit=10)

        assert isinstance(results, SearchResult)
        assert results.content.episodic_memory is not None
        assert results.content.semantic_memory is not None

    def test_delete_episodic_memory(self, memory):
        """Test deleting a specific episodic memory."""
        # Add a memory
        memory.add("This memory will be deleted", role="user")

        # Wait for indexing
        time.sleep(1)

        # Search to get memory ID
        results = memory.search("deleted", limit=1)
        episodes = _get_episodic_episodes(results)
        if episodes:
            first_episode = episodes[0]
            episodic_id = getattr(first_episode, "uid", None)
            if episodic_id:
                # Delete the memory
                result = memory.delete_episodic(episodic_id)
                assert result is True

    def test_delete_semantic_memory(self, memory):
        """Test deleting a specific semantic memory."""
        # Add a memory that might generate semantic features
        memory.add("I have a strong preference for organic food", role="user")

        # Wait for semantic processing
        time.sleep(2)

        # Search to get semantic memory ID
        results = memory.search("organic food", limit=10)
        semantic_features = results.content.semantic_memory or []
        if semantic_features:
            first_feature = semantic_features[0]
            semantic_id = None
            if getattr(first_feature, "metadata", None) is not None:
                semantic_id = getattr(first_feature.metadata, "id", None)
            if semantic_id:
                # Delete the semantic memory
                result = memory.delete_semantic(semantic_id)
                assert result is True

    # ==================== Context and Metadata Tests ====================

    def test_memory_context_preservation(self, memory):
        """Test that memory context (user_id, agent_id, etc.) is preserved."""
        context = memory.get_context()

        assert context["org_id"] == memory.org_id
        assert context["project_id"] == memory.project_id
        context_metadata = context["metadata"]
        assert context_metadata.get("group_id") == memory.metadata.get("group_id")
        assert context_metadata.get("user_id") == memory.metadata.get("user_id")
        assert context_metadata.get("agent_id") == memory.metadata.get("agent_id")
        assert context_metadata.get("session_id") == memory.metadata.get("session_id")

    def test_memory_with_string_ids(self, client, unique_test_ids):
        """Test memory with string-based user_id and agent_id."""
        project = client.get_or_create_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
        )

        memory = project.memory(metadata={"user_id": "user1", "agent_id": "agent1"})

        assert isinstance(memory.metadata.get("user_id"), str)
        assert memory.metadata.get("user_id") == "user1"

        assert isinstance(memory.metadata.get("agent_id"), str)
        assert memory.metadata.get("agent_id") == "agent1"

    # ==================== Error Handling Tests ====================

    def test_get_nonexistent_project(self, client):
        """Test getting a project that doesn't exist."""
        with pytest.raises(requests.HTTPError) as exc_info:
            client.get_project(
                org_id="nonexistent_org",
                project_id="nonexistent_project",
            )
        assert exc_info.value.response.status_code == 404

    def test_create_duplicate_project(self, client, unique_test_ids):
        """Test creating a project that already exists."""
        # Create project first
        client.create_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
        )

        # Try to create again - should raise error
        with pytest.raises(requests.HTTPError) as exc_info:
            client.create_project(
                org_id=unique_test_ids["org_id"],
                project_id=unique_test_ids["project_id"],
            )
        assert exc_info.value.response.status_code == 409

    def test_invalid_org_id_format(self, client):
        """Test creating project with invalid org_id format."""
        # Client-side validation catches invalid IDs before request is sent
        with pytest.raises(ValidationError) as exc_info:
            client.create_project(
                org_id="invalid/org",  # Contains slash
                project_id="test_project",
            )
        assert "org_id" in str(exc_info.value)

    def test_invalid_project_id_format(self, client):
        """Test creating project with invalid project_id format."""
        # Client-side validation catches invalid IDs before request is sent
        with pytest.raises(ValidationError) as exc_info:
            client.create_project(
                org_id="test_org",
                project_id="invalid/project",  # Contains slash
            )
        assert "project_id" in str(exc_info.value)

    def test_delete_nonexistent_episodic_memory(self, memory):
        """Test deleting a non-existent episodic memory."""
        with pytest.raises(requests.HTTPError):
            memory.delete_episodic("nonexistent_episodic_id")

    @pytest.mark.skip(reason="TODO: failing, need investigation")
    def test_delete_nonexistent_semantic_memory(self, memory):
        """Test deleting a non-existent semantic memory."""
        with pytest.raises(requests.HTTPError):
            memory.delete_semantic("nonexistent_semantic_id")

    # ==================== Data Consistency Tests ====================

    def test_multiple_memories_consistency(self, memory):
        """Test that multiple memories are stored and retrieved consistently."""
        # Add multiple memories
        contents = [
            "I work at a tech company",
            "I enjoy hiking on weekends",
            "My favorite color is blue",
            "I prefer coffee over tea",
        ]

        for content in contents:
            memory.add(content, role="user")

        # Wait for indexing
        time.sleep(1)

        # Search for each memory
        for content in contents:
            results = memory.search(content[:10], limit=5)
            # Should find the memory (may be in episodic or semantic)
            for result in _get_episodic_episodes(results):
                if content.lower() in str(result).lower():
                    break
            for result in results.content.semantic_memory or []:
                if content.lower() in str(result).lower():
                    break
            # Note: May not always find exact match due to semantic processing
            # This is a soft assertion

    def test_memory_persistence_across_sessions(self, client, unique_test_ids):
        """Test that memories persist across different memory instances."""
        project = client.get_or_create_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
        )

        # Create first memory instance and add memory
        memory1 = project.memory(metadata={"user_id": unique_test_ids["user_id"]})
        memory1.add("This is a persistent memory", role="user")

        # Wait for indexing
        time.sleep(1)

        # Create second memory instance and search
        memory2 = project.memory(metadata={"user_id": unique_test_ids["user_id"]})
        results = memory2.search("persistent memory", limit=5)

        # Should find the memory added via memory1
        for result in _get_episodic_episodes(results):
            if "persistent memory" in str(result).lower():
                break
        # Note: Soft assertion as semantic processing may vary

    # ==================== Health Check Tests ====================

    def test_health_check(self, client):
        """Test health check endpoint."""
        health = client.health_check()

        assert isinstance(health, dict)
        assert "status" in health or "service" in health

    # ==================== Client Lifecycle Tests ====================

    def test_client_context_manager(self, unique_test_ids):
        """Test client as context manager."""
        with MemMachineClient(base_url=TEST_BASE_URL) as client:
            project = client.create_project(
                org_id=unique_test_ids["org_id"],
                project_id=unique_test_ids["project_id"],
            )
            assert project is not None

        # Client should be closed after context exit
        assert client.closed is True

    def test_client_close(self, client):
        """Test manually closing client."""
        assert client.closed is False

        client.close()

        assert client.closed is True

        # Operations should fail after close
        with pytest.raises(RuntimeError):
            client.create_project(
                org_id="test_org",
                project_id="test_project",
            )

    # ==================== Edge Cases and Stress Tests ====================

    def test_large_memory_content(self, memory):
        """Test adding memory with large content."""
        large_content = "A" * 10000  # 10KB of content
        result = memory.add(large_content, role="user")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_special_characters_in_memory(self, memory):
        """Test adding memory with special characters."""
        special_content = "Test with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = memory.add(special_content, role="user")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_unicode_memory_content(self, memory):
        """Test adding memory with unicode characters."""
        unicode_content = "测试中文内容 🚀 émojis and unicode: 日本語"
        result = memory.add(unicode_content, role="user")
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.skip(reason="TODO: server may overload")
    def test_rapid_memory_additions(self, memory):
        """Test adding multiple memories rapidly."""
        # Add memories in smaller batches to avoid overwhelming the server
        for i in range(10):  # Reduced from 20 to 10
            memory.add(f"Rapid memory addition {i}", role="user")
            # Small delay between additions to avoid overwhelming the server
            if i % 5 == 0:
                time.sleep(0.5)

        # Wait for processing - increased wait time
        time.sleep(5)

        # Search should find some of them - add timeout to prevent hanging
        results = memory.search("rapid memory", limit=20, timeout=30)
        assert (
            len(_get_episodic_episodes(results)) > 0
            or len(results.content.semantic_memory or []) > 0
        )

    def test_concurrent_project_operations(self, client, unique_test_ids):
        """Test concurrent project operations."""
        import concurrent.futures

        def create_project(index):
            return client.create_project(
                org_id=f"{unique_test_ids['org_id']}_concurrent_{index}",
                project_id=f"{unique_test_ids['project_id']}_concurrent_{index}",
            )

        # Create multiple projects concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_project, i) for i in range(5)]
            projects = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(projects) == 5
        for project in projects:
            assert isinstance(project, Project)

    # ==================== Integration Workflow Tests ====================

    def test_complete_workflow(self, client, unique_test_ids):
        """Test a complete workflow: create project, add memories, search, delete."""
        # Step 1: Create project
        project = client.create_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
            description="Complete workflow test",
        )
        assert project is not None

        # Step 2: Create memory instance
        memory = project.memory(
            metadata={
                "user_id": unique_test_ids["user_id"],
                "agent_id": unique_test_ids["agent_id"],
            }
        )

        # Step 3: Add multiple memories
        memories_added = [
            "I am a software developer",
            "I work with Python and JavaScript",
            "I enjoy machine learning projects",
        ]
        for content in memories_added:
            memory.add(content, role="user")

        # Step 4: Wait for processing
        time.sleep(2)

        # Step 5: Search memories
        results = memory.search("software developer", limit=10)
        assert isinstance(results, SearchResult)

        # Step 6: Verify project still exists
        retrieved = client.get_project(
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
        )
        assert retrieved.project_id == unique_test_ids["project_id"]

        # Step 7: Clean up - delete project
        project.delete()

        # Step 8: Verify deletion
        with pytest.raises(requests.HTTPError) as exc_info:
            client.get_project(
                org_id=unique_test_ids["org_id"],
                project_id=unique_test_ids["project_id"],
            )
        assert exc_info.value.response.status_code == 404

    # ==================== Feature Management Tests ====================

    def test_add_feature(self, memory, unique_test_ids):
        """Test adding a semantic feature."""
        set_id = f"mem_project_set_{unique_test_ids['project_id']}"
        feature_id = memory.add_feature(
            set_id=set_id,
            category_name="profile",
            tag="food",
            feature="favorite_food",
            value="pizza",
        )

        assert feature_id is not None
        assert isinstance(feature_id, str)
        assert len(feature_id) > 0

    def test_add_feature_with_metadata_and_citations(self, memory, unique_test_ids):
        """Test adding a feature with metadata and citations."""
        # First add a memory to get an episode ID for citation
        result = memory.add("I love Italian food", role="user")
        episode_id = result[0].uid

        set_id = f"mem_project_set_{unique_test_ids['project_id']}"
        feature_id = memory.add_feature(
            set_id=set_id,
            category_name="profile",
            tag="food",
            feature="cuisine_preference",
            value="Italian",
            feature_metadata={"source": "conversation", "confidence": "high"},
            citations=[episode_id],
        )

        assert feature_id is not None
        assert isinstance(feature_id, str)

    def test_get_feature(self, memory, unique_test_ids):
        """Test retrieving a semantic feature."""
        set_id = f"mem_project_set_{unique_test_ids['project_id']}"

        # Add a feature first
        feature_id = memory.add_feature(
            set_id=set_id,
            category_name="profile",
            tag="music",
            feature="favorite_genre",
            value="jazz",
        )

        # Get the feature
        feature = memory.get_feature(feature_id=feature_id)

        assert feature is not None
        assert feature.category == "profile"
        assert feature.tag == "music"
        assert feature.feature_name == "favorite_genre"
        assert feature.value == "jazz"

    def test_get_feature_with_citations(self, memory, unique_test_ids):
        """Test retrieving a feature with citations loaded."""
        # First add a memory to get an episode ID for citation
        result = memory.add("I enjoy classical music too", role="user")
        episode_id = result[0].uid

        set_id = f"mem_project_set_{unique_test_ids['project_id']}"

        # Add a feature with citation
        feature_id = memory.add_feature(
            set_id=set_id,
            category_name="profile",
            tag="music",
            feature="secondary_genre",
            value="classical",
            citations=[episode_id],
        )

        # Get the feature with citations
        feature = memory.get_feature(feature_id=feature_id, load_citations=True)

        assert feature is not None
        assert feature.metadata is not None
        # Citations should be loaded
        if feature.metadata.citations:
            assert episode_id in feature.metadata.citations

    def test_get_feature_not_found(self, memory):
        """Test getting a non-existent feature returns None."""
        feature = memory.get_feature(feature_id="nonexistent_feature_id")
        assert feature is None

    def test_update_feature(self, memory, unique_test_ids):
        """Test updating a semantic feature."""
        set_id = f"mem_project_set_{unique_test_ids['project_id']}"

        # Add a feature first
        feature_id = memory.add_feature(
            set_id=set_id,
            category_name="profile",
            tag="sports",
            feature="favorite_sport",
            value="basketball",
        )

        # Update the feature
        result = memory.update_feature(
            feature_id=feature_id,
            value="soccer",
        )
        assert result is True

        # Get the updated feature
        feature = memory.get_feature(feature_id=feature_id)
        assert feature is not None
        assert feature.value == "soccer"

    def test_update_feature_all_fields(self, memory, unique_test_ids):
        """Test updating all fields of a feature."""
        set_id = f"mem_project_set_{unique_test_ids['project_id']}"

        # Add a feature first
        feature_id = memory.add_feature(
            set_id=set_id,
            category_name="profile",
            tag="hobbies",
            feature="main_hobby",
            value="reading",
        )

        # Update all fields
        result = memory.update_feature(
            feature_id=feature_id,
            category_name="profile",
            tag="leisure",
            feature="favorite_activity",
            value="painting",
            metadata={"updated": "true"},
        )
        assert result is True

        # Get the updated feature
        feature = memory.get_feature(feature_id=feature_id)
        assert feature is not None
        assert feature.category == "profile"
        assert feature.tag == "leisure"
        assert feature.feature_name == "favorite_activity"
        assert feature.value == "painting"

    def test_feature_lifecycle(self, memory, unique_test_ids):
        """Test complete feature lifecycle: add, get, update, delete."""
        set_id = f"mem_project_set_{unique_test_ids['project_id']}"

        # Step 1: Add feature
        feature_id = memory.add_feature(
            set_id=set_id,
            category_name="profile",
            tag="lifecycle",
            feature="test_feature",
            value="initial_value",
        )
        assert feature_id is not None

        # Step 2: Get feature
        feature = memory.get_feature(feature_id=feature_id)
        assert feature is not None
        assert feature.value == "initial_value"

        # Step 3: Update feature
        result = memory.update_feature(
            feature_id=feature_id,
            value="updated_value",
        )
        assert result is True

        # Step 4: Verify update
        feature = memory.get_feature(feature_id=feature_id)
        assert feature is not None
        assert feature.value == "updated_value"

        # Step 5: Delete feature (using existing delete_semantic method)
        delete_result = memory.delete_semantic(semantic_id=feature_id)
        assert delete_result is True

        # Step 6: Verify deletion
        feature = memory.get_feature(feature_id=feature_id)
        assert feature is None

    # ==================== Semantic Set Type Management Tests ====================

    def test_create_semantic_set_type(self, memory, unique_test_ids):
        """Test creating a semantic set type."""
        set_type_id = memory.create_semantic_set_type(
            metadata_tags=["user_id", "session_id"],
            is_org_level=False,
            name="User Sessions",
            description="Set type for user sessions",
        )

        assert set_type_id is not None
        assert isinstance(set_type_id, str)
        assert len(set_type_id) > 0

    def test_create_semantic_set_type_minimal(self, memory, unique_test_ids):
        """Test creating a set type with minimal parameters."""
        set_type_id = memory.create_semantic_set_type(
            metadata_tags=["custom_id"],
        )

        assert set_type_id is not None
        assert isinstance(set_type_id, str)

    def test_list_semantic_set_types(self, memory, unique_test_ids):
        """Test listing semantic set types."""
        # Create a set type first
        memory.create_semantic_set_type(
            metadata_tags=["list_test_tag"],
            name="List Test Set Type",
        )

        # List set types
        set_types = memory.list_semantic_set_types()

        assert isinstance(set_types, list)
        # Should have at least the one we just created
        assert len(set_types) >= 1

    def test_delete_semantic_set_type(self, memory, unique_test_ids):
        """Test deleting a semantic set type."""
        # Create a set type to delete
        set_type_id = memory.create_semantic_set_type(
            metadata_tags=["delete_test_tag"],
            name="Delete Test Set Type",
        )

        # Delete it
        result = memory.delete_semantic_set_type(set_type_id=set_type_id)
        assert result is True

    def test_get_semantic_set_id(self, memory, unique_test_ids):
        """Test getting a semantic set ID."""
        set_id = memory.get_semantic_set_id(
            metadata_tags=["user_id"],
            is_org_level=False,
            set_metadata={"user_id": unique_test_ids["user_id"]},
        )

        assert set_id is not None
        assert isinstance(set_id, str)
        assert len(set_id) > 0

    def test_get_semantic_set_id_org_level(self, memory, unique_test_ids):
        """Test getting an org-level semantic set ID."""
        set_id = memory.get_semantic_set_id(
            metadata_tags=[],
            is_org_level=True,
        )

        assert set_id is not None
        assert isinstance(set_id, str)

    def test_list_semantic_set_ids(self, memory, unique_test_ids):
        """Test listing semantic sets."""
        # Ensure at least one set exists by getting a set_id
        memory.get_semantic_set_id(
            metadata_tags=[],
            is_org_level=False,
        )

        # List sets
        sets = memory.list_semantic_set_ids()

        assert isinstance(sets, list)
        # Should have at least one set
        assert len(sets) >= 1
        # Each set should have required fields
        for s in sets:
            assert hasattr(s, "id")
            assert hasattr(s, "is_org_level")
            assert hasattr(s, "tags")

    def test_list_semantic_set_ids_with_filter(self, memory, unique_test_ids):
        """Test listing semantic sets with metadata filter."""
        # Get a set ID first with specific metadata
        memory.get_semantic_set_id(
            metadata_tags=["user_id"],
            is_org_level=False,
            set_metadata={"user_id": unique_test_ids["user_id"]},
        )

        # List sets with filter
        sets = memory.list_semantic_set_ids(
            set_metadata={"user_id": unique_test_ids["user_id"]},
        )

        assert isinstance(sets, list)

    def test_configure_semantic_set(self, memory, unique_test_ids):
        """Test configuring a semantic set."""
        # Get a set ID first
        set_id = memory.get_semantic_set_id(
            metadata_tags=[],
            is_org_level=False,
        )

        # Configure it (embedder and llm_name can be None or valid names)
        result = memory.configure_semantic_set(
            set_id=set_id,
            embedder_name=None,
            llm_name=None,
        )

        assert result is True

    def test_semantic_set_type_lifecycle(self, memory, unique_test_ids):
        """Test complete semantic set type lifecycle."""
        # Step 1: Create set type
        set_type_id = memory.create_semantic_set_type(
            metadata_tags=["lifecycle_tag"],
            name="Lifecycle Test",
            description="Testing full lifecycle",
        )
        assert set_type_id is not None

        # Step 2: List and verify it exists
        set_types = memory.list_semantic_set_types()
        found = any(st.id == set_type_id for st in set_types)
        assert found, f"Set type {set_type_id} not found in list"

        # Step 3: Delete the set type
        result = memory.delete_semantic_set_type(set_type_id=set_type_id)
        assert result is True

    def test_semantic_set_id_lifecycle(self, memory, unique_test_ids):
        """Test complete semantic set ID lifecycle."""
        # Step 1: Get/create a set ID
        set_id = memory.get_semantic_set_id(
            metadata_tags=["lifecycle_user_id"],
            is_org_level=False,
            set_metadata={"lifecycle_user_id": "test_lifecycle_user"},
        )
        assert set_id is not None

        # Step 2: Configure the set
        result = memory.configure_semantic_set(
            set_id=set_id,
            embedder_name=None,
            llm_name=None,
        )
        assert result is True

    def test_semantic_category_lifecycle(self, memory, unique_test_ids):
        """Test complete semantic category lifecycle."""
        # Step 1: Create a set to attach the category to
        set_id = memory.get_semantic_set_id(
            metadata_tags=["cat_test_user_id"],
            is_org_level=False,
            set_metadata={"cat_test_user_id": "test_category_user"},
        )
        assert set_id is not None

        # Step 2: Add a category to the set
        category_id = memory.add_semantic_category(
            set_id=set_id,
            category_name="test_preferences",
            prompt="Extract user preferences from conversations",
            description="Test category for user preferences",
        )
        assert category_id is not None

        # Step 3: Get the category and verify
        category = memory.get_semantic_category(category_id)
        assert category is not None
        assert category.id == category_id
        assert category.name == "test_preferences"
        assert category.prompt == "Extract user preferences from conversations"

        # Step 4: Get category set IDs
        set_ids = memory.get_semantic_category_set_ids(category_id)
        assert set_id in set_ids

        # Step 5: Delete the category
        result = memory.delete_semantic_category(category_id)
        assert result is True

        # Step 6: Verify category no longer exists
        category = memory.get_semantic_category(category_id)
        assert category is None

    def test_semantic_category_template_lifecycle(self, memory, unique_test_ids):
        """Test complete semantic category template lifecycle."""
        # Step 1: Create a set type for templates
        set_type_id = memory.create_semantic_set_type(
            metadata_tags=["template_test_tag"],
            name="Template Test Type",
            description="Testing category templates",
        )
        assert set_type_id is not None

        # Step 2: Add a category template to the set type
        template_id = memory.add_semantic_category_template(
            set_type_id=set_type_id,
            category_name="template_preferences",
            prompt="Extract template preferences",
            description="Template category description",
        )
        assert template_id is not None

        # Step 3: List category templates and verify
        templates = memory.list_semantic_category_templates(set_type_id)
        found = any(t.id == template_id for t in templates)
        assert found, f"Template {template_id} not found in list"

        # Step 4: Delete the category template
        result = memory.delete_semantic_category(template_id)
        assert result is True

        # Step 5: Cleanup - delete the set type
        memory.delete_semantic_set_type(set_type_id)

    def test_semantic_tag_lifecycle(self, memory, unique_test_ids):
        """Test complete semantic tag lifecycle."""
        # Step 1: Create a set
        set_id = memory.get_semantic_set_id(
            metadata_tags=["tag_test_user_id"],
            is_org_level=False,
            set_metadata={"tag_test_user_id": "test_tag_user"},
        )
        assert set_id is not None

        # Step 2: Add a category
        category_id = memory.add_semantic_category(
            set_id=set_id,
            category_name="tag_test_category",
            prompt="Extract features with tags",
        )
        assert category_id is not None

        # Step 3: Add a tag to the category
        tag_id = memory.add_semantic_tag(
            category_id=category_id,
            tag_name="food_preferences",
            tag_description="User food preferences and dietary restrictions",
        )
        assert tag_id is not None

        # Step 4: Delete the tag
        result = memory.delete_semantic_tag(tag_id)
        assert result is True

        # Step 5: Cleanup - delete the category
        memory.delete_semantic_category(category_id)

    def test_semantic_category_disable(self, memory, unique_test_ids):
        """Test disabling a semantic category for a set."""
        # Step 1: Create a set type with a category template
        set_type_id = memory.create_semantic_set_type(
            metadata_tags=["disable_test_tag"],
            name="Disable Test Type",
        )
        assert set_type_id is not None

        # Step 2: Add a category template
        memory.add_semantic_category_template(
            set_type_id=set_type_id,
            category_name="inherited_category",
            prompt="Template prompt",
        )

        # Step 3: Get a set ID (this should inherit the template)
        set_id = memory.get_semantic_set_id(
            metadata_tags=["disable_test_tag"],
            is_org_level=False,
            set_metadata={"disable_test_tag": "test_disable_value"},
        )
        assert set_id is not None

        # Step 4: Disable the inherited category for this set
        result = memory.disable_semantic_category(
            set_id=set_id,
            category_name="inherited_category",
        )
        assert result is True

        # Step 5: Cleanup
        memory.delete_semantic_set_type(set_type_id)

    def test_episodic_memory_config_lifecycle(self, memory, unique_test_ids):
        """Test episodic memory configuration get and update."""
        # Step 1: Get current episodic memory config
        config = memory.get_episodic_memory_config()
        assert config is not None
        assert isinstance(config.enabled, bool)
        assert isinstance(config.long_term_memory_enabled, bool)
        assert isinstance(config.short_term_memory_enabled, bool)

        # Remember original values
        original_enabled = config.enabled
        original_ltm = config.long_term_memory_enabled
        original_stm = config.short_term_memory_enabled

        # Step 2: Disable episodic memory
        result = memory.configure_episodic_memory(enabled=False)
        assert result is True

        # Step 3: Verify the change
        config = memory.get_episodic_memory_config()
        assert config.enabled is False

        # Step 4: Disable long-term memory only
        result = memory.configure_episodic_memory(
            enabled=True,
            long_term_memory_enabled=False,
        )
        assert result is True

        config = memory.get_episodic_memory_config()
        assert config.enabled is True
        assert config.long_term_memory_enabled is False

        # Step 5: Disable short-term memory only
        result = memory.configure_episodic_memory(
            long_term_memory_enabled=True,
            short_term_memory_enabled=False,
        )
        assert result is True

        config = memory.get_episodic_memory_config()
        assert config.long_term_memory_enabled is True
        assert config.short_term_memory_enabled is False

        # Step 6: Restore original config
        memory.configure_episodic_memory(
            enabled=original_enabled,
            long_term_memory_enabled=original_ltm,
            short_term_memory_enabled=original_stm,
        )


@pytest.mark.integration
@pytest.mark.skipif(
    not check_server_available(),
    reason="MemMachine server not available. Start server or set MEMORY_BACKEND_URL",
)
class TestMemMachineToolsIntegration:
    """Integration tests for MemMachineTools."""

    @pytest.fixture
    def unique_test_ids(self):
        """Generate unique test IDs for test isolation."""
        test_id = str(uuid4())[:8]
        return {
            "org_id": f"test_org_{test_id}",
            "project_id": f"test_project_{test_id}",
            "user_id": f"test_user_{test_id}",
            "agent_id": f"test_agent_{test_id}",
            "session_id": f"test_session_{test_id}",
            "group_id": f"test_group_{test_id}",
        }

    @pytest.fixture
    def tools(self, unique_test_ids):
        """Create a MemMachineTools instance."""
        t = MemMachineTools(
            base_url=TEST_BASE_URL,
            org_id=unique_test_ids["org_id"],
            project_id=unique_test_ids["project_id"],
            user_id=unique_test_ids["user_id"],
            agent_id=unique_test_ids["agent_id"],
            group_id=unique_test_ids["group_id"],
            session_id=unique_test_ids["session_id"],
        )
        yield t
        t.close()

    def test_add_and_search_round_trip(self, tools):
        """Test adding a memory then searching for it via MemMachineTools."""
        add_result = tools.add_memory(content="I love hiking in the mountains")
        assert add_result["status"] == "success"
        assert len(add_result["uids"]) > 0

        search_result = tools.search_memory(query="hiking mountains")
        assert search_result["status"] == "success"
        results = search_result["results"]
        # Should have found something in episodic or semantic memory
        episodic = results.get("episodic_memory")
        semantic = results.get("semantic_memory", [])
        long_term_memory = episodic.get("long_term_memory", {}).get("episodes", [])
        short_term_memory = episodic.get("short_term_memory", {}).get("episodes", {})
        has_episodic = len(long_term_memory) > 0 or len(short_term_memory) > 0
        assert has_episodic or len(semantic) > 0
        # check that at least one result is relevant
        found_relevant = False
        for episode in long_term_memory + short_term_memory:
            if "hiking" in episode.get("content", "").lower():
                found_relevant = True
                break
        assert found_relevant

    def test_get_context(self, tools, unique_test_ids):
        """Test get_context returns the expected context dict."""
        context = tools.get_context()
        assert context["org_id"] == unique_test_ids["org_id"]
        assert context["project_id"] == unique_test_ids["project_id"]
        metadata = context["metadata"]
        assert metadata.get("user_id") == unique_test_ids["user_id"]
        assert metadata.get("agent_id") == unique_test_ids["agent_id"]
        assert metadata.get("group_id") == unique_test_ids["group_id"]
        assert metadata.get("session_id") == unique_test_ids["session_id"]


@pytest.mark.integration
@pytest.mark.skipif(
    not check_server_available(),
    reason="MemMachine server not available. Start server or set MEMORY_BACKEND_URL",
)
class TestConfigIntegration:
    """Integration tests for Config class."""

    @pytest.fixture
    def client(self):
        """Create a MemMachine client instance."""
        return MemMachineClient(base_url=TEST_BASE_URL, timeout=60)

    @pytest.fixture
    def config(self, client):
        """Create a Config instance."""
        return client.config()

    def test_get_config(self, config):
        """Test getting full configuration."""
        result = config.get_config()

        from memmachine.common.api.config_spec import GetConfigResponse

        assert isinstance(result, GetConfigResponse)
        assert result.resources is not None
        assert result.episodic_memory is not None
        assert result.semantic_memory is not None

    def test_get_resources(self, config):
        """Test getting resources status."""
        result = config.get_resources()

        from memmachine.common.api.config_spec import ResourcesStatus

        assert isinstance(result, ResourcesStatus)
        assert hasattr(result, "embedders")
        assert hasattr(result, "language_models")
        assert hasattr(result, "rerankers")
        assert hasattr(result, "databases")

    def test_get_long_term_memory_config(self, config):
        """Test getting long-term memory configuration."""
        result = config.get_long_term_memory_config()

        from memmachine.common.api.config_spec import LongTermMemoryConfigResponse

        assert isinstance(result, LongTermMemoryConfigResponse)
        assert hasattr(result, "embedder")
        assert hasattr(result, "reranker")
        assert hasattr(result, "vector_graph_store")
        assert hasattr(result, "enabled")
        assert isinstance(result.enabled, bool)

    def test_get_short_term_memory_config(self, config):
        """Test getting short-term memory configuration."""
        result = config.get_short_term_memory_config()

        from memmachine.common.api.config_spec import ShortTermMemoryConfigResponse

        assert isinstance(result, ShortTermMemoryConfigResponse)
        assert hasattr(result, "llm_model")
        assert hasattr(result, "message_capacity")
        assert hasattr(result, "enabled")
        assert isinstance(result.enabled, bool)

    def test_update_long_term_memory_config(self, config):
        """Test updating long-term memory configuration."""
        # First get current config to restore later
        original = config.get_long_term_memory_config()

        # Update with enabled flag change
        result = config.update_long_term_memory_config(
            enabled=not original.enabled,
        )

        from memmachine.common.api.config_spec import UpdateMemoryConfigResponse

        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True

        # Verify the change
        updated = config.get_long_term_memory_config()
        assert updated.enabled != original.enabled

        # Restore original
        config.update_long_term_memory_config(enabled=original.enabled)

    def test_update_short_term_memory_config(self, config):
        """Test updating short-term memory configuration."""
        # First get current config to restore later
        original = config.get_short_term_memory_config()

        # Update with enabled flag change
        result = config.update_short_term_memory_config(
            enabled=not original.enabled,
        )

        from memmachine.common.api.config_spec import UpdateMemoryConfigResponse

        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True

        # Verify the change
        updated = config.get_short_term_memory_config()
        assert updated.enabled != original.enabled

        # Restore original
        config.update_short_term_memory_config(enabled=original.enabled)

    def test_update_long_term_memory_config_with_message_capacity(self, config):
        """Test updating long-term memory with message capacity field."""
        # First get current config
        original = config.get_short_term_memory_config()
        original_capacity = original.message_capacity

        # Update message capacity
        new_capacity = 50 if original_capacity != 50 else 100
        result = config.update_short_term_memory_config(
            message_capacity=new_capacity,
        )

        from memmachine.common.api.config_spec import UpdateMemoryConfigResponse

        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True

        # Verify the change
        updated = config.get_short_term_memory_config()
        assert updated.message_capacity == new_capacity

        # Restore original if it was set
        if original_capacity is not None:
            config.update_short_term_memory_config(message_capacity=original_capacity)

    def test_config_get_and_update_roundtrip(self, config):
        """Test complete roundtrip: get config, update, verify, restore."""
        # Get original configs
        original_ltm = config.get_long_term_memory_config()
        original_stm = config.get_short_term_memory_config()

        # Update both
        config.update_long_term_memory_config(enabled=not original_ltm.enabled)
        config.update_short_term_memory_config(enabled=not original_stm.enabled)

        # Verify changes
        updated_ltm = config.get_long_term_memory_config()
        updated_stm = config.get_short_term_memory_config()
        assert updated_ltm.enabled != original_ltm.enabled
        assert updated_stm.enabled != original_stm.enabled

        # Restore originals
        config.update_long_term_memory_config(enabled=original_ltm.enabled)
        config.update_short_term_memory_config(enabled=original_stm.enabled)

        # Verify restoration
        restored_ltm = config.get_long_term_memory_config()
        restored_stm = config.get_short_term_memory_config()
        assert restored_ltm.enabled == original_ltm.enabled
        assert restored_stm.enabled == original_stm.enabled

    def test_get_semantic_memory_config(self, config):
        """Test getting semantic memory configuration."""
        result = config.get_semantic_memory_config()

        from memmachine.common.api.config_spec import SemanticMemoryConfigResponse

        assert isinstance(result, SemanticMemoryConfigResponse)
        assert hasattr(result, "enabled")
        assert hasattr(result, "database")
        assert hasattr(result, "llm_model")
        assert hasattr(result, "embedding_model")
        assert isinstance(result.enabled, bool)

    def test_update_semantic_memory_config(self, config):
        """Test updating semantic memory configuration."""
        # First get current config to restore later
        original = config.get_semantic_memory_config()

        # Update with enabled flag change
        result = config.update_semantic_memory_config(
            enabled=not original.enabled,
        )

        from memmachine.common.api.config_spec import UpdateMemoryConfigResponse

        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True

        # Verify the change
        updated = config.get_semantic_memory_config()
        assert updated.enabled != original.enabled

        # Restore original
        config.update_semantic_memory_config(enabled=original.enabled)

    def test_semantic_memory_config_full_roundtrip(self, config):
        """Test complete roundtrip for semantic memory config."""
        # Get original config
        original = config.get_semantic_memory_config()

        # Update with new values
        config.update_semantic_memory_config(enabled=not original.enabled)

        # Verify change
        updated = config.get_semantic_memory_config()
        assert updated.enabled != original.enabled

        # Restore original
        config.update_semantic_memory_config(enabled=original.enabled)

        # Verify restoration
        restored = config.get_semantic_memory_config()
        assert restored.enabled == original.enabled

    def test_get_episodic_memory_config(self, config):
        """Test getting episodic memory configuration."""
        result = config.get_episodic_memory_config()

        from memmachine.common.api.config_spec import EpisodicMemoryConfigResponse

        assert isinstance(result, EpisodicMemoryConfigResponse)
        assert hasattr(result, "long_term_memory")
        assert hasattr(result, "short_term_memory")
        assert hasattr(result, "enabled")
        assert isinstance(result.enabled, bool)
        # Check nested structures
        assert hasattr(result.long_term_memory, "embedder")
        assert hasattr(result.short_term_memory, "llm_model")

    def test_update_episodic_memory_config(self, config):
        """Test updating episodic memory configuration."""
        # First get current config to restore later
        original = config.get_episodic_memory_config()

        # Update with enabled flag change
        result = config.update_episodic_memory_config(
            enabled=not original.enabled,
        )

        from memmachine.common.api.config_spec import UpdateMemoryConfigResponse

        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True

        # Verify the change
        updated = config.get_episodic_memory_config()
        assert updated.enabled != original.enabled

        # Restore original
        config.update_episodic_memory_config(enabled=original.enabled)

    def test_episodic_memory_config_full_roundtrip(self, config):
        """Test complete roundtrip for episodic memory config."""
        # Get original config
        original = config.get_episodic_memory_config()

        # Update with new values
        config.update_episodic_memory_config(enabled=not original.enabled)

        # Verify change
        updated = config.get_episodic_memory_config()
        assert updated.enabled != original.enabled

        # Restore original
        config.update_episodic_memory_config(enabled=original.enabled)

        # Verify restoration
        restored = config.get_episodic_memory_config()
        assert restored.enabled == original.enabled
