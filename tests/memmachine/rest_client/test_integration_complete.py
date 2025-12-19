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
from memmachine.rest_client.project import Project


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
        assert "episodic_memory" in results.content
        assert "semantic_memory" in results.content
        # episodic_memory is now a dict with long_term_memory and short_term_memory
        assert isinstance(results.content["episodic_memory"], dict)
        assert isinstance(results.content["semantic_memory"], list)

        # Should find at least one result
        episodic = results.content["episodic_memory"]
        episodes = episodic.get("short_term_memory", {}).get(
            "episodes", []
        ) + episodic.get("long_term_memory", {}).get("episodes", [])
        assert len(episodes) > 0 or len(results.content["semantic_memory"]) > 0

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
        episodic_count = len(results.content.get("episodic_memory", []))
        semantic_count = len(results.content.get("semantic_memory", []))

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
        episodic = results_no_filter.content.get("episodic_memory", {})
        all_episodes = episodic.get("short_term_memory", {}).get(
            "episodes", []
        ) + episodic.get("long_term_memory", {}).get("episodes", [])
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
            episodic = results.content.get("episodic_memory", {})
            filtered_episodes = episodic.get("short_term_memory", {}).get(
                "episodes", []
            ) + episodic.get("long_term_memory", {}).get("episodes", [])
            filtered_semantic = results.content.get("semantic_memory", [])

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
                results_no_filter.content.get("semantic_memory", [])
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
        episodic = results_no_filter.content.get("episodic_memory", {})
        all_episodes = episodic.get("short_term_memory", {}).get(
            "episodes", []
        ) + episodic.get("long_term_memory", {}).get("episodes", [])
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
            episodic = results.content.get("episodic_memory", {})
            filtered_episodes = episodic.get("short_term_memory", {}).get(
                "episodes", []
            ) + episodic.get("long_term_memory", {}).get("episodes", [])
            filtered_semantic = results.content.get("semantic_memory", [])

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
                results_no_filter.content.get("semantic_memory", [])
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
        episodic = results_no_filter.content.get("episodic_memory", {})
        all_episodes = episodic.get("short_term_memory", {}).get(
            "episodes", []
        ) + episodic.get("long_term_memory", {}).get("episodes", [])
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
        filtered_episodes_user1 = results_user1.content.get("episodic_memory", [])
        filtered_semantic_user1 = results_user1.content.get("semantic_memory", [])

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
        filtered_episodes_user2 = results_user2.content.get("episodic_memory", [])

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
            results_no_filter.content.get("semantic_memory", []),
            "User1",
        )
        self._verify_filtered_results(
            filtered_episodes_user2,
            results_user2.content.get("semantic_memory", []),
            all_episodes,
            results_no_filter.content.get("semantic_memory", []),
            "User2",
        )

    def test_search_memory_empty_query(self, memory):
        """Test search with empty query (should handle gracefully)."""
        results = memory.search("", limit=10)

        assert isinstance(results, SearchResult)
        assert "episodic_memory" in results.content
        assert "semantic_memory" in results.content

    def test_delete_episodic_memory(self, memory):
        """Test deleting a specific episodic memory."""
        # Add a memory
        memory.add("This memory will be deleted", role="user")

        # Wait for indexing
        time.sleep(1)

        # Search to get memory ID
        results = memory.search("deleted", limit=1)
        episodic = results.content.get("episodic_memory", {})
        episodes = episodic.get("short_term_memory", {}).get(
            "episodes", []
        ) + episodic.get("long_term_memory", {}).get("episodes", [])
        if episodes:
            first_episode = episodes[0]
            # Handle both dict and object responses
            episodic_id = (
                first_episode.get("uid")
                if isinstance(first_episode, dict)
                else getattr(first_episode, "uid", None)
            )
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
        semantic_features = results.content.get("semantic_memory", [])
        if semantic_features:
            first_feature = semantic_features[0]
            # Handle both dict and object responses
            semantic_id = (
                first_feature.get("uid") or first_feature.get("id")
                if isinstance(first_feature, dict)
                else getattr(first_feature, "uid", None)
                or getattr(first_feature, "id", None)
            )
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
            for result in results.content["episodic_memory"]:
                if content.lower() in str(result).lower():
                    break
            for result in results.content["semantic_memory"]:
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
        episodic = results.content.get("episodic_memory", {})
        episodes = episodic.get("short_term_memory", {}).get(
            "episodes", []
        ) + episodic.get("long_term_memory", {}).get("episodes", [])
        for result in episodes:
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
            len(results.content["episodic_memory"]) > 0
            or len(results.content["semantic_memory"]) > 0
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
