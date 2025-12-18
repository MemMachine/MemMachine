"""
Project management interface for MemMachine.

This module provides the Project class that represents a MemMachine project,
which serves as the boundary for memory operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import requests

from memmachine.common.api.spec import (
    DeleteProjectSpec,
    EpisodeCountResponse,
    GetProjectSpec,
    ProjectConfig,
    ProjectResponse,
)

if TYPE_CHECKING:
    from .client import MemMachineClient
    from .memory import Memory

logger = logging.getLogger(__name__)


class Project:
    """
    Project interface for MemMachine.

    A Project represents a memory boundary in MemMachine. All memory operations
    are scoped to a specific project within an organization.

    Example:
        ```python
        from memmachine import MemMachineClient

        client = MemMachineClient(base_url="http://localhost:8080")

        # Create a new project
        project = client.create_project(
            org_id="my_org",
            project_id="my_project",
            description="My project description"
        )

        # Or get an existing project
        project = client.get_project(org_id="my_org", project_id="my_project")

        # Create memory instance from project
        memory = project.memory(
            metadata={
                "user_id": "user123",
                "agent_id": "my_agent",
                "group_id": "my_group",
                "session_id": "session456"
            }
        )

        # Use memory
        memory.add("I like pizza")
        results = memory.search("What do I like?")
        ```

    """

    def __init__(
        self,
        client: MemMachineClient,
        org_id: str,
        project_id: str,
        description: str = "",
        config: ProjectConfig | None = None,
    ) -> None:
        """
        Initialize Project instance.

        Args:
            client: MemMachineClient instance
            org_id: Organization identifier
            project_id: Project identifier
            description: Project description
            config: Project configuration (from server)

        """
        self.client = client
        self.org_id = org_id
        self.project_id = project_id
        self.description = description
        self.config = (
            config if config is not None else ProjectConfig(embedder="", reranker="")
        )

    def memory(
        self,
        metadata: dict[str, str] | None = None,
        **kwargs: dict[str, Any],
    ) -> Memory:
        """
        Create a Memory instance for this project.

        Args:
            metadata: Metadata dictionary that will be merged with metadata
                     in add() and search() operations. Common keys include:
                     user_id, agent_id, group_id, session_id, etc.
            **kwargs: Additional configuration options

        Returns:
            Memory instance configured for this project

        """
        from .memory import Memory

        memory = Memory(
            client=self.client,
            org_id=self.org_id,
            project_id=self.project_id,
            metadata=metadata,
            **kwargs,
        )
        return memory

    def delete(self, timeout: int | None = None) -> bool:
        """
        Delete this project.

        Args:
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            True if the project was deleted successfully

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        if self.client.closed:
            raise RuntimeError("Cannot delete project: client has been closed")

        url = f"{self.client.base_url}/api/v2/projects/delete"
        spec = DeleteProjectSpec(org_id=self.org_id, project_id=self.project_id)
        data = spec.model_dump(exclude_none=True)

        try:
            response = self.client.request("POST", url, json=data, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException:
            logger.exception(
                "Failed to delete project %s/%s", self.org_id, self.project_id
            )
            raise
        else:
            logger.debug("Project deleted: %s/%s", self.org_id, self.project_id)
            return True

    def refresh(self, timeout: int | None = None) -> None:
        """
        Refresh project information from the server.

        Args:
            timeout: Request timeout in seconds (uses client default if not provided)

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        if self.client.closed:
            raise RuntimeError("Cannot refresh project: client has been closed")

        url = f"{self.client.base_url}/api/v2/projects/get"
        spec = GetProjectSpec(org_id=self.org_id, project_id=self.project_id)
        data = spec.model_dump(exclude_none=True)

        try:
            response = self.client.request("POST", url, json=data, timeout=timeout)
            response.raise_for_status()
            response_data = response.json()
            # Parse response using Pydantic model for validation
            project_response = ProjectResponse(**response_data)

            # Update project attributes
            self.description = project_response.description
            self.config = project_response.config
        except requests.RequestException:
            logger.exception(
                "Failed to refresh project %s/%s", self.org_id, self.project_id
            )
            raise

    def get_episode_count(self, timeout: int | None = None) -> int:
        """
        Get the episode count for this project.

        Args:
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            The number of episodes associated with this project

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        if self.client.closed:
            raise RuntimeError("Cannot get episode count: client has been closed")

        url = f"{self.client.base_url}/api/v2/projects/episode_count/get"
        spec = GetProjectSpec(org_id=self.org_id, project_id=self.project_id)
        data = spec.model_dump(exclude_none=True)

        try:
            response = self.client.request("POST", url, json=data, timeout=timeout)
            response.raise_for_status()
            result_data = response.json()
            # Parse response using Pydantic model for validation
            episode_count = EpisodeCountResponse(**result_data)
        except requests.RequestException:
            logger.exception(
                "Failed to get episode count for project %s/%s",
                self.org_id,
                self.project_id,
            )
            raise
        else:
            return episode_count.count

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"Project(org_id='{self.org_id}', "
            f"project_id='{self.project_id}', "
            f"description='{self.description}')"
        )

    def __eq__(self, other: object) -> bool:
        """Check if two projects are equal (same org_id and project_id)."""
        if not isinstance(other, Project):
            return False
        return self.org_id == other.org_id and self.project_id == other.project_id
