"""
Project management interface for MemMachine.

This module provides the Project class that represents a MemMachine project,
which serves as the boundary for memory operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import requests

from memmachine.common.api.spec import DeleteProjectSpec, GetProjectSpec

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
            group_id="my_group",  # Optional: stored in metadata
            agent_id="my_agent",  # Optional: stored in metadata
            user_id="user123",    # Optional: stored in metadata
            session_id="session456"  # Optional: stored in metadata
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
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize Project instance.

        Args:
            client: MemMachineClient instance
            org_id: Organization identifier
            project_id: Project identifier
            description: Project description
            configuration: Project configuration (from server)
            metadata: Project metadata (from server)

        """
        self.client = client
        self.org_id = org_id
        self.project_id = project_id
        self.description = description
        self.configuration = configuration or {}
        self.metadata = metadata or {}

    def memory(
        self,
        group_id: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        **kwargs: dict[str, Any],
    ) -> Memory:
        """
        Create a Memory instance for this project.

        Args:
            group_id: Group identifier (optional, will be stored in metadata)
            agent_id: Agent identifier (optional, will be stored in metadata)
            user_id: User identifier (optional, will be stored in metadata)
            session_id: Session identifier (optional, will be stored in metadata)
            **kwargs: Additional configuration options

        Returns:
            Memory instance configured for this project

        """
        from .memory import Memory

        memory = Memory(
            client=self.client,
            org_id=self.org_id,
            project_id=self.project_id,
            group_id=group_id,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
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
            project_data = response.json()

            # Update project attributes
            # Server API uses "config" but Project class uses "configuration" for consistency
            self.description = project_data.get("description", self.description)
            self.configuration = project_data.get("config", self.configuration)
            # Server does not return "metadata" in ProjectResponse, so we keep existing value
            # self.metadata remains unchanged
        except requests.RequestException:
            logger.exception(
                "Failed to refresh project %s/%s", self.org_id, self.project_id
            )
            raise

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
