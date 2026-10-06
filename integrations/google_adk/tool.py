"""Expose MemMachine memory operations as Google ADK function tools."""

from __future__ import annotations

from typing import Literal

from memmachine_client import MemMachineClient, Memory


class MemMachineTools:
    """Store and retrieve memories within a configured MemMachine context.

    The bound methods on this instance (``add_memory`` and ``search_memory``)
    are plain callables with type hints and docstrings, so they can be passed
    directly to a Google ADK ``LlmAgent`` via its ``tools`` argument.
    """

    def __init__(
        self,
        client: MemMachineClient | None = None,
        base_url: str = "http://localhost:8080",
        org_id: str = "google_adk_org",
        project_id: str = "google_adk_project",
        group_id: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize tools with a fixed project and metadata scope.

        Args:
            client: Optional client managed by the caller.
            base_url: Backend URL used when creating a client.
            org_id: Organization containing the memory project.
            project_id: Project to get or create on the first tool call.
            group_id: Optional group metadata and search filter.
            agent_id: Optional agent metadata and search filter.
            user_id: Optional user metadata and search filter.
            session_id: Optional session metadata and search filter.
        """
        self._owns_client = client is None
        self.client = client or MemMachineClient(base_url=base_url)
        self.org_id = org_id
        self.project_id = project_id
        self.group_id = group_id
        self.agent_id = agent_id
        self.user_id = user_id
        self.session_id = session_id

    @property
    def tools(self) -> list:
        """Callables to register with an ADK agent's ``tools`` argument."""
        return [self.add_memory, self.search_memory]

    def get_memory(self) -> Memory:
        """Get a memory handle using this toolkit's configured context."""
        project = self.client.get_or_create_project(
            org_id=self.org_id,
            project_id=self.project_id,
        )
        metadata = {
            key: value
            for key, value in (
                ("group_id", self.group_id),
                ("agent_id", self.agent_id),
                ("user_id", self.user_id),
                ("session_id", self.session_id),
            )
            if value
        }
        return project.memory(metadata=metadata)

    def add_memory(
        self,
        content: str,
        role: Literal["user", "assistant", "system"] = "user",
    ) -> dict:
        """Store facts, preferences, or conversation context for later recall.

        Args:
            content: Information to remember, including enough context to recall it.
            role: Who provided the information: user, assistant, or system.

        Returns:
            A dict with the success status and the stored memory identifiers.
        """
        if not content.strip():
            return {"status": "error", "error_message": "Memory content must not be empty"}

        try:
            results = self.get_memory().add(content=content, role=role)
        except Exception as error:  # noqa: BLE001 - surface any backend failure to the agent
            return {"status": "error", "error_message": f"Failed to add memory: {error}"}
        identifiers = [result.uid for result in results]
        if not identifiers:
            return {
                "status": "error",
                "error_message": "MemMachine returned no identifiers for the new memory",
            }
        return {"status": "success", "memory_ids": identifiers}

    def search_memory(self, query: str, limit: int = 5) -> dict:
        """Recall relevant stored facts and past conversations before answering.

        Args:
            query: Natural-language description of the information to recall.
            limit: Maximum number of search results; must be positive.

        Returns:
            A dict with MemMachine's episodic and semantic search results.
        """
        if not query.strip():
            return {"status": "error", "error_message": "Search query must not be empty"}
        if limit <= 0:
            return {"status": "error", "error_message": "Search limit must be positive"}

        try:
            result = self.get_memory().search(query=query, limit=limit)
        except Exception as error:  # noqa: BLE001 - surface any backend failure to the agent
            return {"status": "error", "error_message": f"Failed to search memory: {error}"}
        return {"status": "success", "results": result.model_dump(mode="json")}

    def close(self) -> None:
        """Close the client only if this toolkit created it."""
        if self._owns_client:
            self.client.close()
