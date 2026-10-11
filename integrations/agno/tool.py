"""Expose MemMachine memory operations as an Agno toolkit."""

import json
from typing import Literal

from memmachine_client import MemMachineClient, Memory

from agno.tools import Toolkit


class MemMachineTools(Toolkit):
    """Store and retrieve memories within a configured MemMachine context."""

    def __init__(
        self,
        client: MemMachineClient | None = None,
        base_url: str = "http://localhost:8080",
        org_id: str = "agno_org",
        project_id: str = "agno_project",
        group_id: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize tools with fixed project and metadata scope.

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
        self.client = (
            client if client is not None else MemMachineClient(base_url=base_url)
        )
        self._owns_client = client is None
        self.org_id = org_id
        self.project_id = project_id
        self.group_id = group_id
        self.agent_id = agent_id
        self.user_id = user_id
        self.session_id = session_id
        super().__init__(
            name="memmachine_tools",
            tools=[self.add_memory, self.search_memory],
        )

    def get_memory(self) -> Memory:
        """Get a memory handle using this toolkit's configured context."""
        project = self.client.get_or_create_project(
            org_id=self.org_id,
            project_id=self.project_id,
        )
        return project.memory(
            group_id=self.group_id,
            agent_id=self.agent_id,
            user_id=self.user_id,
            session_id=self.session_id,
        )

    def add_memory(
        self,
        content: str,
        role: Literal["user", "assistant", "system"] = "user",
    ) -> str:
        """Store facts, preferences, or conversation context for later recall.

        Args:
            content: Information to remember, including enough context to recall it.
            role: Who provided the information: user, assistant, or system.

        Returns:
            JSON containing the success status and stored memory identifiers.
        """
        if not content.strip():
            raise ValueError("Memory content must not be empty")
        results = self.get_memory().add(content=content, role=role)
        if not results:
            raise RuntimeError("MemMachine returned no identifiers for the new memory")
        return json.dumps(
            {"status": "success", "uids": [result.uid for result in results]}
        )

    def search_memory(self, query: str, limit: int = 5) -> str:
        """Recall relevant stored facts and past conversations before answering.

        Args:
            query: Natural-language description of the information to recall.
            limit: Maximum number of search results; must be positive.

        Returns:
            JSON containing MemMachine's episodic and semantic search results.
        """
        if not query.strip():
            raise ValueError("Search query must not be empty")
        if limit < 1:
            raise ValueError("Search limit must be positive")
        result = self.get_memory().search(query=query, limit=limit)
        return result.model_dump_json(exclude_none=True)

    def close(self) -> None:
        """Close the client only if this toolkit created it."""
        if self._owns_client:
            self.client.close()
