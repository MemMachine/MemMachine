"""
Tools for integrating MemMachine memory operations into LangGraph.

This module provides tools that can be used in LangGraph workflows
to enable AI agents with persistent memory capabilities.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from memmachine.common.data_types import PropertyValue
from memmachine.rest_client.client import MemMachineClient

if TYPE_CHECKING:
    from memmachine.rest_client.memory import Memory


class MemMachineTools:
    """
    Tools for integrating MemMachine memory operations into LangGraph.

    This class provides static methods that can be used as tools in LangGraph workflows.
    """

    def __init__(
        self,
        client: MemMachineClient | None = None,
        base_url: str = "http://localhost:8080",
        org_id: str = "langgraph_org",
        project_id: str = "langgraph_project",
        group_id: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """
        Initialize MemMachine tools.

        Args:
            client: Optional MemMachineClient instance. If not provided, creates a new one.
            base_url: Base URL for MemMachine server
            org_id: Organization ID for v2 API (required)
            project_id: Project ID for v2 API (required)
            group_id: Optional group ID (stored in properties)
            agent_id: Optional agent ID (stored in properties)
            user_id: Optional user ID (stored in properties)
            session_id: Optional session ID (stored in properties)

        """
        self.client = client or MemMachineClient(base_url=base_url)
        self.org_id = org_id
        self.project_id = project_id
        self.group_id = group_id
        self.agent_id = agent_id
        self.user_id = user_id
        self.session_id = session_id

    def _build_properties(
        self,
        group_id: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, PropertyValue]:
        """
        Build properties dict from context fields, using overrides or instance defaults.

        Args:
            group_id: Group ID override
            agent_id: Agent ID override
            user_id: User ID override
            session_id: Session ID override

        Returns:
            Properties dictionary with non-None values

        """
        properties = {}
        resolved_group_id = group_id or self.group_id
        resolved_agent_id = agent_id or self.agent_id
        resolved_user_id = user_id or self.user_id
        resolved_session_id = session_id or self.session_id

        if resolved_group_id:
            properties["group_id"] = resolved_group_id
        if resolved_agent_id:
            properties["agent_id"] = resolved_agent_id
        if resolved_user_id:
            properties["user_id"] = resolved_user_id
        if resolved_session_id:
            properties["session_id"] = resolved_session_id

        return properties

    def get_memory(
        self,
        org_id: str | None = None,
        project_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        group_id: str | None = None,
        session_id: str | None = None,
    ) -> "Memory":
        """
        Get or create a memory instance for the specified context.

        Args:
            org_id: Organization ID (overrides default)
            project_id: Project ID (overrides default)
            user_id: User ID (overrides default, stored in properties)
            agent_id: Agent ID (overrides default, stored in properties)
            group_id: Group ID (overrides default, stored in properties)
            session_id: Session ID (overrides default, stored in properties)

        Returns:
            Memory instance

        """
        # Get or create project
        project = self.client.get_or_create_project(
            org_id=org_id or self.org_id,
            project_id=project_id or self.project_id,
        )

        properties = self._build_properties(
            group_id=group_id,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
        )

        return project.memory(properties=properties)

    def add_memory(
        self,
        content: str,
        role: str = "user",
        org_id: str | None = None,
        project_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        group_id: str | None = None,
        session_id: str | None = None,
        properties: dict[str, PropertyValue] | None = None,
    ) -> dict[str, Any]:
        """
        Add a memory to MemMachine.

        This tool stores important information about the user or conversation into memory.
        Use this automatically whenever the user shares new facts, preferences, plans,
        emotions, or other details that could be useful for future context.

        Args:
            content: The content to store in memory. Should include full conversation context.
            role: Message role - "user", "assistant", or "system" (default: "user")
            org_id: Organization ID (overrides default)
            project_id: Project ID (overrides default)
            user_id: User ID (overrides default, stored in properties)
            agent_id: Agent ID (overrides default, stored in properties)
            group_id: Group ID (overrides default, stored in properties)
            session_id: Session ID (overrides default, stored in properties)
            properties: Additional properties for the episode (values must be bool, int, float, str, or datetime)

        Returns:
            Dictionary with success status and message

        """
        try:
            memory = self.get_memory(
                org_id, project_id, user_id, agent_id, group_id, session_id
            )
            results = memory.add(
                content=content,
                role=role,
                properties=properties or {},
            )
            if results:
                return {
                    "status": "success",
                    "message": f"Memory added successfully: {content[:50]}...",
                    "content": content,
                    "uids": [r.uid for r in results],
                }
        except Exception:
            return {
                "status": "error",
                "message": "Error adding memory",
            }
        else:
            return {
                "status": "error",
                "message": "Failed to add memory",
            }

    def search_memory(
        self,
        query: str,
        org_id: str | None = None,
        project_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        group_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        score_threshold: float | None = None,
        filter_dict: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Search for memories in MemMachine.

        This tool retrieves relevant context, memories or profile for a user whenever
        context is missing or unclear. Use this whenever you need to recall what has been
        previously discussed, even if it was from an earlier conversation or session.
        This searches both profile memory (long-term user traits and facts) and episodic
        memory (past conversations and experiences).

        Args:
            query: Search query string
            org_id: Organization ID (overrides default)
            project_id: Project ID (overrides default)
            user_id: User ID (overrides default, stored in properties)
            agent_id: Agent ID (overrides default, stored in properties)
            group_id: Group ID (overrides default, stored in properties)
            session_id: Session ID (overrides default, stored in properties)
            limit: Maximum number of results to return (default: 20)
            score_threshold: Minimum score to include in results
            filter_dict: Additional filters for the search

        Returns:
            Dictionary containing search results and relevant memories

        """
        try:
            memory = self.get_memory(
                org_id, project_id, user_id, agent_id, group_id, session_id
            )
            result = memory.search(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                filter_dict=filter_dict,
            )

            # Format results for easier consumption
            # result is a SearchResult Pydantic model with .content (SearchResultContent)
            formatted_results: dict[str, Any] = {
                "query": query,
                "episodic_memory": [],
                "semantic_memory": [],
            }

            if result.content:
                # Extract episodic memories
                if result.content.episodic_memory is not None:
                    episodic = result.content.episodic_memory
                    # EpisodicSearchResult - serialize to dict for downstream use
                    formatted_results["episodic_memory"] = episodic.model_dump()

                # Extract semantic memories
                if result.content.semantic_memory is not None:
                    formatted_results["semantic_memory"] = [
                        feat.model_dump() for feat in result.content.semantic_memory
                    ]

            return {
                "status": "success",
                "results": formatted_results,
                "summary": self._format_search_summary(formatted_results),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error searching memory: {e!s}",
            }

    def _format_search_summary(self, results: dict[str, Any]) -> str:
        """
        Format search results into a readable summary.

        Args:
            results: Search results dictionary

        Returns:
            Formatted summary string

        """
        summary_parts = []

        episodic_memories = results.get("episodic_memory")
        if episodic_memories:
            if isinstance(episodic_memories, dict):
                # EpisodicSearchResult dumped as dict
                episodes = episodic_memories.get("episodes", [])
                if episodes:
                    summary_parts.append(f"Found {len(episodes)} episodic memories:")
                    for i, mem in enumerate(episodes[:3], 1):
                        content = (
                            mem.get("content", "")
                            if isinstance(mem, dict)
                            else str(mem)
                        )
                        summary_parts.append(f"  {i}. {content[:100]}...")
            elif isinstance(episodic_memories, list) and episodic_memories:
                summary_parts.append(
                    f"Found {len(episodic_memories)} episodic memories:"
                )
                for i, mem in enumerate(episodic_memories[:3], 1):
                    content = (
                        mem.get("content", "") if isinstance(mem, dict) else str(mem)
                    )
                    summary_parts.append(f"  {i}. {content[:100]}...")

        semantic_memories = results.get("semantic_memory", [])
        if semantic_memories:
            summary_parts.append(f"Found {len(semantic_memories)} semantic memories")

        if not summary_parts:
            return "No relevant memories found."

        return "\n".join(summary_parts)

    def get_context(
        self,
        org_id: str | None = None,
        project_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        group_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get the current memory context.

        Args:
            org_id: Organization ID (overrides default)
            project_id: Project ID (overrides default)
            user_id: User ID (overrides default)
            agent_id: Agent ID (overrides default)
            group_id: Group ID (overrides default)
            session_id: Session ID (overrides default)

        Returns:
            Dictionary containing context information

        """
        memory = self.get_memory(
            org_id, project_id, user_id, agent_id, group_id, session_id
        )
        return memory.get_context()

    def close(self) -> None:
        """Close the client and clean up resources."""
        if self.client:
            self.client.close()


# Convenience functions for LangGraph tool creation
def create_add_memory_tool(
    tools: MemMachineTools,
) -> Callable[[str, str | None, dict[str, Any] | None], dict[str, Any]]:
    """
    Create an add_memory tool function for LangGraph.

    Args:
        tools: MemMachineTools instance

    Returns:
        Tool function that can be used in LangGraph

    """

    def add_memory_tool(
        content: str,
        user_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Tool to add a memory to MemMachine.

        Args:
            content: The content to store in memory
            user_id: Optional user ID override
            properties: Optional properties for the memory

        Returns:
            Result dictionary with status and message

        """
        return tools.add_memory(
            content=content,
            user_id=user_id,
            properties=properties,
        )

    return add_memory_tool


def create_search_memory_tool(
    tools: MemMachineTools,
) -> Callable[[str, str | None, int], dict[str, Any]]:
    """
    Create a search_memory tool function for LangGraph.

    Args:
        tools: MemMachineTools instance

    Returns:
        Tool function that can be used in LangGraph

    """

    def search_memory_tool(
        query: str,
        user_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Tool to search memories in MemMachine.

        Args:
            query: Search query string
            user_id: Optional user ID override
            limit: Maximum number of results to return

        Returns:
            Result dictionary with search results

        """
        return tools.search_memory(
            query=query,
            user_id=user_id,
            limit=limit,
        )

    return search_memory_tool
