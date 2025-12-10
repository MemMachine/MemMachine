from typing import Any

import requests

from memmachine import MemMachineClient


class MemMachineTools:
    """Tools for integrating MemMachine memory operations into LangGraph.

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
        """Initialize MemMachine tools.

        Args:
            client: Optional MemMachineClient instance. If not provided, creates a new one.
            base_url: Base URL for MemMachine server
            org_id: Organization ID for v2 API (required)
            project_id: Project ID for v2 API (required)
            group_id: Optional group ID (stored in metadata)
            agent_id: Optional agent ID (stored in metadata)
            user_id: Optional user ID (stored in metadata)
            session_id: Optional session ID (stored in metadata)

        """
        self.client = client or MemMachineClient(base_url=base_url)
        self.org_id = org_id
        self.project_id = project_id
        self.group_id = group_id
        self.agent_id = agent_id
        self.user_id = user_id
        self.session_id = session_id

    def get_memory(
        self,
        org_id: str | None = None,
        project_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        group_id: str | None = None,
        session_id: str | None = None,
    ):
        """Get or create a memory instance for the specified context.

        Args:
            org_id: Organization ID (overrides default)
            project_id: Project ID (overrides default)
            user_id: User ID (overrides default, stored in metadata)
            agent_id: Agent ID (overrides default, stored in metadata)
            group_id: Group ID (overrides default, stored in metadata)
            session_id: Session ID (overrides default, stored in metadata)

        Returns:
            Memory instance

        """
        # Get or create project
        try:
            project = self.client.get_project(
                org_id=org_id or self.org_id,
                project_id=project_id or self.project_id,
            )
        except requests.RequestException:
            # Project doesn't exist, create it
            project = self.client.create_project(
                org_id=org_id or self.org_id,
                project_id=project_id or self.project_id,
            )

        return project.memory(
            group_id=group_id or self.group_id,
            agent_id=agent_id or self.agent_id,
            user_id=user_id or self.user_id,
            session_id=session_id or self.session_id,
        )

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
        episode_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a memory to MemMachine.

        This tool stores important information about the user or conversation into memory.
        Use this automatically whenever the user shares new facts, preferences, plans,
        emotions, or other details that could be useful for future context.

        Args:
            content: The content to store in memory. Should include full conversation context.
            role: Message role - "user", "assistant", or "system" (default: "user")
            org_id: Organization ID (overrides default)
            project_id: Project ID (overrides default)
            user_id: User ID (overrides default, stored in metadata)
            agent_id: Agent ID (overrides default, stored in metadata)
            group_id: Group ID (overrides default, stored in metadata)
            session_id: Session ID (overrides default, stored in metadata)
            episode_type: Type of episode (default: "text", stored in metadata)
            metadata: Additional metadata for the episode

        Returns:
            Dictionary with success status and message

        """
        try:
            memory = self.get_memory(
                org_id, project_id, user_id, agent_id, group_id, session_id
            )
            success = memory.add(
                content=content,
                role=role,
                episode_type=episode_type,
                metadata=metadata or {},
            )
            if success:
                return {
                    "status": "success",
                    "message": f"Memory added successfully: {content[:50]}...",
                    "content": content,
                }
            return {
                "status": "error",
                "message": "Failed to add memory",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error adding memory: {e!s}",
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
        limit: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search for memories in MemMachine.

        This tool retrieves relevant context, memories or profile for a user whenever
        context is missing or unclear. Use this whenever you need to recall what has been
        previously discussed, even if it was from an earlier conversation or session.
        This searches both profile memory (long-term user traits and facts) and episodic
        memory (past conversations and experiences).

        Args:
            query: Search query string
            org_id: Organization ID (overrides default)
            project_id: Project ID (overrides default)
            user_id: User ID (overrides default, stored in metadata)
            agent_id: Agent ID (overrides default, stored in metadata)
            group_id: Group ID (overrides default, stored in metadata)
            session_id: Session ID (overrides default, stored in metadata)
            limit: Maximum number of results to return (default: 5)
            filter_dict: Additional filters for the search

        Returns:
            Dictionary containing search results and relevant memories

        """
        try:
            memory = self.get_memory(
                org_id, project_id, user_id, agent_id, group_id, session_id
            )
            results = memory.search(
                query=query,
                limit=limit,
                filter_dict=filter_dict,
            )

            # Format results for easier consumption
            # v2 API returns "semantic_memory" instead of "profile_memory"
            formatted_results = {
                "query": query,
                "episodic_memory": [],
                "profile_memory": [],
            }

            if results:
                # Extract episodic memories
                if results.get("episodic_memory"):
                    episodic = results["episodic_memory"]
                    if isinstance(episodic, list) and len(episodic) > 0:
                        if isinstance(episodic[0], list):
                            formatted_results["episodic_memory"] = episodic[0]
                        else:
                            formatted_results["episodic_memory"] = episodic

                # Extract profile/semantic memories (v2 API uses "semantic_memory")
                if "profile_memory" in results:
                    formatted_results["profile_memory"] = results["profile_memory"]
                elif "semantic_memory" in results:
                    # Map semantic_memory to profile_memory for backward compatibility
                    formatted_results["profile_memory"] = results["semantic_memory"]

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
        """Format search results into a readable summary.

        Args:
            results: Search results dictionary

        Returns:
            Formatted summary string

        """
        summary_parts = []

        episodic_memories = results.get("episodic_memory", [])
        if episodic_memories:
            summary_parts.append(f"Found {len(episodic_memories)} episodic memories:")
            for i, mem in enumerate(episodic_memories[:3], 1):  # Show top 3
                content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
                summary_parts.append(f"  {i}. {content[:100]}...")

        profile_memories = results.get("profile_memory", [])
        if profile_memories:
            summary_parts.append(f"Found {len(profile_memories)} profile memories")

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
        """Get the current memory context.

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
def create_add_memory_tool(tools: MemMachineTools):
    """Create an add_memory tool function for LangGraph.

    Args:
        tools: MemMachineTools instance

    Returns:
        Tool function that can be used in LangGraph

    """

    def add_memory_tool(
        content: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Tool to add a memory to MemMachine.

        Args:
            content: The content to store in memory
            user_id: Optional user ID override
            metadata: Optional metadata for the memory

        Returns:
            Result dictionary with status and message

        """
        return tools.add_memory(
            content=content,
            user_id=user_id,
            metadata=metadata,
        )

    return add_memory_tool


def create_search_memory_tool(tools: MemMachineTools):
    """Create a search_memory tool function for LangGraph.

    Args:
        tools: MemMachineTools instance

    Returns:
        Tool function that can be used in LangGraph

    """

    def search_memory_tool(
        query: str,
        user_id: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Tool to search memories in MemMachine.

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
