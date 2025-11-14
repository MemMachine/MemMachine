from typing import Any

from memmachine import MemMachineClient


class MemMachineTools:
    """
    Tools for integrating MemMachine memory operations into LangGraph.

    This class provides static methods that can be used as tools in LangGraph workflows.
    """

    def __init__(
        self,
        client: MemMachineClient | None = None,
        base_url: str = "http://localhost:8080",
        group_id: str = "langgraph_group",
        agent_id: str = "langgraph_agent",
        user_id: str = "default_user",
        session_id: str | None = None,
    ):
        """
        Initialize MemMachine tools.

        Args:
            client: Optional MemMachineClient instance. If not provided, creates a new one.
            base_url: Base URL for MemMachine server
            group_id: Default group ID for memory operations
            agent_id: Default agent ID for memory operations
            user_id: Default user ID for memory operations
            session_id: Optional session ID. If not provided, will be auto-generated.
        """
        self.client = client or MemMachineClient(base_url=base_url)
        self.group_id = group_id
        self.agent_id = agent_id
        self.user_id = user_id
        self.session_id = session_id

    def get_memory(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        group_id: str | None = None,
        session_id: str | None = None,
    ):
        """
        Get or create a memory instance for the specified context.

        Args:
            user_id: User ID (overrides default)
            agent_id: Agent ID (overrides default)
            group_id: Group ID (overrides default)
            session_id: Session ID (overrides default)

        Returns:
            Memory instance
        """
        return self.client.memory(
            group_id=group_id or self.group_id,
            agent_id=agent_id or self.agent_id,
            user_id=user_id or self.user_id,
            session_id=session_id or self.session_id,
        )

    def add_memory(
        self,
        content: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        group_id: str | None = None,
        session_id: str | None = None,
        episode_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Add a memory to MemMachine.

        This tool stores important information about the user or conversation into memory.
        Use this automatically whenever the user shares new facts, preferences, plans,
        emotions, or other details that could be useful for future context.

        Args:
            content: The content to store in memory. Should include full conversation context.
            user_id: User ID (overrides default)
            agent_id: Agent ID (overrides default)
            group_id: Group ID (overrides default)
            session_id: Session ID (overrides default)
            episode_type: Type of episode (default: "text")
            metadata: Additional metadata for the episode

        Returns:
            Dictionary with success status and message
        """
        try:
            memory = self.get_memory(user_id, agent_id, group_id, session_id)
            success = memory.add(
                content=content,
                episode_type=episode_type,
                metadata=metadata or {},
            )
            if success:
                return {
                    "status": "success",
                    "message": f"Memory added successfully: {content[:50]}...",
                    "content": content,
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to add memory",
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error adding memory: {str(e)}",
            }

    def search_memory(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        group_id: str | None = None,
        session_id: str | None = None,
        limit: int = 5,
        filter_dict: dict[str, Any] | None = None,
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
            user_id: User ID (overrides default)
            agent_id: Agent ID (overrides default)
            group_id: Group ID (overrides default)
            session_id: Session ID (overrides default)
            limit: Maximum number of results to return (default: 5)
            filter_dict: Additional filters for the search

        Returns:
            Dictionary containing search results and relevant memories
        """
        try:
            memory = self.get_memory(user_id, agent_id, group_id, session_id)
            results = memory.search(
                query=query,
                limit=limit,
                filter_dict=filter_dict,
            )

            # Format results for easier consumption
            formatted_results = {
                "query": query,
                "episodic_memory": [],
                "profile_memory": [],
            }

            if results:
                # Extract episodic memories
                if "episodic_memory" in results and results["episodic_memory"]:
                    episodic = results["episodic_memory"]
                    if isinstance(episodic, list) and len(episodic) > 0:
                        if isinstance(episodic[0], list):
                            formatted_results["episodic_memory"] = episodic[0]
                        else:
                            formatted_results["episodic_memory"] = episodic

                # Extract profile memories
                if "profile_memory" in results:
                    formatted_results["profile_memory"] = results["profile_memory"]

            return {
                "status": "success",
                "results": formatted_results,
                "summary": self._format_search_summary(formatted_results),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error searching memory: {str(e)}",
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
        user_id: str | None = None,
        agent_id: str | None = None,
        group_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get the current memory context.

        Args:
            user_id: User ID (overrides default)
            agent_id: Agent ID (overrides default)
            group_id: Group ID (overrides default)
            session_id: Session ID (overrides default)

        Returns:
            Dictionary containing context information
        """
        memory = self.get_memory(user_id, agent_id, group_id, session_id)
        return memory.get_context()

    def close(self):
        """Close the client and clean up resources."""
        if self.client:
            self.client.close()


# Convenience functions for LangGraph tool creation
def create_add_memory_tool(tools: MemMachineTools):
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
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Tool to add a memory to MemMachine.

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
        limit: int = 5,
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
