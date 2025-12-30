"""
Tools for integrating MemMachine memory operations into CrewAI.

This module provides tools that can be used in CrewAI agents
to enable AI agents with persistent memory capabilities.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from memmachine import MemMachineClient
from memmachine.common.api import EpisodeType

# Try to import CrewAI tools, fallback to function-based tools if not available
try:
    from crewai.tools import BaseTool

    CREWAI_AVAILABLE = True
except ImportError:
    try:
        import crewai_tools  # noqa: F401

        CREWAI_AVAILABLE = True
        BaseTool = None  # Use decorator-based approach
    except ImportError:
        CREWAI_AVAILABLE = False
        BaseTool = None

if TYPE_CHECKING:
    from memmachine.rest_client.memory import Memory


class MemMachineTools:
    """
    Tools for integrating MemMachine memory operations into CrewAI.

    This class provides methods that can be used as tools in CrewAI agents.
    """

    def __init__(
        self,
        client: MemMachineClient | None = None,
        base_url: str = "http://localhost:8080",
        org_id: str = "crewai_org",
        project_id: str = "crewai_project",
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
    ) -> "Memory":
        """
        Get or create a memory instance for the specified context.

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
        project = self.client.get_or_create_project(
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
        episode_type: EpisodeType | None = EpisodeType.MESSAGE,
        metadata: dict[str, Any] | None = None,
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
            user_id: User ID (overrides default, stored in metadata)
            agent_id: Agent ID (overrides default, stored in metadata)
            group_id: Group ID (overrides default, stored in metadata)
            session_id: Session ID (overrides default, stored in metadata)
            episode_type: Type of episode (default: EpisodeType.MESSAGE)
            metadata: Additional metadata for the episode

        Returns:
            Dictionary with success status and message

        """
        try:
            memory = self.get_memory(
                org_id, project_id, user_id, agent_id, group_id, session_id
            )
            # memory.add() returns list[AddMemoryResult]
            results = memory.add(
                content=content,
                role=role,
                episode_type=episode_type,
                metadata=metadata or {},
            )
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error adding memory: {e!s}",
            }
        else:
            # Check if results list is not empty
            if results:
                # Extract UIDs from AddMemoryResult objects
                uids = [result.uid for result in results if hasattr(result, "uid")]
                return {
                    "status": "success",
                    "message": f"Memory added successfully: {content[:50]}...",
                    "content": content,
                    "uids": uids,
                }
            return {
                "status": "error",
                "message": "Failed to add memory: no results returned",
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
            search_result = memory.search(
                query=query,
                limit=limit,
                filter_dict=filter_dict,
            )

            # Format results for easier consumption
            # SearchResult is a Pydantic model with 'content' attribute
            # v2 API returns "semantic_memory" instead of "profile_memory"
            formatted_results = {
                "query": query,
                "episodic_memory": [],
                "profile_memory": [],
            }

            # Access the content dictionary from SearchResult
            results = search_result.content if hasattr(search_result, "content") else {}

            if results:
                # Extract episodic memories
                if results.get("episodic_memory"):
                    episodic = results["episodic_memory"]
                    if isinstance(episodic, list) and episodic:
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


def create_memmachine_tools(
    base_url: str = "http://localhost:8080",
    org_id: str = "crewai_org",
    project_id: str = "crewai_project",
    group_id: str | None = None,
    agent_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> list[Any]:
    """
    Create CrewAI tools for MemMachine memory operations.

    This function creates CrewAI-compatible tools that can be used by agents
    to add and search memories.

    Args:
        base_url: Base URL for MemMachine server
        org_id: Organization ID for v2 API
        project_id: Project ID for v2 API
        group_id: Optional group ID (stored in metadata)
        agent_id: Optional agent ID (stored in metadata)
        user_id: Optional user ID (stored in metadata)
        session_id: Optional session ID (stored in metadata)

    Returns:
        List of CrewAI Tool objects

    """
    tools_instance = MemMachineTools(
        base_url=base_url,
        org_id=org_id,
        project_id=project_id,
        group_id=group_id,
        agent_id=agent_id,
        user_id=user_id,
        session_id=session_id,
    )

    # Try to use BaseTool if available (CrewAI native)
    if BaseTool is not None:
        return _create_basetool_tools(tools_instance)

    # Fallback to function-based tools (crewai_tools decorator)
    try:
        from crewai_tools import tool

        return _create_decorator_tools(tools_instance, tool)
    except ImportError:
        return _create_simple_function_tools(tools_instance)


def _create_basetool_tools(tools_instance: MemMachineTools) -> list[Any]:
    """
    Create BaseTool-based tools for CrewAI.

    Args:
        tools_instance: MemMachineTools instance

    Returns:
        List of BaseTool instances

    """

    class AddMemoryTool(BaseTool):
        name: str = "Add Memory to MemMachine"
        description: str = (
            "Add a memory to MemMachine. Use this to store important information, "
            "facts, preferences, or conversation context that should be remembered for future interactions."
        )

        def _run(self, content: str, role: str = "user") -> str:
            result = tools_instance.add_memory(content=content, role=role)
            if result["status"] == "success":
                return result["message"]
            return f"Error: {result.get('message', 'Unknown error')}"

    class SearchMemoryTool(BaseTool):
        name: str = "Search Memory in MemMachine"
        description: str = (
            "Search for memories in MemMachine. Use this to retrieve relevant context, "
            "past conversations, or user preferences when you need to recall information from previous interactions."
        )

        def _run(self, query: str, limit: int = 5) -> str:
            result = tools_instance.search_memory(query=query, limit=limit)
            if result["status"] == "success":
                return result.get("summary", "No memories found")
            return f"Error: {result.get('message', 'Unknown error')}"

    return [AddMemoryTool(), SearchMemoryTool()]


def _create_decorator_tools(
    tools_instance: MemMachineTools,
    tool: Callable[..., Any],  # type: ignore[type-arg]
) -> list[Callable[..., Any]]:
    """
    Create decorator-based tools using crewai_tools.

    Args:
        tools_instance: MemMachineTools instance
        tool: The tool decorator from crewai_tools

    Returns:
        List of decorated tool functions

    """

    @tool("Add Memory to MemMachine")
    def add_memory_tool(content: str, role: str = "user") -> str:
        """
        Add a memory to MemMachine.

        Use this to store important information, facts, preferences, or conversation context that should be remembered for future interactions.

        Args:
            content: The content to store in memory. Include full context and important details.
            role: Message role - "user", "assistant", or "system" (default: "user")

        Returns:
            Status message indicating success or failure

        """
        result = tools_instance.add_memory(content=content, role=role)
        if result["status"] == "success":
            return result["message"]
        return f"Error: {result.get('message', 'Unknown error')}"

    @tool("Search Memory in MemMachine")
    def search_memory_tool(query: str, limit: int = 5) -> str:
        """
        Search for memories in MemMachine.

        Use this to retrieve relevant context, past conversations, or user preferences when you need to recall information from previous interactions.

        Args:
            query: Search query string describing what you're looking for
            limit: Maximum number of results to return (default: 5)

        Returns:
            Formatted summary of relevant memories found

        """
        result = tools_instance.search_memory(query=query, limit=limit)
        if result["status"] == "success":
            return result.get("summary", "No memories found")
        return f"Error: {result.get('message', 'Unknown error')}"

    return [add_memory_tool, search_memory_tool]


def _create_simple_function_tools(tools_instance: MemMachineTools) -> list[Any]:
    """
    Create simple function-based tools when CrewAI tools are not available.

    Args:
        tools_instance: MemMachineTools instance

    Returns:
        List of function-based tools

    """

    def add_memory_tool(content: str, role: str = "user") -> str:
        """Add a memory to MemMachine."""
        result = tools_instance.add_memory(content=content, role=role)
        if result["status"] == "success":
            return result["message"]
        return f"Error: {result.get('message', 'Unknown error')}"

    def search_memory_tool(query: str, limit: int = 5) -> str:
        """Search for memories in MemMachine."""
        result = tools_instance.search_memory(query=query, limit=limit)
        if result["status"] == "success":
            return result.get("summary", "No memories found")
        return f"Error: {result.get('message', 'Unknown error')}"

    # Add metadata for CrewAI
    add_memory_tool.name = "Add Memory to MemMachine"
    add_memory_tool.description = (
        "Add a memory to MemMachine. Use this to store important information, "
        "facts, preferences, or conversation context that should be remembered for future interactions."
    )
    search_memory_tool.name = "Search Memory in MemMachine"
    search_memory_tool.description = (
        "Search for memories in MemMachine. Use this to retrieve relevant context, "
        "past conversations, or user preferences when you need to recall information from previous interactions."
    )

    return [add_memory_tool, search_memory_tool]
