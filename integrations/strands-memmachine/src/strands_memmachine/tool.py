"""
Memory management tool for the MemMachine Platform.

Provides persistent episodic and semantic memory capabilities for Strands agents
via the MemMachine REST API. Supports storing, searching, listing, and deleting
memories with metadata filtering and pagination. Also supports full project
management — create, retrieve, list, and delete isolated memory namespaces.

Configuration:
    MEMMACHINE_API_KEY (required): Bearer token for MemMachine API authentication.
    MEMMACHINE_BASE_URL (optional): API base URL. Defaults to https://api.memmachine.ai.
                                    Set to a custom URL for self-hosted deployments.

Usage:
    from strands import Agent
    from strands_memmachine import memmachine_memory

    agent = Agent(tools=[memmachine_memory])

    agent.tool.memmachine_memory(
        action="store",
        content="User prefers aisle seats on flights",
        metadata={"category": "travel", "user_id": "alice"},
    )

    agent.tool.memmachine_memory(
        action="search",
        query="flight preferences",
        top_k=5,
    )

    agent.tool.memmachine_memory(
        action="list",
        filter="metadata.user_id=alice AND metadata.category=travel",
        page_size=20,
    )

    agent.tool.memmachine_memory(
        action="delete",
        memory_type="episodic",
        memory_id="mem-123",
    )

    agent.tool.memmachine_memory(
        action="create_project",
        project_id="my-project",
        description="Project for travel preferences",
    )

    agent.tool.memmachine_memory(
        action="get_project",
        project_id="my-project",
    )

    agent.tool.memmachine_memory(
        action="list_projects",
    )

    agent.tool.memmachine_memory(
        action="get_episode_count",
        project_id="my-project",
    )

    agent.tool.memmachine_memory(
        action="delete_project",
        project_id="my-project",
    )
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from strands.types.tools import ToolResult, ToolResultContent, ToolUse

logger = logging.getLogger(__name__)

console = Console()

TOOL_SPEC = {
    "name": "memmachine_memory",
    "description": (
        "Memory management tool for storing, searching, and managing memories using MemMachine.\n\n"
        "MemMachine provides a persistent memory layer for AI agents with episodic (conversational)\n"
        "and semantic (factual) memory types. Memory is organized into Projects — isolated namespaces\n"
        "that keep memories separated by use case, user, or application.\n\n"
        "Memory Actions:\n"
        "- store: Store new memory messages with optional metadata, producer, and memory type selection.\n"
        "- search: Semantic search across episodic and/or semantic memories.\n"
        "- list: List memories with pagination and optional metadata filtering.\n"
        "- delete: Delete episodic or semantic memories by ID (single or bulk).\n\n"
        "Project Actions:\n"
        "- create_project: Create a new isolated memory namespace.\n"
        "- delete_project: Delete a project and all its associated memories permanently.\n"
        "- get_project: Retrieve a project by its identifier.\n"
        "- list_projects: List all projects for the authenticated user.\n"
        "- get_episode_count: Retrieve the total number of episodes recorded for a project.\n\n"
        "Metadata filter syntax (used in 'search' and 'list'):\n"
        "  Single condition : metadata.key=value\n"
        "  Multiple conditions: metadata.key1=value1 AND metadata.key2=value2\n"
        "  Example: metadata.user_id=alice AND metadata.category=travel\n\n"
        "Configuration (environment variables):\n"
        "  MEMMACHINE_API_KEY (required): API key from https://console.memmachine.ai\n"
        "  MEMMACHINE_BASE_URL (optional): Override API base URL for self-hosted instances.\n"
        "                                  Defaults to https://api.memmachine.ai"
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform.",
                    "enum": [
                        "store",
                        "search",
                        "list",
                        "delete",
                        "create_project",
                        "delete_project",
                        "get_project",
                        "list_projects",
                        "get_episode_count",
                    ],
                },
                "content": {
                    "type": "string",
                    "description": "Memory content to store. Required for the 'store' action.",
                },
                "query": {
                    "type": "string",
                    "description": "Natural language search query. Required for the 'search' action.",
                },
                "memory_id": {
                    "type": "string",
                    "description": "Single memory ID to delete. Required for 'delete' when memory_ids is not provided.",
                },
                "memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of memory IDs for bulk deletion. "
                        "Required for 'delete' when memory_id is not provided. "
                        'Example: ["mem-1", "mem-2", "mem-3"]'
                    ),
                },
                "memory_type": {
                    "type": "string",
                    "description": (
                        "Memory type to target. Required for 'delete'. "
                        "Optional for 'list' to filter by type. "
                        "Must be 'episodic' or 'semantic'."
                    ),
                    "enum": ["episodic", "semantic"],
                },
                "types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["episodic", "semantic"]},
                    "description": (
                        "Memory types to write to or search across for 'store' and 'search' actions. "
                        "Defaults to both episodic and semantic when omitted."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results returned by 'search'. Default: 10.",
                },
                "page_size": {
                    "type": "integer",
                    "description": "Number of memories per page for 'list'. Default: 100.",
                },
                "page_num": {
                    "type": "integer",
                    "description": "Zero-based page number for 'list'. Default: 0.",
                },
                "filter": {
                    "type": "string",
                    "description": (
                        "Metadata filter expression for 'search' and 'list' actions. "
                        "Format: metadata.key=value\n"
                        "Multiple conditions joined with AND: "
                        "metadata.user_id=alice AND metadata.category=travel"
                    ),
                },
                "producer": {
                    "type": "string",
                    "description": "Identity of the message producer for 'store'. Default: 'user'.",
                },
                "produced_for": {
                    "type": "string",
                    "description": "Intended recipient of the stored memory for 'store'.",
                },
                "metadata": {
                    "type": "object",
                    "description": (
                        "Arbitrary key-value pairs stored alongside the memory. "
                        "All values are coerced to strings. "
                        'Example: {"user_id": "alice", "category": "travel"}'
                    ),
                },
                "project_id": {
                    "type": "string",
                    "description": (
                        "Project identifier. Acts as an isolated memory namespace. "
                        "Required for create_project, delete_project, get_project, and get_episode_count. "
                        "If empty, the user's default project is used. "
                        "Allowed characters: letters, numbers, underscores, hyphens, colons, and Unicode."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Optional human-readable description of the project. Used in create_project.",
                },
            },
            "required": ["action"],
        }
    },
}


class MemMachineClient:
    """HTTP client for the MemMachine Platform REST API.

    Communicates with the MemMachine API using Bearer token authentication.
    Reads credentials and endpoint configuration from environment variables.

    Environment Variables:
        MEMMACHINE_API_KEY: Required. Bearer token for authentication.
        MEMMACHINE_BASE_URL: Optional. Defaults to https://api.memmachine.ai.
    """

    def __init__(self) -> None:
        """Initialize the MemMachine client.

        Raises:
            ValueError: If MEMMACHINE_API_KEY is not set in the environment.
        """
        self.api_key = os.environ.get("MEMMACHINE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MEMMACHINE_API_KEY environment variable is required. "
                "Obtain your API key from https://console.memmachine.ai"
            )

        self.base_url = os.environ.get("MEMMACHINE_BASE_URL", "https://api.memmachine.ai").rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        logger.debug("MemMachineClient initialized with base_url=%s", self.base_url)

    def _request(self, method: str, path: str, json_data: Optional[Dict] = None) -> requests.Response:
        """Execute an authenticated HTTP request against the MemMachine API.

        Args:
            method: HTTP method (e.g., 'GET', 'POST').
            path: API endpoint path relative to base_url (e.g., '/v2/memories').
            json_data: Optional JSON-serializable payload for the request body.

        Returns:
            The HTTP response object from the API.

        Raises:
            requests.HTTPError: If the API returns a non-2xx HTTP status code.
        """
        url = f"{self.base_url}{path}"
        logger.debug("MemMachine API %s %s", method, url)
        response = self.session.request(method, url, json=json_data, timeout=60)
        response.raise_for_status()
        return response

    def store_memory(
        self,
        content: str,
        types: Optional[List[str]] = None,
        producer: str = "user",
        produced_for: str = "",
        metadata: Optional[Dict] = None,
    ) -> Any:
        """Store a memory entry in MemMachine.

        Args:
            content: The text content of the memory.
            types: Memory types to write to (e.g., ['episodic', 'semantic']).
                   Defaults to both when omitted.
            producer: Identity label for the message producer. Default: 'user'.
            produced_for: Intended recipient of the memory.
            metadata: Arbitrary key-value pairs to attach to the memory.
                      All values are coerced to strings before storage.

        Returns:
            API response dict containing a 'results' list with UIDs of stored memories.
        """
        message: Dict[str, Any] = {"content": content, "producer": producer}

        if produced_for:
            message["produced_for"] = produced_for

        if metadata:
            message["metadata"] = {k: str(v) for k, v in metadata.items()}

        payload: Dict[str, Any] = {"messages": [message]}

        if types:
            payload["types"] = types

        return self._request("POST", "/v2/memories", json_data=payload).json()

    def search_memories(
        self,
        query: str,
        top_k: int = 10,
        types: Optional[List[str]] = None,
        filter_str: Optional[str] = None,
    ) -> Any:
        """Perform a semantic search across stored memories.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return. Default: 10.
            types: Memory types to search (e.g., ['episodic', 'semantic']).
                   Defaults to both when omitted.
            filter_str: Metadata filter expression.
                        Format: 'metadata.key=value AND metadata.key2=value2'.

        Returns:
            API response dict containing search results under a 'content' key.
        """
        payload: Dict[str, Any] = {"query": query, "top_k": top_k}

        if types:
            payload["types"] = types

        if filter_str:
            payload["filter"] = filter_str

        return self._request("POST", "/v2/memories/search", json_data=payload).json()

    def list_memories(
        self,
        page_size: int = 100,
        page_num: int = 0,
        memory_type: Optional[str] = None,
        filter_str: Optional[str] = None,
    ) -> Any:
        """List stored memories with pagination and optional filtering.

        Args:
            page_size: Number of memories to return per page. Default: 100.
            page_num: Zero-based page index. Default: 0.
            memory_type: Restrict results to a specific type ('episodic' or 'semantic').
                         Returns both types when omitted.
            filter_str: Metadata filter expression.
                        Format: 'metadata.key=value AND metadata.key2=value2'.

        Returns:
            API response dict containing listed memories under a 'content' key.
        """
        payload: Dict[str, Any] = {"page_size": page_size, "page_num": page_num}

        if memory_type:
            payload["type"] = memory_type

        if filter_str:
            payload["filter"] = filter_str

        return self._request("POST", "/v2/memories/list", json_data=payload).json()

    def delete_episodic_memory(
        self,
        memory_id: Optional[str] = None,
        memory_ids: Optional[List[str]] = None,
    ) -> None:
        """Delete one or more episodic memories by ID.

        Args:
            memory_id: Single episodic memory UID to delete.
            memory_ids: List of episodic memory UIDs for bulk deletion.

        Raises:
            ValueError: If neither memory_id nor memory_ids is provided.
        """
        if not memory_id and not memory_ids:
            raise ValueError("Either memory_id or memory_ids must be provided for episodic deletion.")

        payload: Dict[str, Any] = {}

        if memory_id:
            payload["episodic_id"] = memory_id

        if memory_ids:
            payload["episodic_ids"] = memory_ids

        self._request("POST", "/v2/memories/episodic/delete", json_data=payload)

    def delete_semantic_memory(
        self,
        memory_id: Optional[str] = None,
        memory_ids: Optional[List[str]] = None,
    ) -> None:
        """Delete one or more semantic memories by ID.

        Args:
            memory_id: Single semantic memory UID to delete.
            memory_ids: List of semantic memory UIDs for bulk deletion.

        Raises:
            ValueError: If neither memory_id nor memory_ids is provided.
        """
        if not memory_id and not memory_ids:
            raise ValueError("Either memory_id or memory_ids must be provided for semantic deletion.")

        payload: Dict[str, Any] = {}

        if memory_id:
            payload["semantic_id"] = memory_id

        if memory_ids:
            payload["semantic_ids"] = memory_ids

        self._request("POST", "/v2/memories/semantic/delete", json_data=payload)

    def create_project(
        self,
        project_id: str,
        description: str = "",
    ) -> Any:
        """Create a new project in MemMachine.

        Each project acts as an isolated memory namespace. All memories inserted
        into a project belong exclusively to that project.

        Args:
            project_id: Unique identifier for the project. Allowed characters:
                        letters, numbers, underscores, hyphens, colons, and Unicode.
            description: Optional human-readable description of the project.

        Returns:
            API response dict containing the fully resolved project record.
        """
        payload: Dict[str, Any] = {"project_id": project_id}

        if description:
            payload["description"] = description

        return self._request("POST", "/v2/projects", json_data=payload).json()

    def delete_project(self, project_id: str) -> Any:
        """Delete a project and all its associated memories permanently.

        Args:
            project_id: Identifier of the project to delete.

        Returns:
            API response dict confirming deletion.
        """
        return self._request("POST", "/v2/projects/delete", json_data={"project_id": project_id}).json()

    def get_project(self, project_id: str) -> Any:
        """Retrieve a project by its identifier.

        Args:
            project_id: Identifier of the project to retrieve.

        Returns:
            API response dict containing the project record and configuration.
        """
        return self._request("POST", "/v2/projects/get", json_data={"project_id": project_id}).json()

    def list_projects(self) -> Any:
        """List all projects for the authenticated user.

        Returns:
            API response containing an array of project records.
        """
        return self._request("POST", "/v2/projects/list", json_data={}).json()

    def get_episode_count(self, project_id: str) -> Any:
        """Retrieve the total number of episodes recorded for a project.

        Args:
            project_id: Identifier of the project to query.

        Returns:
            API response dict containing the episode count.
        """
        return self._request("POST", "/v2/projects/episode_count/get", json_data={"project_id": project_id}).json()


def _extract_items_from_node(data: Any) -> List[Dict]:
    """Extract a flat list of dict items from a potentially nested API response node.

    Handles the following response structures:
    - Direct list: [{"uid": "..."}, ...]
    - Dict with a known list key: {"episodes": [...]} or {"memories": [...]}
    - Nested dict one level deep: {"long_term_memory": {"episodes": [...]}}

    Args:
        data: The data node to extract items from.

    Returns:
        A list of dict items found in the node. Returns an empty list if none are found.
    """
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if not isinstance(data, dict):
        return []

    known_list_keys = ("episodes", "memories", "results", "items")

    for key in known_list_keys:
        if key in data and isinstance(data[key], list):
            return [item for item in data[key] if isinstance(item, dict)]

    for val in data.values():
        if isinstance(val, dict):
            for key in known_list_keys:
                if key in val and isinstance(val[key], list):
                    return [item for item in val[key] if isinstance(item, dict)]

    return []


def _extract_memory_entries(content: Dict) -> List[Dict]:
    """Extract a flat list of memory entries from an API response content block.

    Args:
        content: The 'content' dict from a MemMachine API response.

    Returns:
        A flat list of memory entry dicts, each tagged with a '_type' key.
    """
    entries: List[Dict] = []
    type_key_map = {
        "episodic_memory": "episodic",
        "semantic_memory": "semantic",
    }

    for response_key, type_label in type_key_map.items():
        data = content.get(response_key)
        if data is None:
            continue

        for item in _extract_items_from_node(data):
            entries.append({**item, "_type": type_label})

    return entries


def _format_store_response(results: List[Dict]) -> Panel:
    """Render a Rich panel summarising stored memory UIDs.

    Args:
        results: List of result dicts from the store API response.

    Returns:
        A Rich Panel containing a table of stored memory UIDs.
    """
    if not results:
        return Panel("No memories were stored.", title="[bold yellow]No Memories Stored", border_style="yellow")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim")
    table.add_column("UID", style="cyan")

    for i, result in enumerate(results, 1):
        table.add_row(str(i), result.get("uid", "unknown"))

    return Panel(table, title="[bold green]Memories Stored Successfully", border_style="green")


def _format_search_response(response_data: Dict) -> Panel:
    """Render a Rich panel displaying semantic search results.

    Args:
        response_data: Full API response dict from the search endpoint.

    Returns:
        A Rich Panel displaying results as a table, or raw JSON as a fallback.
    """
    content = response_data.get("content", {})
    if not content:
        return Panel(
            "No memories matched the query.",
            title="[bold yellow]No Matches Found",
            border_style="yellow",
        )

    entries = _extract_memory_entries(content)
    if entries:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Type", style="cyan", width=10)
        table.add_column("Content", style="yellow", width=50)
        table.add_column("Score", style="green", width=10)
        table.add_column("Timestamp", style="blue", width=22)

        for entry in entries:
            raw_content = str(entry.get("content", entry.get("memory", "")))
            content_preview = raw_content[:80] + "..." if len(raw_content) > 80 else raw_content

            score = entry.get("score", "N/A")
            if isinstance(score, (int, float)):
                if score > 0.8:
                    score_str = f"[green]{score:.3f}[/green]"
                elif score > 0.5:
                    score_str = f"[yellow]{score:.3f}[/yellow]"
                else:
                    score_str = f"[red]{score:.3f}[/red]"
            else:
                score_str = str(score)

            timestamp = str(entry.get("created_at", entry.get("timestamp", "N/A")))
            table.add_row(entry.get("_type", "N/A"), content_preview, score_str, timestamp)

        return Panel(table, title="[bold green]Search Results", border_style="green")

    formatted = json.dumps(content, indent=2, default=str)
    if len(formatted) > 3000:
        formatted = formatted[:3000] + "\n... (truncated)"

    return Panel(formatted, title="[bold green]Search Results (Raw)", border_style="green")


def _format_list_response(response_data: Dict) -> Panel:
    """Render a Rich panel listing retrieved memories.

    Args:
        response_data: Full API response dict from the list endpoint.

    Returns:
        A Rich Panel displaying memories as a table, or raw JSON as a fallback.
    """
    content = response_data.get("content", {})
    if not content:
        return Panel("No memories found.", title="[bold yellow]No Memories", border_style="yellow")

    entries = _extract_memory_entries(content)
    if entries:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Type", style="cyan", width=10)
        table.add_column("Content", style="yellow", width=45)
        table.add_column("ID", style="green", width=15)
        table.add_column("Timestamp", style="blue", width=22)
        table.add_column("Metadata", style="magenta", width=20)

        for entry in entries:
            raw_content = str(entry.get("content", entry.get("memory", "")))
            content_preview = raw_content[:70] + "..." if len(raw_content) > 70 else raw_content
            mem_id = str(entry.get("uid", entry.get("id", "N/A")))
            timestamp = str(entry.get("created_at", entry.get("timestamp", "N/A")))
            metadata = entry.get("metadata", {})
            metadata_str = json.dumps(metadata) if metadata else "None"

            table.add_row(entry.get("_type", "N/A"), content_preview, mem_id, timestamp, metadata_str)

        return Panel(table, title="[bold green]Memories", border_style="green")

    formatted = json.dumps(content, indent=2, default=str)
    if len(formatted) > 3000:
        formatted = formatted[:3000] + "\n... (truncated)"

    return Panel(formatted, title="[bold green]Memories (Raw)", border_style="green")


def _format_delete_response(
    memory_type: str,
    memory_id: Optional[str] = None,
    memory_ids: Optional[List[str]] = None,
) -> Panel:
    """Render a Rich panel confirming a successful memory deletion.

    Args:
        memory_type: The type of memory deleted ('episodic' or 'semantic').
        memory_id: Single memory UID that was deleted.
        memory_ids: List of memory UIDs that were deleted in a bulk operation.

    Returns:
        A Rich Panel confirming the IDs that were removed.
    """
    ids = [memory_id] if memory_id else (memory_ids or [])
    lines = [f"{memory_type.capitalize()} memory deleted successfully:"] + [f"  UID: {uid}" for uid in ids]

    return Panel(
        "\n".join(lines),
        title=f"[bold green]{memory_type.capitalize()} Memory Deleted",
        border_style="green",
    )


def memmachine_memory(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands tool for managing persistent agent memory via the MemMachine Platform API.

    Supports memory operations (store, search, list, delete) and project management
    (create_project, delete_project, get_project, list_projects, get_episode_count).
    Memory can be classified as episodic (conversational) or semantic (factual), or both.

    Set BYPASS_TOOL_CONSENT=true to suppress confirmation panels in automated
    or test environments.

    Args:
        tool: ToolUse object with the following 'input' fields:
            action (str, required):
                Operation to perform. One of:
                store | search | list | delete |
                create_project | delete_project | get_project | list_projects | get_episode_count.
            content (str):
                Memory text to store. Required when action='store'.
            query (str):
                Natural language search query. Required when action='search'.
            memory_id (str):
                Single UID to delete. Required for delete when memory_ids is absent.
            memory_ids (list[str]):
                List of UIDs for bulk deletion. Required for delete when memory_id is absent.
            memory_type (str):
                Target memory type: 'episodic' or 'semantic'.
                Required for delete. Optional filter for list.
            types (list[str]):
                Memory types to write to or search. Defaults to both when omitted.
            top_k (int):
                Max results for search. Default: 10.
            page_size (int):
                Memories per page for list. Default: 100.
            page_num (int):
                Zero-based page index for list. Default: 0.
            filter (str):
                Metadata filter expression. Format: 'metadata.key=value'.
                Multiple conditions: 'metadata.k1=v1 AND metadata.k2=v2'.
            producer (str):
                Producer identity label for store. Default: 'user'.
            produced_for (str):
                Intended recipient for store.
            metadata (dict):
                Arbitrary key-value pairs to attach to the stored memory.
            project_id (str):
                Project identifier. Required for create_project, delete_project,
                get_project, and get_episode_count.
            description (str):
                Optional project description. Used in create_project.
        **kwargs: Reserved for future Strands tool interface extensions.

    Returns:
        ToolResult with status='success' and a JSON-encoded content payload,
        or status='error' with a descriptive error message.
    """
    tool_use_id = "default-id"

    try:
        tool_input = tool.get("input", {})
        tool_use_id = tool.get("toolUseId", "default-id")

        if not tool_input.get("action"):
            raise ValueError("action parameter is required.")

        client = MemMachineClient()

        bypass_consent = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"
        action = tool_input["action"]
        is_mutative = action in {"store", "delete"}

        if is_mutative and not bypass_consent:
            if action == "store":
                if not tool_input.get("content"):
                    raise ValueError("content is required for the store action.")
                preview = tool_input["content"][:15000]
                if len(tool_input["content"]) > 15000:
                    preview += "..."
                console.print(Panel(preview, title="[bold green]Memory to Store", border_style="green"))

            elif action == "delete":
                mid = tool_input.get("memory_id", "")
                mids = tool_input.get("memory_ids", [])
                ids_display = mid if mid else ", ".join(mids)
                console.print(
                    Panel(
                        f"Memory Type: {tool_input.get('memory_type', 'unknown')}\nMemory ID(s): {ids_display}",
                        title="[bold red]Memory Pending Permanent Deletion",
                        border_style="red",
                    )
                )

        if action == "store":
            if not tool_input.get("content"):
                raise ValueError("content is required for the store action.")

            response = client.store_memory(
                content=tool_input["content"],
                types=tool_input.get("types"),
                producer=tool_input.get("producer", "user"),
                produced_for=tool_input.get("produced_for", ""),
                metadata=tool_input.get("metadata"),
            )
            console.print(_format_store_response(response.get("results", [])))

            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(response, indent=2, default=str))],
            )

        elif action == "search":
            if not tool_input.get("query"):
                raise ValueError("query is required for the search action.")

            response = client.search_memories(
                query=tool_input["query"],
                top_k=tool_input.get("top_k", 10),
                types=tool_input.get("types"),
                filter_str=tool_input.get("filter"),
            )
            console.print(_format_search_response(response))

            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(response, indent=2, default=str))],
            )

        elif action == "list":
            response = client.list_memories(
                page_size=tool_input.get("page_size", 100),
                page_num=tool_input.get("page_num", 0),
                memory_type=tool_input.get("memory_type"),
                filter_str=tool_input.get("filter"),
            )
            console.print(_format_list_response(response))

            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(response, indent=2, default=str))],
            )

        elif action == "delete":
            memory_type = tool_input.get("memory_type")
            if not memory_type:
                raise ValueError("memory_type is required for the delete action ('episodic' or 'semantic').")

            memory_id = tool_input.get("memory_id")
            memory_ids: Optional[List[str]] = tool_input.get("memory_ids")

            if not memory_id and not memory_ids:
                raise ValueError("memory_id or memory_ids is required for the delete action.")

            if memory_type == "episodic":
                client.delete_episodic_memory(memory_id=memory_id, memory_ids=memory_ids)
            elif memory_type == "semantic":
                client.delete_semantic_memory(memory_id=memory_id, memory_ids=memory_ids)
            else:
                raise ValueError(f"Invalid memory_type '{memory_type}'. Must be 'episodic' or 'semantic'.")

            console.print(_format_delete_response(memory_type, memory_id, memory_ids))

            deleted_ids = memory_id or ", ".join(memory_ids or [])
            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[
                    ToolResultContent(text=f"{memory_type.capitalize()} memory deleted successfully: {deleted_ids}")
                ],
            )

        elif action == "create_project":
            project_id = tool_input.get("project_id")
            if not project_id:
                raise ValueError("project_id is required for the create_project action.")

            response = client.create_project(
                project_id=project_id,
                description=tool_input.get("description", ""),
            )
            console.print(
                Panel(
                    f"Project '{project_id}' created successfully.",
                    title="[bold green]Project Created",
                    border_style="green",
                )
            )

            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(response, indent=2, default=str))],
            )

        elif action == "delete_project":
            project_id = tool_input.get("project_id")
            if not project_id:
                raise ValueError("project_id is required for the delete_project action.")

            response = client.delete_project(project_id=project_id)
            console.print(
                Panel(
                    f"Project '{project_id}' and all associated memories deleted permanently.",
                    title="[bold red]Project Deleted",
                    border_style="red",
                )
            )

            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(response, indent=2, default=str))],
            )

        elif action == "get_project":
            project_id = tool_input.get("project_id")
            if not project_id:
                raise ValueError("project_id is required for the get_project action.")

            response = client.get_project(project_id=project_id)
            console.print(
                Panel(
                    json.dumps(response, indent=2, default=str),
                    title="[bold green]Project Details",
                    border_style="green",
                )
            )

            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(response, indent=2, default=str))],
            )

        elif action == "list_projects":
            response = client.list_projects()
            console.print(
                Panel(
                    json.dumps(response, indent=2, default=str),
                    title="[bold green]Projects",
                    border_style="green",
                )
            )

            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(response, indent=2, default=str))],
            )

        elif action == "get_episode_count":
            project_id = tool_input.get("project_id")
            if not project_id:
                raise ValueError("project_id is required for the get_episode_count action.")

            response = client.get_episode_count(project_id=project_id)
            console.print(
                Panel(
                    json.dumps(response, indent=2, default=str),
                    title="[bold green]Episode Count",
                    border_style="green",
                )
            )

            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(response, indent=2, default=str))],
            )

        else:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of: store, search, list, delete, "
                f"create_project, delete_project, get_project, list_projects, get_episode_count."
            )

    except Exception as exc:
        console.print(Panel(Text(str(exc), style="red"), title="Memory Operation Error", border_style="red"))
        return ToolResult(
            toolUseId=tool_use_id,
            status="error",
            content=[ToolResultContent(text=f"Error: {exc}")],
        )
