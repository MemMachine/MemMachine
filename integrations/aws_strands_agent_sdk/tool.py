"""
MemMachine tools for AWS Strands Agent SDK integration.

This module provides tools that can be integrated into AWS Strands Agent SDK
to enable AI agents with persistent memory capabilities.
"""

import logging
from collections.abc import Callable
from typing import Any
from uuid import uuid4

try:
    from memmachine.rest_client import MemMachineClient
except ImportError as e:
    raise ImportError(
        "memmachine package is required. Install it with: pip install memmachine"
    ) from e

logger = logging.getLogger(__name__)


class MemMachineTools:
    """
    Wrapper class for MemMachine client that provides tool-friendly methods.

    This class wraps the MemMachine client and provides methods that return
    structured results suitable for use with agent tool systems.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        org_id: str = "default_org",
        project_id: str = "default_project",
        group_id: str = "default_group",
        agent_id: str = "default_agent",
        user_id: str = "default_user",
        session_id: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Initialize MemMachine tools.

        Args:
            base_url: Base URL of the MemMachine server
            api_key: API key for authentication (optional)
            org_id: Organization identifier (required for v2 API)
            project_id: Project identifier (required for v2 API)
            group_id: Group identifier for the memory context
            agent_id: Agent identifier for the memory context
            user_id: User identifier for the memory context
            session_id: Session identifier (auto-generated if not provided)
            **kwargs: Additional configuration options

        """
        self.client = MemMachineClient(base_url=base_url, api_key=api_key, **kwargs)
        self.org_id = org_id
        self.project_id = project_id
        self.group_id = group_id
        self.agent_id = agent_id
        self.user_id = user_id
        self.session_id = session_id or f"session_{uuid4().hex}"

        # Get or create project (v2 API)
        self.project = self.client.get_or_create_project(
            org_id=self.org_id,
            project_id=self.project_id,
            description=f"Project for {self.group_id}/{self.agent_id}",
        )

        # Create memory instance from project (v2 API)
        self.memory = self.project.memory(
            group_id=self.group_id,
            agent_id=self.agent_id,
            user_id=self.user_id,
            session_id=self.session_id,
        )

    def add_memory(
        self,
        content: str,
        user_id: str | None = None,
        episode_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store important information about the user or conversation into memory.

        Args:
            content: The content to store in memory
            user_id: User ID override (uses instance user_id if not provided)
            episode_type: Type of episode (default: "text")
            metadata: Additional metadata dictionary

        Returns:
            Dictionary with status, message, and content fields

        """
        try:
            # If user_id is different, create a new memory instance
            if user_id and user_id != self.user_id:
                memory = self.project.memory(
                    group_id=self.group_id,
                    agent_id=self.agent_id,
                    user_id=user_id,
                    session_id=self.session_id,
                )
            else:
                memory = self.memory

            success = memory.add(
                content=content,
                episode_type=episode_type,
                metadata=metadata or {},
            )

            if success:
                return {
                    "status": "success",
                    "message": "Memory added successfully",
                    "content": content,
                }
        except Exception:
            logger.exception("Error adding memory")
            return {
                "status": "error",
                "message": "Error adding memory",
                "content": content,
            }
        else:
            return {
                "status": "error",
                "message": "Failed to add memory",
                "content": content,
            }
            return {
                "status": "error",
                "message": "Error adding memory",
                "content": content,
            }

    def search_memory(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve relevant context, memories, or profile for a user.

        Args:
            query: Search query string
            user_id: User ID override (uses instance user_id if not provided)
            limit: Maximum number of results (default: 5, max: 20)
            score_threshold: Minimum score to include in results

        Returns:
            Dictionary with status, results, and summary fields

        """
        try:
            # Clamp limit to reasonable range
            limit = max(1, min(limit, 20))

            # If user_id is different, create a new memory instance
            if user_id and user_id != self.user_id:
                memory = self.project.memory(
                    group_id=self.group_id,
                    agent_id=self.agent_id,
                    user_id=user_id,
                    session_id=self.session_id,
                )
            else:
                memory = self.memory

            results = memory.search(
                query=query, limit=limit, score_threshold=score_threshold
            )

            # Extract episodic memory results
            episodic_memory = results.get("episodic_memory", [])
            profile_memory = results.get("profile_memory", [])

            # Create summary
            summary_parts = []
            if episodic_memory:
                summary_parts.append(
                    f"Found {len(episodic_memory)} relevant episodic memories"
                )
            if profile_memory:
                summary_parts.append(
                    f"Found {len(profile_memory)} relevant profile memories"
                )
            summary = ". ".join(summary_parts) if summary_parts else "No memories found"
        except Exception:
            logger.exception("Error searching memory")
            return {
                "status": "error",
                "message": "Error searching memory",
                "results": {"episodic_memory": [], "profile_memory": []},
                "summary": "Search failed",
            }
        else:
            return {
                "status": "success",
                "results": {
                    "episodic_memory": episodic_memory,
                    "profile_memory": profile_memory,
                },
                "summary": summary,
            }
            return {
                "status": "error",
                "message": "Error searching memory",
                "results": {"episodic_memory": [], "profile_memory": []},
                "summary": "Search failed",
            }

    def get_context(self, user_id: str | None = None) -> dict[str, Any]:
        """
        Get the current memory context configuration.

        Args:
            user_id: User ID override (uses instance user_id if not provided)

        Returns:
            Dictionary containing group_id, agent_id, user_id, and session_id

        """
        try:
            # If user_id is different, create a new memory instance
            if user_id and user_id != self.user_id:
                memory = self.project.memory(
                    group_id=self.group_id,
                    agent_id=self.agent_id,
                    user_id=user_id,
                    session_id=self.session_id,
                )
            else:
                memory = self.memory

            context = memory.get_context()
        except Exception:
            logger.exception("Error getting context")
            return {
                "status": "error",
                "message": "Error getting context",
            }
        else:
            return {
                "status": "success",
                **context,
            }

    def close(self) -> None:
        """Close the client and clean up resources."""
        self.client.close()


def get_memmachine_tools(
    base_url: str,
    api_key: str | None = None,
    org_id: str = "default_org",
    project_id: str = "default_project",
    group_id: str = "default_group",
    agent_id: str = "default_agent",
    user_id: str = "default_user",
    session_id: str | None = None,
    **kwargs: dict[str, Any],
) -> tuple[MemMachineTools, list[dict[str, Any]]]:
    """
    Initialize MemMachine tools and return tool schemas for AWS Strands Agent SDK.

    Args:
        base_url: Base URL of the MemMachine server
        api_key: API key for authentication (optional)
        org_id: Organization identifier (required for v2 API)
        project_id: Project identifier (required for v2 API)
        group_id: Group identifier for the memory context
        agent_id: Agent identifier for the memory context
        user_id: User identifier for the memory context
        session_id: Session identifier (auto-generated if not provided)
        **kwargs: Additional configuration options

    Returns:
        Tuple of (MemMachineTools instance, list of tool schemas)

    """
    # Initialize tools
    tools = MemMachineTools(
        base_url=base_url,
        api_key=api_key,
        org_id=org_id,
        project_id=project_id,
        group_id=group_id,
        agent_id=agent_id,
        user_id=user_id,
        session_id=session_id,
        **kwargs,
    )

    # Define tool schemas in AWS Bedrock tool specification format
    # AWS Strands Agent SDK expects tools directly as toolSpec content (not nested)
    tool_schemas_raw = [
        {
            "toolSpec": {
                "name": "add_memory",
                "description": "Store important information about the user or conversation into memory. Use this to remember user preferences, facts, or important details from the conversation.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The content to store in memory. Should be a clear, factual statement.",
                            },
                            "user_id": {
                                "type": "string",
                                "description": "Optional user ID override. If not provided, uses the default user ID.",
                            },
                            "episode_type": {
                                "type": "string",
                                "description": "Type of episode (default: 'text')",
                                "default": "text",
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Additional metadata as key-value pairs",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["content"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "search_memory",
                "description": "Retrieve relevant context, memories, or profile for a user. Use this to recall past conversations, user preferences, or relevant information.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string. Should be a natural language question or description of what to search for.",
                            },
                            "user_id": {
                                "type": "string",
                                "description": "Optional user ID override. If not provided, uses the default user ID.",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 5, max: 20)",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20,
                            },
                        },
                        "required": ["query"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_context",
                "description": "Get the current memory context configuration including group_id, agent_id, user_id, and session_id.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "Optional user ID override. If not provided, uses the default user ID.",
                            },
                        },
                        "required": [],
                    }
                },
            }
        },
    ]

    # Return toolSpec content directly (AWS Strands Agent SDK format)
    # If you need nested format {"toolSpec": {...}}, use get_memmachine_tools_nested() instead
    tool_schemas = [schema["toolSpec"] for schema in tool_schemas_raw]

    return tools, tool_schemas


def get_memmachine_tools_nested(
    base_url: str,
    api_key: str | None = None,
    org_id: str = "default_org",
    project_id: str = "default_project",
    group_id: str = "default_group",
    agent_id: str = "default_agent",
    user_id: str = "default_user",
    session_id: str | None = None,
    **kwargs: dict[str, Any],
) -> tuple[MemMachineTools, list[dict[str, Any]]]:
    """
    Initialize MemMachine tools and return tool schemas in nested format {"toolSpec": {...}}.

    Some SDKs may require the nested format. Use this if the default flat format doesn't work.

    Args:
        base_url: Base URL of the MemMachine server
        api_key: API key for authentication (optional)
        org_id: Organization identifier (required for v2 API)
        project_id: Project identifier (required for v2 API)
        group_id: Group identifier for the memory context
        agent_id: Agent identifier for the memory context
        user_id: User identifier for the memory context
        session_id: Session identifier (auto-generated if not provided)
        **kwargs: Additional configuration options

    Returns:
        Tuple of (MemMachineTools instance, list of tool schemas in nested format)

    """
    tools = MemMachineTools(
        base_url=base_url,
        api_key=api_key,
        org_id=org_id,
        project_id=project_id,
        group_id=group_id,
        agent_id=agent_id,
        user_id=user_id,
        session_id=session_id,
        **kwargs,
    )

    # Return nested format {"toolSpec": {...}}
    tool_schemas = [
        {
            "toolSpec": {
                "name": "add_memory",
                "description": "Store important information about the user or conversation into memory. Use this to remember user preferences, facts, or important details from the conversation.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The content to store in memory. Should be a clear, factual statement.",
                            },
                            "user_id": {
                                "type": "string",
                                "description": "Optional user ID override. If not provided, uses the default user ID.",
                            },
                            "episode_type": {
                                "type": "string",
                                "description": "Type of episode (default: 'text')",
                                "default": "text",
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Additional metadata as key-value pairs",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["content"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "search_memory",
                "description": "Retrieve relevant context, memories, or profile for a user. Use this to recall past conversations, user preferences, or relevant information.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string. Should be a natural language question or description of what to search for.",
                            },
                            "user_id": {
                                "type": "string",
                                "description": "Optional user ID override. If not provided, uses the default user ID.",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 5, max: 20)",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20,
                            },
                        },
                        "required": ["query"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_context",
                "description": "Get the current memory context configuration including group_id, agent_id, user_id, and session_id.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "Optional user ID override. If not provided, uses the default user ID.",
                            },
                        },
                        "required": [],
                    }
                },
            }
        },
    ]

    return tools, tool_schemas


def create_tool_handler(tools: MemMachineTools) -> dict[str, Callable]:
    """
    Create a tool handler dictionary for executing tool calls.

    Args:
        tools: MemMachineTools instance

    Returns:
        Dictionary mapping tool names to their handler functions

    """
    return {
        "add_memory": tools.add_memory,
        "search_memory": tools.search_memory,
        "get_context": tools.get_context,
    }
