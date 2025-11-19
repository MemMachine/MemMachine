"""
Memory management interface for MemMachine.

This module provides the Memory class that handles episodic and profile memory
operations for a specific context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import requests

if TYPE_CHECKING:
    from .client import MemMachineClient

logger = logging.getLogger(__name__)


class Memory:
    """
    Memory interface for managing episodic and profile memory.

    This class provides methods for adding, searching, and managing memories
    within a specific context (group, agent, user, session).

    Example:
        ```python
        from memmachine import MemMachineClient

        client = MemMachineClient()
        memory = client.memory(
            org_id="my_org",
            project_id="my_project",
            group_id="my_group",  # Optional: stored in metadata
            agent_id="my_agent",  # Optional: stored in metadata
            user_id="user123",    # Optional: stored in metadata
            session_id="session456"  # Optional: stored in metadata
        )

        # Add a memory (role defaults to "user")
        memory.add("I like pizza", metadata={"type": "preference"})
        
        # Add assistant response
        memory.add("I understand you like pizza", role="assistant")
        
        # Add system message
        memory.add("System initialized", role="system")

        # Search memories
        results = memory.search("What do I like to eat?")
        ```

    """

    def __init__(
        self,
        client: MemMachineClient,
        org_id: str,
        project_id: str,
        group_id: str | None = None,
        agent_id: str | list[str] | None = None,
        user_id: str | list[str] | None = None,
        session_id: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Initialize Memory instance.

        Args:
            client: MemMachineClient instance
            org_id: Organization identifier (required for v2 API)
            project_id: Project identifier (required for v2 API)
            group_id: Group identifier (optional, will be stored in metadata)
            agent_id: Agent identifier(s) (optional, will be stored in metadata)
            user_id: User identifier(s) (optional, will be stored in metadata)
            session_id: Session identifier (optional, will be stored in metadata)
            **kwargs: Additional configuration options

        """
        self.client = client
        self._client_closed = False
        self._extra_options = kwargs

        # v2 API requires org_id and project_id
        if not org_id:
            raise ValueError("org_id is required for v2 API")
        if not project_id:
            raise ValueError("project_id is required for v2 API")
        
        self.__org_id = org_id
        self.__project_id = project_id

        # Store old context fields for backward compatibility and metadata
        self.__group_id = group_id
        self.__session_id = session_id

        # Normalize agent_id and user_id to lists
        if agent_id is None:
            self.__agent_id = None
        elif isinstance(agent_id, list):
            self.__agent_id = agent_id if agent_id else None
        else:
            self.__agent_id = [agent_id]

        if user_id is None:
            self.__user_id = None
        elif isinstance(user_id, list):
            self.__user_id = user_id if user_id else None
        else:
            self.__user_id = [user_id]

    @property
    def org_id(self) -> str:
        """
        Get the org_id (read-only).

        Returns:
            Organization identifier

        """
        return self.__org_id

    @property
    def project_id(self) -> str:
        """
        Get the project_id (read-only).

        Returns:
            Project identifier

        """
        return self.__project_id

    @property
    def user_id(self) -> list[str] | None:
        """
        Get the user_id list (read-only).

        Returns:
            List of user identifiers, or None if not set

        """
        return self.__user_id

    @property
    def agent_id(self) -> list[str] | None:
        """
        Get the agent_id list (read-only).

        Returns:
            List of agent identifiers, or None if not set

        """
        return self.__agent_id

    @property
    def group_id(self) -> str | None:
        """
        Get the group_id (read-only).

        Returns:
            Group identifier, or None if not set

        """
        return self.__group_id

    @property
    def session_id(self) -> str | None:
        """
        Get the session_id (read-only).

        Returns:
            Session identifier, or None if not set

        """
        return self.__session_id

    def _build_metadata(self, additional_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Build metadata dictionary including old context fields.

        Args:
            additional_metadata: Additional metadata to include

        Returns:
            Dictionary with all metadata including old context fields

        """
        metadata = additional_metadata.copy() if additional_metadata else {}
        
        # Add old context fields to metadata if they exist
        if self.__group_id:
            metadata["group_id"] = self.__group_id
        if self.__user_id:
            metadata["user_id"] = self.__user_id if len(self.__user_id) > 1 else self.__user_id[0]
        if self.__agent_id:
            metadata["agent_id"] = self.__agent_id if len(self.__agent_id) > 1 else self.__agent_id[0]
        if self.__session_id:
            metadata["session_id"] = self.__session_id
        
        return metadata

    def add(  # noqa: C901
        self,
        content: str,
        role: str = "user",
        producer: str | None = None,
        produced_for: str | None = None,
        episode_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Add a memory episode.

        Args:
            content: The content to store in memory
            role: Message role - "user", "assistant", or "system" (default: "user")
            producer: Who produced this content (defaults to first user_id)
            produced_for: Who this content is for (defaults to first agent_id)
            episode_type: Type of episode (default: "text") - stored in metadata
            metadata: Additional metadata for the episode

        Returns:
            True if the memory was added successfully

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        if self._client_closed:
            raise RuntimeError("Cannot add memory: client has been closed")

        # Set default producer and produced_for if not provided
        # In v2 API, these are just string fields, no strict validation needed
        if not producer:
            # Use first user_id if available, otherwise use a default
            if self.__user_id and len(self.__user_id) > 0:
                producer = self.__user_id[0]
            else:
                producer = "user"  # Default fallback
        if not produced_for:
            # Use first agent_id if available, otherwise use a default
            if self.__agent_id and len(self.__agent_id) > 0:
                produced_for = self.__agent_id[0]
            else:
                produced_for = "agent"  # Default fallback

        # Log the request details for debugging
        logger.debug(
            (
                "Adding memory: org_id=%s, project_id=%s, producer=%s, "
                "group_id=%s, user_id=%s, agent_id=%s, session_id=%s"
            ),
            self.__org_id,
            self.__project_id,
            producer,
            self.__group_id,
            self.__user_id,
            self.__agent_id,
            self.__session_id,
        )

        try:
            # Use v2 API: convert to v2 format
            from datetime import datetime
            
            # Validate role
            valid_roles = {"user", "assistant", "system"}
            if role not in valid_roles:
                logger.warning(
                    f"Role '{role}' is not a standard role. Expected one of {valid_roles}. "
                    "Using as-is."
                )
            
            # Build metadata including old context fields and episode_type
            combined_metadata = self._build_metadata(metadata)
            if episode_type:
                combined_metadata["episode_type"] = episode_type
            
            # Convert to v2 API format
            v2_data = {
                "org_id": self.__org_id,
                "project_id": self.__project_id,
                "messages": [{
                    "content": content,
                    "producer": producer,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "role": role,  # "user", "assistant", or "system"
                    "metadata": combined_metadata
                }]
            }
            
            response = self.client._session.post(
                f"{self.client.base_url}/api/v2/memories",
                json=v2_data,
                timeout=self.client.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            # Try to get detailed error information from response
            error_detail = ""
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = f" Response: {e.response.text}"
                except Exception:
                    error_detail = f" Status: {e.response.status_code}"
            logger.exception("Failed to add memory%s", error_detail)
            raise
        except Exception:
            logger.exception("Failed to add memory")
            raise
        else:
            logger.debug("Successfully added memory: %s...", content[:50])
            return True

    def search(
        self,
        query: str,
        limit: int | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Search for memories.

        Args:
            query: Search query string
            limit: Maximum number of results to return
            filter_dict: Additional filters for the search

        Returns:
            Dictionary containing search results from both episodic and profile memory

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        if self._client_closed:
            raise RuntimeError("Cannot search memories: client has been closed")

        # Use v2 API: convert to v2 format
        # Convert filter_dict to string format if needed
        filter_str = ""
        if filter_dict:
            # Simple conversion - you may need to adjust based on your filter format
            import json
            filter_str = json.dumps(filter_dict)
        
        # Convert to v2 API format
        v2_search_data = {
            "org_id": self.__org_id,
            "project_id": self.__project_id,
            "query": query,
            "top_k": limit or 10,
            "filter": filter_str,
            "types": ["episodic", "semantic"]  # Search both types
        }

        try:
            response = self.client._session.post(
                f"{self.client.base_url}/api/v2/memories/search",
                json=v2_search_data,
                timeout=self.client.timeout,
            )
            response.raise_for_status()
            data = response.json()
            logger.info("Search completed for query: %s", query)
            # v2 API returns SearchResult with content field
            return data.get("content", {})
        except Exception:
            logger.exception("Failed to search memories")
            raise

    def get_context(self) -> dict[str, Any]:
        """
        Get the current memory context.

        Returns:
            Dictionary containing the context information

        """
        return {
            "org_id": self.__org_id,
            "project_id": self.__project_id,
            "group_id": self.__group_id,
            "agent_id": self.__agent_id,
            "user_id": self.__user_id,
            "session_id": self.__session_id,
        }

    def mark_client_closed(self) -> None:
        """Mark this memory instance as closed by its owning client."""
        self._client_closed = True

    def __repr__(self) -> str:
        """Return a developer-friendly description of the memory context."""
        return (
            f"Memory(org_id='{self.org_id}', "
            f"project_id='{self.project_id}', "
            f"group_id='{self.group_id}', "
            f"user_id='{self.user_id}', "
            f"session_id='{self.session_id}')"
        )
