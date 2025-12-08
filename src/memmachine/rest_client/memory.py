"""
Memory management interface for MemMachine.

This module provides the Memory class that handles episodic and profile memory
operations for a specific context.
"""

from __future__ import annotations

import logging
from datetime import UTC
from typing import TYPE_CHECKING, Any

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

        client = MemMachineClient(base_url="http://localhost:8080")

        # Get or create a project
        project = client.get_project(org_id="my_org", project_id="my_project")
        # Or create a new project
        # project = client.create_project(org_id="my_org", project_id="my_project")

        # Create memory from project
        memory = project.memory(
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
        agent_id: str | None = None,
        user_id: str | None = None,
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
            agent_id: Agent identifier (optional, will be stored in metadata)
            user_id: User identifier (optional, will be stored in metadata)
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
        self.__agent_id = agent_id
        self.__user_id = user_id
        self.__session_id = session_id

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
    def user_id(self) -> str | None:
        """
        Get the user_id (read-only).

        Returns:
            User identifier, or None if not set

        """
        return self.__user_id

    @property
    def agent_id(self) -> str | None:
        """
        Get the agent_id (read-only).

        Returns:
            Agent identifier, or None if not set

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

    def _build_metadata(
        self, additional_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
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
            metadata["user_id"] = self.__user_id
        if self.__agent_id:
            metadata["agent_id"] = self.__agent_id
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
        timeout: int | None = None,
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
            timeout: Request timeout in seconds (uses client default if not provided)

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
            # Use user_id if available, otherwise use a default
            producer = self.__user_id or "user"
        if not produced_for:
            # Use agent_id if available, otherwise use a default
            produced_for = self.__agent_id or "agent"

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
                    "Role '%s' is not a standard role. Expected one of %s. Using as-is.",
                    role,
                    valid_roles,
                )

            # Build metadata including old context fields and episode_type
            combined_metadata = self._build_metadata(metadata)
            if episode_type:
                combined_metadata["episode_type"] = episode_type

            # Convert to v2 API format
            v2_data = {
                "org_id": self.__org_id,
                "project_id": self.__project_id,
                "messages": [
                    {
                        "content": content,
                        "producer": producer,
                        "produced_for": produced_for or "",
                        "timestamp": datetime.now(tz=UTC)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "role": role,  # "user", "assistant", or "system"
                        "metadata": combined_metadata,
                    }
                ],
            }

            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories",
                json=v2_data,
                timeout=timeout,
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
        filter_dict: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Search for memories.

        This method automatically applies built-in filters based on the Memory instance's
        context (user_id, agent_id, session_id) via `get_default_filter_dict()`. These
        built-in filters are merged with any user-provided `filter_dict`, with user-provided
        filters taking precedence if there are key conflicts.

        Args:
            query: Search query string
            limit: Maximum number of results to return
            filter_dict: Additional filters for the search (key-value pairs as strings).
                        These filters will be merged with built-in filters (user_id, agent_id,
                        session_id). User-provided filters take precedence over built-in filters
                        if there are key conflicts.
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            Dictionary containing search results from both episodic and profile memory

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        Examples:
            ```python
            memory = project.memory(user_id="user1", agent_id="agent1", session_id="session1")

            # Built-in filters (user_id, agent_id, session_id) are automatically applied
            results = memory.search("query")
            # Equivalent to: search("query", filter_dict={"metadata.user_id": "user1", ...})

            # User-provided filters are merged with built-in filters
            results = memory.search("query", filter_dict={"category": "work"})
            # Final filter: {"metadata.user_id": "user1", "metadata.agent_id": "agent1",
            #               "metadata.session_id": "session1", "category": "work"}

            # User-provided filters override built-in filters for the same key
            results = memory.search("query", filter_dict={"metadata.user_id": "user2"})
            # Final filter: {"metadata.user_id": "user2", "metadata.agent_id": "agent1", ...}
            ```

        """
        if self._client_closed:
            raise RuntimeError("Cannot search memories: client has been closed")

        # Get built-in filters from context (user_id, agent_id, session_id)
        built_in_filters = self.get_default_filter_dict()

        # Merge built-in filters with user-provided filters
        # User-provided filters take precedence if there are key conflicts
        merged_filters = {**built_in_filters}
        if filter_dict:
            merged_filters.update(filter_dict)

        # Use v2 API: convert to v2 format
        # Convert merged filter_dict to string format: key='value' AND key='value'
        filter_str = ""
        if merged_filters:
            filter_str = self._dict_to_filter_string(merged_filters)

        # Convert to v2 API format
        v2_search_data = {
            "org_id": self.__org_id,
            "project_id": self.__project_id,
            "query": query,
            "top_k": limit or 10,
            "filter": filter_str,
            "types": ["episodic", "semantic"],  # Search both types
        }

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/search",
                json=v2_search_data,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            logger.info("Search completed for query: %s", query)
            # v2 API returns SearchResult with content field
            # SearchResult structure: { "status": 0, "content": { "episodic_memory": {...}, "semantic_memory": [...] } }
            content = data.get("content", {})

            # Process episodic_memory: it's a QueryResponse object with structure:
            # {
            #   "long_term_memory": { "episodes": [...] },
            #   "short_term_memory": { "episodes": [...], "episode_summary": [...] }
            # }
            episodic_memory = content.get("episodic_memory", {})
            if isinstance(episodic_memory, dict):
                # Extract episodes from nested long_term_memory and short_term_memory objects
                long_term_memory = episodic_memory.get("long_term_memory", {})
                short_term_memory = episodic_memory.get("short_term_memory", {})

                # Get episodes from each memory type
                long_term_episodes = (
                    long_term_memory.get("episodes", [])
                    if isinstance(long_term_memory, dict)
                    else []
                )
                short_term_episodes = (
                    short_term_memory.get("episodes", [])
                    if isinstance(short_term_memory, dict)
                    else []
                )
                # episode_summary is inside short_term_memory
                episode_summary = (
                    short_term_memory.get("episode_summary", [])
                    if isinstance(short_term_memory, dict)
                    else []
                )

                # Combine episodes for backward compatibility
                combined_episodes = []
                if isinstance(long_term_episodes, list):
                    combined_episodes.extend(long_term_episodes)
                if isinstance(short_term_episodes, list):
                    combined_episodes.extend(short_term_episodes)

                # Return in a format compatible with old API
                return {
                    "episodic_memory": combined_episodes if combined_episodes else [],
                    "episode_summary": episode_summary
                    if isinstance(episode_summary, list)
                    else [],
                    "semantic_memory": content.get("semantic_memory", []),
                }
            # Fallback: return as-is if it's already a list (for backward compatibility)
            return {
                "episodic_memory": episodic_memory
                if isinstance(episodic_memory, list)
                else [],
                "episode_summary": [],
                "semantic_memory": content.get("semantic_memory", []),
            }
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

    def get_current_metadata(self) -> dict[str, Any]:
        """
        Get current Memory instance metadata and built-in filters for logging/debugging.

        This method returns a dictionary containing:
        - Context information (org_id, project_id, group_id, agent_id, user_id, session_id)
        - Built-in filter dictionary (from get_default_filter_dict())
        - Built-in filter string (SQL-like format)

        Useful for logging and debugging to see what filters are automatically applied
        during search operations.

        Returns:
            Dictionary containing:
            - "context": Context information (org_id, project_id, etc.)
            - "built_in_filters": Built-in filter dictionary (metadata.user_id, etc.)
            - "built_in_filter_string": Built-in filter string in SQL-like format

        Examples:
            ```python
            memory = project.memory(user_id="user1", agent_id="agent1", session_id="session1")

            # Get current metadata for logging
            metadata = memory.get_current_metadata()
            print(f"Current context: {metadata['context']}")
            print(f"Built-in filters: {metadata['built_in_filters']}")
            print(f"Filter string: {metadata['built_in_filter_string']}")

            # Output:
            # Current context: {'org_id': 'org1', 'project_id': 'proj1', ...}
            # Built-in filters: {'metadata.user_id': 'user1', 'metadata.agent_id': 'agent1', ...}
            # Filter string: metadata.user_id='user1' AND metadata.agent_id='agent1' AND ...
            ```

        """
        built_in_filters = self.get_default_filter_dict()
        built_in_filter_string = (
            self._dict_to_filter_string(built_in_filters) if built_in_filters else ""
        )

        return {
            "context": self.get_context(),
            "built_in_filters": built_in_filters,
            "built_in_filter_string": built_in_filter_string,
        }

    def delete_episodic(
        self,
        episodic_id: str = "",
        episodic_ids: list[str] | None = None,
        timeout: int | None = None,
    ) -> bool:
        """
        Delete a specific episodic memory by ID.

        Args:
            episodic_id: The unique identifier of the episodic memory to delete
            episodic_ids: List of episodic memory IDs to delete (optional, can be used instead of episodic_id)
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            True if deletion was successful

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        Example:
            ```python
            # Delete a specific episodic memory
            memory.delete_episodic(episodic_id="episode_123")
            ```

        """
        if self._client_closed:
            raise RuntimeError("Cannot delete episodic memory: client has been closed")

        v2_delete_data = {
            "org_id": self.__org_id,
            "project_id": self.__project_id,
            "episodic_id": episodic_id,
            "episodic_ids": episodic_ids or [],
        }

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/episodic/delete",
                json=v2_delete_data,
                timeout=timeout,
            )
            response.raise_for_status()
        except Exception:
            logger.exception("Failed to delete episodic memory %s", episodic_id)
            raise
        else:
            logger.info("Episodic memory %s deleted successfully", episodic_id)
            return True

    def delete_semantic(
        self,
        semantic_id: str = "",
        semantic_ids: list[str] | None = None,
        timeout: int | None = None,
    ) -> bool:
        """
        Delete a specific semantic memory by ID.

        Args:
            semantic_id: The unique identifier of the semantic memory to delete
            semantic_ids: List of semantic memory IDs to delete
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            True if deletion was successful

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        Example:
            ```python
            # Delete a specific semantic memory
            memory.delete_semantic(semantic_id="feature_123")
            ```

        """
        if self._client_closed:
            raise RuntimeError("Cannot delete semantic memory: client has been closed")

        v2_delete_data = {
            "org_id": self.__org_id,
            "project_id": self.__project_id,
            "semantic_id": semantic_id,
            "semantic_ids": semantic_ids or [],
        }

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/delete",
                json=v2_delete_data,
                timeout=timeout,
            )
            response.raise_for_status()
        except Exception:
            logger.exception("Failed to delete semantic memory %s", semantic_id)
            raise
        else:
            logger.info("Semantic memory %s deleted successfully", semantic_id)
            return True

    def get_default_filter_dict(self) -> dict[str, str]:
        """
        Get default filter_dict based on Memory context (user_id, agent_id, session_id).

        This method returns a dictionary with metadata filters for the current Memory
        instance's context. These filters are automatically applied in the `search()` method
        and merged with any user-provided filters.

        Note: You don't need to manually merge this with your filter_dict when calling
        search() - it's done automatically. This method is mainly useful for:
        - Debugging/logging (see `get_current_metadata()`)
        - Understanding what filters are being applied
        - Manual filter construction if needed

        Only includes fields that are not None.

        Returns:
            Dictionary with metadata filters for non-None context fields

        Examples:
            ```python
            memory = project.memory(user_id="user1", agent_id="agent1", session_id="session1")

            # Get default filter dict (for debugging/logging)
            default_filters = memory.get_default_filter_dict()
            # Returns: {"metadata.user_id": "user1", "metadata.agent_id": "agent1", "metadata.session_id": "session1"}

            # These filters are automatically applied in search()
            results = memory.search("query")
            # Built-in filters are automatically included

            # User-provided filters are merged with built-in filters
            results = memory.search("query", filter_dict={"category": "work"})
            # Final filter includes both built-in and user-provided filters
            ```

        """
        default_filter: dict[str, str] = {}

        if self.__user_id is not None:
            default_filter["metadata.user_id"] = self.__user_id

        if self.__agent_id is not None:
            default_filter["metadata.agent_id"] = self.__agent_id

        if self.__session_id is not None:
            default_filter["metadata.session_id"] = self.__session_id

        return default_filter

    def _dict_to_filter_string(self, filter_dict: dict[str, str]) -> str:
        """
        Convert filter_dict to SQL-like filter string format: key='value' AND key='value'.

        Args:
            filter_dict: Dictionary of filter conditions (all values must be strings)

        Returns:
            Filter string in SQL-like format

        Raises:
            TypeError: If any value in filter_dict is not a string

        Examples:
            {"metadata.user_id": "test"} -> "metadata.user_id='test'"
            {"category": "work", "type": "preference"} -> "category='work' AND type='preference'"
            {"name": "O'Brien"} -> "name='O''Brien'"  # Single quotes are escaped

        """
        conditions = []

        for key, value in filter_dict.items():
            # Validate that value is a string
            if not isinstance(value, str):
                raise TypeError(
                    f"All filter_dict values must be strings, but got {type(value).__name__} "
                    f"for key '{key}': {value!r}"
                )

            # Validate that key is a string
            if not isinstance(key, str):
                raise TypeError(
                    f"All filter_dict keys must be strings, but got {type(key).__name__} "
                    f"for key: {key!r}"
                )

            # Escape single quotes in strings (SQL standard: ' -> '')
            escaped_value = value.replace("'", "''")
            conditions.append(f"{key}='{escaped_value}'")

        return " AND ".join(conditions)

    def mark_client_closed(self) -> None:
        """Mark this memory instance as closed by its owning client."""
        self._client_closed = True

    def __repr__(self) -> str:
        """Return a developer-friendly description of the memory context."""
        return (
            f"Memory(org_id='{self.org_id}', "
            f"project_id='{self.project_id}', "
            f"group_id='{self.group_id}', "
            f"agent_id='{self.agent_id}', "
            f"user_id='{self.user_id}', "
            f"session_id='{self.session_id}')"
        )
