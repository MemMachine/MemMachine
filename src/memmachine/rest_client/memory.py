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

from memmachine.common.api import EpisodeType, MemoryType
from memmachine.common.api.spec import (
    AddMemoriesResponse,
    AddMemoriesSpec,
    AddMemoryResult,
    DeleteEpisodicMemorySpec,
    DeleteSemanticMemorySpec,
    ListMemoriesSpec,
    MemoryMessage,
    SearchMemoriesSpec,
    SearchResult,
)

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

        # Create memory from project with metadata
        memory = project.memory(
            metadata={
                "user_id": "user123",
                "agent_id": "my_agent",
                "group_id": "my_group",
                "session_id": "session456"
            }
        )

        # Add a memory (role defaults to "user")
        # Instance metadata is merged with additional metadata
        memory.add("I like pizza", metadata={"type": "preference"})

        # Add assistant response
        memory.add("I understand you like pizza", role="assistant")

        # Add system message
        memory.add("System initialized", role="system")

        # Search memories (filters based on metadata are automatically applied)
        results = memory.search("What do I like to eat?")
        ```

    """

    def __init__(
        self,
        client: MemMachineClient,
        org_id: str,
        project_id: str,
        metadata: dict[str, str] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Initialize Memory instance.

        Args:
            client: MemMachineClient instance
            org_id: Organization identifier (required for v2 API)
            project_id: Project identifier (required for v2 API)
            metadata: Metadata dictionary that will be merged with metadata
                     in add() and search() operations. Common keys include:
                     user_id, agent_id, group_id, session_id, etc.
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

        # Store metadata dictionary
        self.__metadata = metadata.copy() if metadata else {}

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
    def metadata(self) -> dict[str, str]:
        """
        Get the metadata dictionary (read-only).

        Returns:
            Metadata dictionary

        """
        return self.__metadata.copy()

    def _build_metadata(
        self, additional_metadata: dict[str, str] | None = None
    ) -> dict[str, str]:
        """
        Build metadata dictionary by merging instance metadata with additional metadata.

        Args:
            additional_metadata: Additional metadata to merge (takes precedence)

        Returns:
            Dictionary with merged metadata (additional_metadata overrides instance metadata)

        """
        # Start with instance metadata
        merged_metadata = self.__metadata.copy()

        # Merge additional metadata (additional_metadata takes precedence)
        if additional_metadata:
            merged_metadata.update(additional_metadata)

        return merged_metadata

    def add(
        self,
        content: str,
        role: str = "user",
        producer: str | None = None,
        produced_for: str | None = None,
        episode_type: EpisodeType | None = EpisodeType.MESSAGE,
        memory_types: list[MemoryType] | None = None,
        metadata: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> list[AddMemoryResult]:
        """
        Add a memory episode.

        Args:
            content: The content to store in memory
            role: Message role - "user", "assistant", or "system" (default: "user")
            producer: Who produced this content (defaults to first user_id)
            produced_for: Who this content is for (defaults to first agent_id)
            episode_type: Type of episode (default: "message") - stored in metadata
            memory_types: List of MemoryType to store this memory under (default: both episodic and semantic)
            metadata: Additional metadata for the episode
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            List of AddMemoryResult objects containing UID results from the server.
            Each result has a "uid" attribute with the memory identifier.

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        if memory_types is None:
            memory_types = []
        if self._client_closed:
            raise RuntimeError("Cannot add memory: client has been closed")

        # If producer and produced_for are not provided, leave as None
        # No automatic fallback to metadata values

        # Log the request details for debugging
        logger.debug(
            ("Adding memory: org_id=%s, project_id=%s, producer=%s, metadata=%s"),
            self.__org_id,
            self.__project_id,
            producer,
            self.__metadata,
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
                combined_metadata["episode_type"] = episode_type.value

            # Convert None to empty string for producer and produced_for
            # (MemoryMessage requires str, not None)
            producer_str = producer if producer is not None else ""
            produced_for_str = produced_for if produced_for is not None else ""

            # Use shared API Pydantic models
            message = MemoryMessage(
                content=content,
                producer=producer_str,
                produced_for=produced_for_str,
                timestamp=datetime.now(tz=UTC),
                role=role,  # "user", "assistant", or "system"
                metadata=combined_metadata,
                episode_type=episode_type,
            )

            spec = AddMemoriesSpec(
                org_id=self.__org_id,
                project_id=self.__project_id,
                messages=[message],
                types=memory_types,
            )
            v2_data = spec.model_dump(mode="json", exclude_none=True)

            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()

            # Parse response using Pydantic model for validation
            add_response = AddMemoriesResponse(**response_data)

            logger.debug(
                "Successfully added memory: %s... (UIDs: %s)",
                content[:50],
                [result.uid for result in add_response.results],
            )
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
            return add_response.results

    def search(
        self,
        query: str,
        limit: int | None = None,
        filter_dict: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> SearchResult:
        """
        Search for memories.

        This method automatically applies built-in filters based on the Memory instance's
        metadata via `get_default_filter_dict()`. These built-in filters are merged with any
        user-provided `filter_dict`, with user-provided filters taking precedence if there
        are key conflicts.

        Args:
            query: Search query string
            limit: Maximum number of results to return
            filter_dict: Additional filters for the search (key-value pairs as strings).
                        These filters will be merged with built-in filters from metadata.
                        User-provided filters take precedence over built-in filters
                        if there are key conflicts.
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            SearchResult object containing search results from both episodic and semantic memory

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        if self._client_closed:
            raise RuntimeError("Cannot search memories: client has been closed")

        # Get built-in filters from metadata
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

        # Use shared API Pydantic models
        spec = SearchMemoriesSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            query=query,
            top_k=limit or 10,
            filter=filter_str,
            types=[MemoryType.Episodic, MemoryType.Semantic],  # Search both types
        )
        v2_search_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/search",
                json=v2_search_data,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            # Parse response using Pydantic model for validation
            search_result = SearchResult(**response_data)
            logger.info("Search completed for query: %s", query)
        except Exception:
            logger.exception("Failed to search memories")
            raise
        else:
            return search_result

    def list(
        self,
        memory_type: MemoryType = MemoryType.Episodic,
        page_size: int = 100,
        page_num: int = 0,
        filter_dict: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> SearchResult:
        """
        List memories in this project (v2 API).

        Calls: POST /api/v2/memories/list

        Args:
            memory_type: Which memory store to list (Episodic or Semantic)
            page_size: Page size (server default is 100)
            page_num: Page number (0-based)
            filter_dict: Optional extra filters; merged with built-in context filters
            timeout: Request timeout override

        Returns:
            SearchResult object containing list results

        """
        if self._client_closed:
            raise RuntimeError("Cannot list memories: client has been closed")

        built_in_filters = self.get_default_filter_dict()
        merged_filters = {**built_in_filters}
        if filter_dict:
            merged_filters.update(filter_dict)

        filter_str = (
            self._dict_to_filter_string(merged_filters) if merged_filters else ""
        )

        spec = ListMemoriesSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            page_size=page_size,
            page_num=page_num,
            filter=filter_str,
            type=memory_type,
        )
        v2_list_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/list",
                json=v2_list_data,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            # Parse response using Pydantic model for validation
            search_result = SearchResult(**response_data)
            logger.info(
                "List completed for org_id=%s project_id=%s type=%s page_num=%s page_size=%s",
                self.__org_id,
                self.__project_id,
                getattr(memory_type, "value", memory_type),
                page_num,
                page_size,
            )
        except Exception:
            logger.exception("Failed to list memories")
            raise
        else:
            return search_result

    def get_context(self) -> dict[str, Any]:
        """
        Get the current memory context.

        Returns:
            Dictionary containing the context information (org_id, project_id, and metadata).
            The metadata field is a dict[str, str].

        """
        return {
            "org_id": self.__org_id,
            "project_id": self.__project_id,
            "metadata": self.__metadata.copy(),  # dict[str, str]
        }

    def get_current_metadata(self) -> dict[str, Any]:
        """
        Get current Memory instance metadata and built-in filters for logging/debugging.

        This method returns a dictionary containing:
        - Context information (org_id, project_id, metadata)
        - Built-in filter dictionary (from get_default_filter_dict())
        - Built-in filter string (SQL-like format)

        Useful for logging and debugging to see what filters are automatically applied
        during search operations.

        Returns:
            Dictionary containing:
            - "context": Context information (org_id, project_id, metadata)
            - "built_in_filters": Built-in filter dictionary (metadata.* keys)
            - "built_in_filter_string": Built-in filter string in SQL-like format

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

        """
        if self._client_closed:
            raise RuntimeError("Cannot delete episodic memory: client has been closed")

        spec = DeleteEpisodicMemorySpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            episodic_id=episodic_id,
            episodic_ids=episodic_ids or [],
        )
        v2_delete_data = spec.model_dump(mode="json", exclude_none=True)

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

        """
        if self._client_closed:
            raise RuntimeError("Cannot delete semantic memory: client has been closed")

        spec = DeleteSemanticMemorySpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            semantic_id=semantic_id,
            semantic_ids=semantic_ids or [],
        )
        v2_delete_data = spec.model_dump(mode="json", exclude_none=True)

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
        Get default filter_dict based on Memory metadata.

        This method returns a dictionary with metadata filters for the current Memory
        instance's metadata. These filters are automatically applied in the `search()` method
        and merged with any user-provided filters.

        Note: You don't need to manually merge this with your filter_dict when calling
        search() - it's done automatically. This method is mainly useful for:
        - Debugging/logging (see `get_current_metadata()`)
        - Understanding what filters are being applied
        - Manual filter construction if needed

        Only includes fields that are strings (for filter compatibility).

        Returns:
            Dictionary with metadata filters (keys prefixed with "metadata.")

        """
        default_filter: dict[str, str] = {}

        # Convert metadata values to filter format (only string values)
        for key, value in self.__metadata.items():
            if isinstance(value, str):
                default_filter[f"metadata.{key}"] = value

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
            f"metadata={self.__metadata})"
        )
