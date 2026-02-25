"""
Memory management interface for MemMachine.

This module provides the Memory class that handles episodic and profile memory
operations for a specific context.
"""

from __future__ import annotations

import builtins
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import requests
from pydantic import JsonValue

from memmachine.common.api import EpisodeType, MemoryType
from memmachine.common.api.spec import (
    AddFeatureResponse,
    AddFeatureSpec,
    AddMemoriesResponse,
    AddMemoriesSpec,
    AddMemoryResult,
    AddSemanticCategoryResponse,
    AddSemanticCategorySpec,
    AddSemanticCategoryTemplateSpec,
    AddSemanticTagResponse,
    AddSemanticTagSpec,
    ConfigureEpisodicMemorySpec,
    ConfigureSemanticSetSpec,
    CreateSemanticSetTypeResponse,
    CreateSemanticSetTypeSpec,
    DeleteEpisodicMemorySpec,
    DeleteSemanticCategorySpec,
    DeleteSemanticMemorySpec,
    DeleteSemanticSetTypeSpec,
    DeleteSemanticTagSpec,
    DisableSemanticCategorySpec,
    EpisodicMemoryConfigEntry,
    GetEpisodicMemoryConfigSpec,
    GetFeatureSpec,
    GetSemanticCategorySetIdsResponse,
    GetSemanticCategorySetIdsSpec,
    GetSemanticCategorySpec,
    GetSemanticSetIdResponse,
    GetSemanticSetIdSpec,
    ListMemoriesSpec,
    ListResult,
    ListSemanticCategoryTemplatesResponse,
    ListSemanticCategoryTemplatesSpec,
    ListSemanticSetIdsResponse,
    ListSemanticSetIdsSpec,
    ListSemanticSetTypesResponse,
    ListSemanticSetTypesSpec,
    MemoryMessage,
    SearchMemoriesSpec,
    SearchResult,
    SemanticCategoryEntry,
    SemanticCategoryTemplateEntry,
    SemanticFeature,
    SemanticSetEntry,
    SemanticSetTypeEntry,
    UpdateFeatureSpec,
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

    def _validate_role(self, role: str) -> None:
        """Validate the role provided."""
        valid_roles = {"user", "assistant", "system"}
        if role and role not in valid_roles:
            logger.warning(
                "Role '%s' is not a standard role. Expected one of %s. Using as-is.",
                role,
                valid_roles,
            )

    def _build_memory_message(
        self,
        content: str,
        role: str,
        producer: str | None,
        produced_for: str | None,
        episode_type: EpisodeType | None,
        metadata: dict[str, str] | None,
        timestamp: datetime | None,
    ) -> MemoryMessage:
        """Build a MemoryMessage object from parameters."""
        # Build metadata including old context fields and episode_type
        combined_metadata = self._build_metadata(metadata)
        if episode_type is not None:
            combined_metadata["episode_type"] = episode_type.value

        # Use shared API Pydantic models
        message = MemoryMessage(  # type: ignore[call-arg,arg-type]
            content=content,
            role=role,
            metadata=combined_metadata,
        )

        # Only set fields if explicitly provided (let server defaults work for None)
        if producer is not None:
            message.producer = producer
        if produced_for is not None:
            message.produced_for = produced_for
        if timestamp is not None:
            message.timestamp = timestamp
        if episode_type is not None:
            message.episode_type = episode_type

        return message

    def add(
        self,
        content: str,
        role: str = "",
        producer: str | None = None,
        produced_for: str | None = None,
        episode_type: EpisodeType | None = None,
        memory_types: builtins.list[MemoryType] | None = None,
        metadata: dict[str, str] | None = None,
        timestamp: datetime | None = None,
        timeout: int | None = None,
    ) -> builtins.list[AddMemoryResult]:
        """
        Add a memory episode.

        Args:
            content: The content to store in memory
            role: Message role - "user", "assistant", or "system" (default: "")
            producer: Who produced this content (default: "user" if not provided, set by server)
            produced_for: Who this content is for (default: "" if not provided)
            episode_type: Type of episode (default: None, server will use "message")
            memory_types: List of MemoryType to store this memory under (default: both episodic and semantic)
            metadata: Additional metadata for the episode
            timestamp: Optional timestamp for the memory. If not provided, server will use current UTC time.
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
            # Validate role
            self._validate_role(role)

            # Build metadata including old context fields and episode_type
            message = self._build_memory_message(
                content=content,
                role=role,
                producer=producer,
                produced_for=produced_for,
                episode_type=episode_type,
                metadata=metadata,
                timestamp=timestamp,
            )

            spec = AddMemoriesSpec(
                org_id=self.__org_id,
                project_id=self.__project_id,
                messages=[message],
                types=memory_types,
            )
            v2_data = spec.model_dump(mode="json", exclude_unset=True)

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
        expand_context: int = 0,
        score_threshold: float | None = None,
        filter_dict: dict[str, str] | None = None,
        set_metadata: dict[str, JsonValue] | None = None,
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
            expand_context: The number of additional episodes to include
                            around each matched episode from long term memory.
            score_threshold: Minimum score to include in results.
            filter_dict: Additional filters for the search (key-value pairs as strings).
                        These filters will be merged with built-in filters from metadata.
                        User-provided filters take precedence over built-in filters
                        if there are key conflicts.
            set_metadata: Optional metadata key-value pairs used to select semantic sets.
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
            expand_context=expand_context,
            score_threshold=score_threshold,
            filter=filter_str,
            set_metadata=set_metadata,
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
        set_metadata: dict[str, JsonValue] | None = None,
        timeout: int | None = None,
    ) -> ListResult:
        """
        List memories in this project (v2 API).

        Calls: POST /api/v2/memories/list

        Args:
            memory_type: Which memory store to list (Episodic or Semantic)
            page_size: Page size (server default is 100)
            page_num: Page number (0-based)
            filter_dict: Optional extra filters; merged with built-in context filters
            set_metadata: Optional metadata key-value pairs used to select semantic sets.
            timeout: Request timeout override

        Returns:
            ListResult object containing list results

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
            set_metadata=set_metadata,
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
            search_result = ListResult(**response_data)
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
        episodic_ids: builtins.list[str] | None = None,
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
        semantic_ids: builtins.list[str] | None = None,
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

    def add_feature(
        self,
        *,
        set_id: str,
        category_name: str,
        tag: str,
        feature: str,
        value: str,
        feature_metadata: dict[str, JsonValue] | None = None,
        citations: builtins.list[str] | None = None,
        timeout: int | None = None,
    ) -> str:
        """
        Add a semantic feature.

        Args:
            set_id: Set ID to add the feature to.
            category_name: Category name to attach the feature to.
            tag: Tag name to associate with the feature.
            feature: Feature name/key.
            value: Feature value.
            feature_metadata: Optional metadata to store alongside the feature.
            citations: Optional episode IDs supporting this feature.
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            The created feature ID.

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        if self._client_closed:
            raise RuntimeError("Cannot add feature: client has been closed")

        spec = AddFeatureSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            set_id=set_id,
            category_name=category_name,
            tag=tag,
            feature=feature,
            value=value,
            feature_metadata=feature_metadata,
            citations=citations,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/feature",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            add_response = AddFeatureResponse(**response_data)
            logger.debug(
                "Successfully added feature: %s (ID: %s)",
                feature,
                add_response.feature_id,
            )
        except Exception:
            logger.exception("Failed to add feature %s", feature)
            raise
        else:
            return add_response.feature_id

    def get_feature(
        self,
        feature_id: str,
        load_citations: bool = False,
        timeout: int | None = None,
    ) -> SemanticFeature | None:
        """
        Get a semantic feature by ID.

        Args:
            feature_id: Feature identifier.
            load_citations: Whether to load referenced episode IDs.
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            The feature, or None if it does not exist.

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        if self._client_closed:
            raise RuntimeError("Cannot get feature: client has been closed")

        spec = GetFeatureSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            feature_id=feature_id,
            load_citations=load_citations,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/feature/get",
                json=v2_data,
                timeout=timeout,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            response_data = response.json()
            feature = SemanticFeature(**response_data)
            logger.debug("Successfully retrieved feature: %s", feature_id)
        except requests.HTTPError as e:
            if (
                hasattr(e, "response")
                and e.response is not None
                and e.response.status_code == 404
            ):
                return None
            raise
        except Exception:
            logger.exception("Failed to get feature %s", feature_id)
            raise
        else:
            return feature

    def update_feature(
        self,
        *,
        feature_id: str,
        category_name: str | None = None,
        feature: str | None = None,
        value: str | None = None,
        tag: str | None = None,
        metadata: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> bool:
        """
        Update an existing semantic feature.

        Only fields that are not None are updated.

        Args:
            feature_id: Feature identifier.
            category_name: New category name.
            feature: New feature name/key.
            value: New feature value.
            tag: New tag name.
            metadata: Replacement metadata payload.
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            True if update was successful.

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        if self._client_closed:
            raise RuntimeError("Cannot update feature: client has been closed")

        spec = UpdateFeatureSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            feature_id=feature_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            metadata=metadata,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/feature/update",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            logger.debug("Successfully updated feature: %s", feature_id)
        except Exception:
            logger.exception("Failed to update feature %s", feature_id)
            raise
        else:
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

    def create_semantic_set_type(
        self,
        *,
        metadata_tags: builtins.list[str],
        is_org_level: bool = False,
        name: str | None = None,
        description: str | None = None,
        timeout: int | None = None,
    ) -> str:
        """
        Create a new semantic set type.

        A set type defines a template for grouping semantic features based on
        metadata tags.

        Args:
            metadata_tags: Ordered list of metadata tag keys that define the set type.
            is_org_level: Whether the set type is org-scoped (True) or project-scoped (False).
            name: Optional human-readable name for the set type.
            description: Optional description of the set type's purpose.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            The created set type ID.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot create set type: client has been closed")

        spec = CreateSemanticSetTypeSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            is_org_level=is_org_level,
            metadata_tags=metadata_tags,
            name=name,
            description=description,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/set_type",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            result = CreateSemanticSetTypeResponse(**response_data)
            logger.debug("Successfully created set type: %s", result.set_type_id)
        except Exception:
            logger.exception("Failed to create set type")
            raise
        else:
            return result.set_type_id

    def list_semantic_set_types(
        self,
        timeout: int | None = None,
    ) -> builtins.list[SemanticSetTypeEntry]:
        """
        List all semantic set types.

        Args:
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            List of semantic set type entries.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot list set types: client has been closed")

        spec = ListSemanticSetTypesSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/set_type/list",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            result = ListSemanticSetTypesResponse(**response_data)
            logger.debug("Successfully listed %d set types", len(result.set_types))
        except Exception:
            logger.exception("Failed to list set types")
            raise
        else:
            return result.set_types

    def delete_semantic_set_type(
        self,
        set_type_id: str,
        timeout: int | None = None,
    ) -> bool:
        """
        Delete a semantic set type.

        Args:
            set_type_id: The unique identifier of the set type to delete.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            True if deletion was successful.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot delete set type: client has been closed")

        spec = DeleteSemanticSetTypeSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            set_type_id=set_type_id,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/set_type/delete",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            logger.debug("Successfully deleted set type: %s", set_type_id)
        except Exception:
            logger.exception("Failed to delete set type %s", set_type_id)
            raise
        else:
            return True

    def get_semantic_set_id(
        self,
        *,
        metadata_tags: builtins.list[str],
        is_org_level: bool = False,
        set_metadata: dict[str, JsonValue] | None = None,
        timeout: int | None = None,
    ) -> str:
        """
        Get or create a semantic set ID.

        Args:
            metadata_tags: Ordered list of metadata tag keys defining the set type.
            is_org_level: Whether the set is org-scoped (True) or project-scoped (False).
            set_metadata: Optional metadata key-value pairs used to identify the set.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            The set ID.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot get set ID: client has been closed")

        spec = GetSemanticSetIdSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            is_org_level=is_org_level,
            metadata_tags=metadata_tags,
            set_metadata=set_metadata,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/set_id/get",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            result = GetSemanticSetIdResponse(**response_data)
            logger.debug("Successfully got set ID: %s", result.set_id)
        except Exception:
            logger.exception("Failed to get set ID")
            raise
        else:
            return result.set_id

    def list_semantic_set_ids(
        self,
        set_metadata: dict[str, JsonValue] | None = None,
        timeout: int | None = None,
    ) -> builtins.list[SemanticSetEntry]:
        """
        List all semantic sets.

        Args:
            set_metadata: Optional metadata key-value pairs to filter sets.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            List of semantic set entries.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot list sets: client has been closed")

        spec = ListSemanticSetIdsSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            set_metadata=set_metadata,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/set_id/list",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            result = ListSemanticSetIdsResponse(**response_data)
            logger.debug("Successfully listed %d sets", len(result.sets))
        except Exception:
            logger.exception("Failed to list sets")
            raise
        else:
            return result.sets

    def configure_semantic_set(
        self,
        *,
        set_id: str,
        embedder_name: str | None = None,
        llm_name: str | None = None,
        timeout: int | None = None,
    ) -> bool:
        """
        Configure a semantic set.

        Args:
            set_id: The set ID to configure.
            embedder_name: Optional embedder name override for this set.
            llm_name: Optional language model name override for this set.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            True if configuration was successful.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot configure set: client has been closed")

        spec = ConfigureSemanticSetSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            set_id=set_id,
            embedder_name=embedder_name,
            llm_name=llm_name,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/set/configure",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            logger.debug("Successfully configured set: %s", set_id)
        except Exception:
            logger.exception("Failed to configure set %s", set_id)
            raise
        else:
            return True

    def get_semantic_category(
        self,
        category_id: str,
        timeout: int | None = None,
    ) -> SemanticCategoryEntry | None:
        """
        Get a semantic category by ID.

        Args:
            category_id: The unique identifier of the category to retrieve.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            The category entry, or None if not found.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot get category: client has been closed")

        spec = GetSemanticCategorySpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            category_id=category_id,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/category/get",
                json=v2_data,
                timeout=timeout,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            response_data = response.json()
            if response_data is None:
                return None
            result = SemanticCategoryEntry(**response_data)
            logger.debug("Successfully retrieved category: %s", category_id)
        except requests.HTTPError as e:
            if (
                hasattr(e, "response")
                and e.response is not None
                and e.response.status_code == 404
            ):
                return None
            raise
        except Exception:
            logger.exception("Failed to get category %s", category_id)
            raise
        else:
            return result

    def add_semantic_category(
        self,
        *,
        set_id: str,
        category_name: str,
        prompt: str,
        description: str | None = None,
        timeout: int | None = None,
    ) -> str:
        """
        Add a semantic category to a set.

        Args:
            set_id: The set ID to add the category to.
            category_name: Human-readable name for the category.
            prompt: The prompt template used for extracting features.
            description: Optional description of the category.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            The created category ID.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot add category: client has been closed")

        spec = AddSemanticCategorySpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            set_id=set_id,
            category_name=category_name,
            prompt=prompt,
            description=description,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/category",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            result = AddSemanticCategoryResponse(**response_data)
            logger.debug("Successfully added category: %s", result.category_id)
        except Exception:
            logger.exception("Failed to add category %s", category_name)
            raise
        else:
            return result.category_id

    def add_semantic_category_template(
        self,
        *,
        set_type_id: str,
        category_name: str,
        prompt: str,
        description: str | None = None,
        timeout: int | None = None,
    ) -> str:
        """
        Add a category template to a set type.

        Category templates are inherited by all set IDs mapped to the set type.

        Args:
            set_type_id: The set type ID to add the template to.
            category_name: Human-readable name for the category.
            prompt: The prompt template used for extracting features.
            description: Optional description of the category.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            The created category template ID.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot add category template: client has been closed")

        spec = AddSemanticCategoryTemplateSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            set_type_id=set_type_id,
            category_name=category_name,
            prompt=prompt,
            description=description,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/category/template",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            result = AddSemanticCategoryResponse(**response_data)
            logger.debug("Successfully added category template: %s", result.category_id)
        except Exception:
            logger.exception("Failed to add category template %s", category_name)
            raise
        else:
            return result.category_id

    def list_semantic_category_templates(
        self,
        set_type_id: str,
        timeout: int | None = None,
    ) -> builtins.list[SemanticCategoryTemplateEntry]:
        """
        List all category templates for a set type.

        Args:
            set_type_id: The set type ID to list templates for.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            List of category template entries.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot list category templates: client has been closed")

        spec = ListSemanticCategoryTemplatesSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            set_type_id=set_type_id,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/category/template/list",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            result = ListSemanticCategoryTemplatesResponse(**response_data)
            logger.debug(
                "Successfully listed %d category templates", len(result.categories)
            )
        except Exception:
            logger.exception("Failed to list category templates for %s", set_type_id)
            raise
        else:
            return result.categories

    def disable_semantic_category(
        self,
        *,
        set_id: str,
        category_name: str,
        timeout: int | None = None,
    ) -> bool:
        """
        Disable a semantic category for a specific set.

        Args:
            set_id: The set ID to disable the category for.
            category_name: The name of the category to disable.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            True if the category was disabled successfully.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot disable category: client has been closed")

        spec = DisableSemanticCategorySpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            set_id=set_id,
            category_name=category_name,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/category/disable",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            logger.debug(
                "Successfully disabled category %s for set %s", category_name, set_id
            )
        except Exception:
            logger.exception(
                "Failed to disable category %s for set %s", category_name, set_id
            )
            raise
        else:
            return True

    def get_semantic_category_set_ids(
        self,
        category_id: str,
        timeout: int | None = None,
    ) -> builtins.list[str]:
        """
        Get the set IDs that use a specific category.

        Args:
            category_id: The category ID to get set IDs for.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            List of set IDs using the category.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot get category set IDs: client has been closed")

        spec = GetSemanticCategorySetIdsSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            category_id=category_id,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/category/set_ids/get",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            result = GetSemanticCategorySetIdsResponse(**response_data)
            logger.debug(
                "Successfully retrieved %d set IDs for category %s",
                len(result.set_ids),
                category_id,
            )
        except Exception:
            logger.exception("Failed to get set IDs for category %s", category_id)
            raise
        else:
            return result.set_ids

    def delete_semantic_category(
        self,
        category_id: str,
        timeout: int | None = None,
    ) -> bool:
        """
        Delete a semantic category.

        Args:
            category_id: The unique identifier of the category to delete.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            True if deletion was successful.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot delete category: client has been closed")

        spec = DeleteSemanticCategorySpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            category_id=category_id,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/category/delete",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            logger.debug("Successfully deleted category: %s", category_id)
        except Exception:
            logger.exception("Failed to delete category %s", category_id)
            raise
        else:
            return True

    def add_semantic_tag(
        self,
        *,
        category_id: str,
        tag_name: str,
        tag_description: str,
        timeout: int | None = None,
    ) -> str:
        """
        Add a tag to a semantic category.

        Args:
            category_id: The category ID to add the tag to.
            tag_name: Human-readable name for the tag.
            tag_description: Description of what this tag represents.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            The created tag ID.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot add tag: client has been closed")

        spec = AddSemanticTagSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            category_id=category_id,
            tag_name=tag_name,
            tag_description=tag_description,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/category/tag",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            result = AddSemanticTagResponse(**response_data)
            logger.debug("Successfully added tag: %s", result.tag_id)
        except Exception:
            logger.exception(
                "Failed to add tag %s to category %s", tag_name, category_id
            )
            raise
        else:
            return result.tag_id

    def delete_semantic_tag(
        self,
        tag_id: str,
        timeout: int | None = None,
    ) -> bool:
        """
        Delete a semantic tag.

        Args:
            tag_id: The unique identifier of the tag to delete.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            True if deletion was successful.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError("Cannot delete tag: client has been closed")

        spec = DeleteSemanticTagSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            tag_id=tag_id,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memories/semantic/category/tag/delete",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            logger.debug("Successfully deleted tag: %s", tag_id)
        except Exception:
            logger.exception("Failed to delete tag %s", tag_id)
            raise
        else:
            return True

    # --- Episodic Memory Configuration Methods ---

    def get_episodic_memory_config(
        self,
        timeout: int | None = None,
    ) -> EpisodicMemoryConfigEntry:
        """
        Get episodic memory configuration for this project.

        Args:
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            The episodic memory configuration entry.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError(
                "Cannot get episodic memory config: client has been closed"
            )

        spec = GetEpisodicMemoryConfigSpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memory/episodic/config/get",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            result = EpisodicMemoryConfigEntry(**response_data)
            logger.debug("Successfully retrieved episodic memory config")
        except Exception:
            logger.exception("Failed to get episodic memory config")
            raise
        else:
            return result

    def configure_episodic_memory(
        self,
        *,
        enabled: bool | None = None,
        long_term_memory_enabled: bool | None = None,
        short_term_memory_enabled: bool | None = None,
        timeout: int | None = None,
    ) -> bool:
        """
        Configure episodic memory for this project.

        Allows enabling or disabling episodic memory components independently.
        Only provided (non-None) values are updated.

        Args:
            enabled: Whether episodic memory is enabled overall.
            long_term_memory_enabled: Whether long-term memory is enabled.
            short_term_memory_enabled: Whether short-term memory is enabled.
            timeout: Request timeout in seconds (uses client default if not provided).

        Returns:
            True if configuration was successful.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If the client has been closed.

        """
        if self._client_closed:
            raise RuntimeError(
                "Cannot configure episodic memory: client has been closed"
            )

        spec = ConfigureEpisodicMemorySpec(
            org_id=self.__org_id,
            project_id=self.__project_id,
            enabled=enabled,
            long_term_memory_enabled=long_term_memory_enabled,
            short_term_memory_enabled=short_term_memory_enabled,
        )
        v2_data = spec.model_dump(mode="json", exclude_none=True)

        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/memory/episodic/config",
                json=v2_data,
                timeout=timeout,
            )
            response.raise_for_status()
            logger.debug("Successfully configured episodic memory")
        except Exception:
            logger.exception("Failed to configure episodic memory")
            raise
        else:
            return True

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
