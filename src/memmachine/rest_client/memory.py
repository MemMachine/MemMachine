"""
Memory management interface for MemMachine.

This module provides the Memory class that handles episodic and profile memory
operations for a specific context.
"""

import logging
from typing import Any
from uuid import uuid4

import requests

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
            group_id="my_group",
            agent_id="my_agent",
            user_id="user123",
            session_id="session456"
        )

        # Add a memory
        memory.add("I like pizza", metadata={"type": "preference"})

        # Search memories
        results = memory.search("What do I like to eat?")
        ```
    """

    def __init__(
        self,
        client,
        group_id: str | None = None,
        agent_id: str | list[str] | None = None,
        user_id: str | list[str] | None = None,
        session_id: str | None = None,
        **kwargs,
    ):
        """
        Initialize Memory instance.

        Args:
            client: MemMachineClient instance
            group_id: Group identifier
            agent_id: Agent identifier(s)
            user_id: User identifier(s)
            session_id: Session identifier
            **kwargs: Additional configuration options
        """
        self.client = client
        self._client_closed = False

        # Store group_id as private attribute
        self.__group_id = group_id

        # Normalize agent_id and user_id to lists and store as private attributes
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

        # Store session_id as private attribute
        self.__session_id = session_id or f"session_{uuid4().hex}"

        # Validate required fields
        if not self.__user_id or not self.__agent_id:
            raise ValueError(
                "Both user_id and agent_id are required and cannot be empty"
            )

        # Ensure group_id is non-empty to avoid server defaulting issues
        # Since user_id is validated above, we can safely use the first element
        if not self.__group_id:
            self.__group_id = self.__user_id[0]

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
    def session_id(self) -> str:
        """
        Get the session_id (read-only).

        Returns:
            Session identifier
        """
        return self.__session_id

    def add(
        self,
        content: str,
        producer: str | None = None,
        produced_for: str | None = None,
        episode_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Add a memory episode.

        Args:
            content: The content to store in memory
            producer: Who produced this content (defaults to first user_id)
            produced_for: Who this content is for (defaults to first agent_id)
            episode_type: Type of episode (default: "text")
            metadata: Additional metadata for the episode

        Returns:
            True if the memory was added successfully

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed
        """
        if self._client_closed:
            raise RuntimeError("Cannot add memory: client has been closed")

        # Set default producer and produced_for to match the session context
        # These must be in the user_id or agent_id lists sent in headers
        # Since __init__ validates that user_id and agent_id are not empty,
        # we can safely use the first element
        if self.__user_id is None or len(self.__user_id) == 0:
            raise RuntimeError(
                "user_id must not be None or empty. This should have been validated in __init__."
            )
        if self.__agent_id is None or len(self.__agent_id) == 0:
            raise RuntimeError(
                "agent_id must not be None or empty. This should have been validated in __init__."
            )
        if not producer:
            producer = self.__user_id[0]
        if not produced_for:
            produced_for = self.__agent_id[0]

        # Validate that producer and produced_for are in the session context
        # Server requires these to be in either user_id or agent_id lists
        # Since __init__ validates that user_id and agent_id are not empty,
        # we can safely use them directly
        if producer not in self.__user_id and producer not in self.__agent_id:
            raise ValueError(
                f"producer '{producer}' must be in user_id {self.__user_id} or agent_id {self.__agent_id}. "
                f"Current context: user_id={self.__user_id}, agent_id={self.__agent_id}"
            )
        if produced_for not in self.__user_id and produced_for not in self.__agent_id:
            raise ValueError(
                f"produced_for '{produced_for}' must be in user_id {self.__user_id} or agent_id {self.__agent_id}. "
                f"Current context: user_id={self.__user_id}, agent_id={self.__agent_id}"
            )

        episode_data = {
            "producer": producer,
            "produced_for": produced_for,
            "episode_content": content,
            "episode_type": episode_type,
            "metadata": metadata or {},
        }

        # Prepare session headers - these must match what the server expects
        # Important: The user_id and agent_id in headers must match what was used
        # when the session was created, or the session must be recreated
        headers = {}
        if self.__group_id:
            headers["group-id"] = self.__group_id
        if self.__session_id:
            headers["session-id"] = self.__session_id
        if self.__agent_id:
            headers["agent-id"] = ",".join(self.__agent_id)
        if self.__user_id:
            headers["user-id"] = ",".join(self.__user_id)

        # Log the request details for debugging
        logger.debug(
            f"Adding memory: producer={producer}, produced_for={produced_for}, "
            f"user_id={self.__user_id}, agent_id={self.__agent_id}, "
            f"group_id={self.__group_id}, session_id={self.__session_id}"
        )

        try:
            response = self.client._session.post(
                f"{self.client.base_url}/v1/memories",
                json=episode_data,
                headers=headers,
                timeout=self.client.timeout,
            )
            response.raise_for_status()
            logger.debug(f"Successfully added memory: {content[:50]}...")
            return True
        except requests.RequestException as e:
            # Try to get detailed error information from response
            error_detail = ""
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = f" Response: {e.response.text}"
                except Exception:
                    error_detail = f" Status: {e.response.status_code}"
            logger.error(f"Failed to add memory: {e}{error_detail}")
            raise
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise

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

        search_data = {"query": query, "filter": filter_dict, "limit": limit}

        # Prepare session headers
        headers = {}
        if self.__group_id:
            headers["group-id"] = self.__group_id
        if self.__session_id:
            headers["session-id"] = self.__session_id
        if self.__agent_id:
            headers["agent-id"] = ",".join(self.__agent_id)
        if self.__user_id:
            headers["user-id"] = ",".join(self.__user_id)

        try:
            response = self.client._session.post(
                f"{self.client.base_url}/v1/memories/search",
                json=search_data,
                headers=headers,
                timeout=self.client.timeout,
            )
            response.raise_for_status()
            data = response.json()
            logger.info(f"Search completed for query: {query}")
            return data.get("content", {})
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise

    def get_context(self) -> dict[str, Any]:
        """
        Get the current memory context.

        Returns:
            Dictionary containing the context information
        """
        return {
            "group_id": self.__group_id,
            "agent_id": self.__agent_id,
            "user_id": self.__user_id,
            "session_id": self.__session_id,
        }

    def __repr__(self):
        return f"Memory(group_id='{self.group_id}', user_id='{self.user_id}', session_id='{self.session_id}')"
