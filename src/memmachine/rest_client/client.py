"""Client utilities for interacting with the MemMachine HTTP API."""

import logging
from types import TracebackType
from typing import Any
from weakref import WeakSet

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .memory import Memory

logger = logging.getLogger(__name__)


class MemMachineClient:
    """
    Main client class for interacting with MemMachine memory system.

    This client provides a high-level interface for managing episodic and profile
    memory. It handles authentication and provides convenient methods for memory operations.

    Example:
        ```python
        from memmachine import MemMachineClient

        # Initialize client
        client = MemMachineClient(
            api_key="your_api_key",
            base_url="http://localhost:8080"
        )

        # Create a project (optional, project is auto-created on first use)
        client.create_project(
            org_id="my_org",
            project_id="my_project",
            description="My project description"
        )

        # Create a memory instance (v2 API requires org_id and project_id)
        memory = client.memory(
            org_id="my_org",
            project_id="my_project",
            group_id="my_group",  # Optional: stored in metadata
            agent_id="my_agent",  # Optional: stored in metadata
            user_id="user123",    # Optional: stored in metadata
            session_id="session456"  # Optional: stored in metadata
        )

        # Add memory (role defaults to "user")
        memory.add("I like pizza")
        
        # Add assistant response
        memory.add("I understand you like pizza", role="assistant")

        # Search memories
        results = memory.search("What do I like to eat?")
        ```

    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Initialize the MemMachine client.

        Args:
            api_key: API key for authentication (optional for local development)
            base_url: Base URL of the MemMachine server (required).
                     Should be provided explicitly or via MEMORY_BACKEND_URL environment variable.
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            **kwargs: Additional configuration options

        Raises:
            ValueError: If base_url is not provided

        """
        self.api_key = api_key
        self._extra_options = kwargs
        # base_url is required
        if base_url is None:
            raise ValueError(
                "base_url is required. Please provide it explicitly or set MEMORY_BACKEND_URL environment variable.",
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._closed = False

        # Track Memory objects created by this client (using WeakSet to avoid circular references)
        self._memory_objects: WeakSet[Memory] = WeakSet()

        # Setup session with retry strategy
        self._session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # Set default headers
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "MemMachineClient/1.0.0",
            },
        )

        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def create_project(
        self,
        org_id: str,
        project_id: str,
        description: str = "",
        embedder: str = "default",
        reranker: str = "default",
    ) -> bool:
        """
        Create a new project in MemMachine.

        Args:
            org_id: Organization identifier (required)
            project_id: Project identifier (required)
            description: Optional description for the project (default: "")
            embedder: Embedder model name to use (default: "default")
            reranker: Reranker model name to use (default: "default")

        Returns:
            True if the project was created successfully

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        Example:
            ```python
            client = MemMachineClient(base_url="http://localhost:8080")
            client.create_project(
                org_id="my_org",
                project_id="my_project",
                description="My new project"
            )
            ```

        """
        if self._closed:
            raise RuntimeError("Cannot create project: client has been closed")

        url = f"{self.base_url}/api/v2/projects"
        data = {
            "org_id": org_id,
            "project_id": project_id,
            "description": description,
            "config": {
                "embedder": embedder,
                "reranker": reranker,
            },
        }

        try:
            response = self._session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            logger.debug(f"Project created: {org_id}/{project_id}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to create project {org_id}/{project_id}: {e}")
            raise

    def memory(
        self,
        org_id: str,
        project_id: str,
        group_id: str | None = None,
        agent_id: str | list[str] | None = None,
        user_id: str | list[str] | None = None,
        session_id: str | None = None,
        **kwargs: dict[str, Any],
    ) -> Memory:
        """
        Create a Memory instance for a specific context.

        Args:
            org_id: Organization identifier (required for v2 API)
            project_id: Project identifier (required for v2 API)
            group_id: Group identifier (optional, will be stored in metadata)
            agent_id: Agent identifier(s) (optional, will be stored in metadata)
            user_id: User identifier(s) (optional, will be stored in metadata)
            session_id: Session identifier (optional, will be stored in metadata)
            **kwargs: Additional configuration options

        Returns:
            Memory instance configured for the specified context

        """
        memory = Memory(
            client=self,
            org_id=org_id,
            project_id=project_id,
            group_id=group_id,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            **kwargs,
        )
        # Track the Memory object
        self._memory_objects.add(memory)
        return memory

    def health_check(self) -> dict[str, Any]:
        """
        Check the health status of the MemMachine server.

        Returns:
            Dictionary containing health status information

        Raises:
            requests.RequestException: If the health check fails

        """
        try:
            response = self._session.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            logger.exception("Health check failed")
            raise

    def close(self) -> None:
        """Close the client and clean up resources."""
        if self._closed:
            return

        self._closed = True

        # Mark all tracked Memory objects as closed
        for memory in self._memory_objects:
            memory.mark_client_closed()

        # Clear the tracking set
        self._memory_objects.clear()

        # Close the session
        if hasattr(self, "_session"):
            self._session.close()

    def __enter__(self) -> "MemMachineClient":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return f"MemMachineClient(base_url='{self.base_url}')"

    @property
    def session(self) -> requests.Session:
        """Expose the underlying requests session for advanced usage."""
        return self._session
