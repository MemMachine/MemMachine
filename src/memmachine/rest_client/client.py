import logging
from typing import Any, Dict, List, Optional, Union
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

        # Create a memory instance
        memory = client.memory(
            group_id="my_group",
            agent_id="my_agent",
            user_id="user123",
            session_id="session456"
        )

        # Add memory
        memory.add("I like pizza", metadata={"type": "preference"})

        # Search memories
        results = memory.search("What do I like to eat?")
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        **kwargs,
    ):
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
        # base_url is required
        if base_url is None:
            raise ValueError(
                "base_url is required. Please provide it explicitly or set MEMORY_BACKEND_URL environment variable."
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
            }
        )

        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def memory(
        self,
        group_id: Optional[str] = None,
        agent_id: Optional[Union[str, List[str]]] = None,
        user_id: Optional[Union[str, List[str]]] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Memory:
        """
        Create a Memory instance for a specific context.

        Args:
            group_id: Group identifier for the memory context
            agent_id: Agent identifier(s) for the memory context
            user_id: User identifier(s) for the memory context
            session_id: Session identifier for the memory context
            **kwargs: Additional configuration options

        Returns:
            Memory instance configured for the specified context
        """
        memory = Memory(
            client=self,
            group_id=group_id,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            **kwargs,
        )
        # Track the Memory object
        self._memory_objects.add(memory)
        return memory

    def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the MemMachine server.

        Returns:
            Dictionary containing health status information

        Raises:
            requests.RequestException: If the health check fails
        """
        try:
            response = self._session.get(
                f"{self.base_url}/health", timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Health check failed: {e}")
            raise

    def close(self):
        """Close the client and clean up resources."""
        if self._closed:
            return

        self._closed = True

        # Mark all tracked Memory objects as closed
        for memory in self._memory_objects:
            memory._client_closed = True

        # Clear the tracking set
        self._memory_objects.clear()

        # Close the session
        if hasattr(self, "_session"):
            self._session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self):
        return f"MemMachineClient(base_url='{self.base_url}')"
