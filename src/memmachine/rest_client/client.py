"""Client utilities for interacting with the MemMachine HTTP API."""

import logging
from types import TracebackType
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .project import Project

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
        project = client.create_project(
            org_id="my_org",
            project_id="my_project",
            description="My project description"
        )
        # Or if you know the project already exists
        # project = client.get_project(org_id="my_org", project_id="my_project")

        # Create a memory instance from project
        memory = project.memory(
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

    def request(
        self,
        method: str,
        url: str,
        timeout: int | None = None,
        **kwargs: dict[str, Any],
    ) -> requests.Response:
        """
        Make an HTTP request using the client's session.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            timeout: Request timeout in seconds (uses client default if not provided)
            **kwargs: Additional arguments passed to requests.Session.request()

        Returns:
            Response object from the request

        Raises:
            requests.RequestException: If the request fails

        """
        request_timeout = timeout if timeout is not None else self.timeout
        return self._session.request(method, url, timeout=request_timeout, **kwargs)

    def create_project(
        self,
        org_id: str,
        project_id: str,
        description: str = "",
        embedder: str = "",
        reranker: str = "",
        timeout: int | None = None,
    ) -> Project:
        """
        Create a new project in MemMachine.

        Args:
            org_id: Organization identifier (required)
            project_id: Project identifier (required)
            description: Optional description for the project (default: "")
            embedder: Embedder model name to use (default: "").
                     Use "" to let server use its configured defaults, or specify a model name like "default".
            reranker: Reranker model name to use (default: "").
                     Use "" to let server use its configured defaults, or specify a model name like "default".
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            Project instance representing the created project

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        Example:
            ```python
            client = MemMachineClient(base_url="http://localhost:8080")
            project = client.create_project(
                org_id="my_org",
                project_id="my_project",
                description="My new project"
            )
            memory = project.memory(user_id="user123")
            ```

        """
        if self._closed:
            raise RuntimeError("Cannot create project: client has been closed")

        url = f"{self.base_url}/api/v2/projects"
        # Use user input directly - empty string for server defaults, or specific model name
        data = {
            "org_id": org_id,
            "project_id": project_id,
            "description": description,
            "config": {
                "embedder": embedder,
                "reranker": reranker,
            },
        }

        request_timeout = timeout if timeout is not None else self.timeout
        try:
            response = self._session.post(url, json=data, timeout=request_timeout)
            response.raise_for_status()
            project_data = response.json()
        except requests.RequestException:
            logger.exception("Failed to create project %s/%s", org_id, project_id)
            raise
        else:
            logger.debug("Project created: %s/%s", org_id, project_id)
            # Use server response data which contains actual config values
            return Project(
                client=self,
                org_id=org_id,
                project_id=project_id,
                description=project_data.get("description", description),
                # Server API uses "config" but Project class uses "configuration" for consistency
                configuration=project_data.get("config"),
                # Server does not return "metadata" in ProjectResponse, so we pass None
                metadata=None,
            )

    def get_project(
        self,
        org_id: str,
        project_id: str,
        timeout: int | None = None,
    ) -> Project:
        """
        Get an existing project from MemMachine.

        Args:
            org_id: Organization identifier (required)
            project_id: Project identifier (required)
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            Project instance representing the project

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        Example:
            ```python
            client = MemMachineClient(base_url="http://localhost:8080")
            project = client.get_project(
                org_id="my_org",
                project_id="my_project"
            )
            memory = project.memory(user_id="user123")
            ```

        """
        if self._closed:
            raise RuntimeError("Cannot get project: client has been closed")

        # Validate inputs
        if not org_id or not isinstance(org_id, str):
            raise ValueError("org_id must be a non-empty string")
        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string")
        if "/" in org_id:
            raise ValueError("org_id cannot contain '/'")
        if "/" in project_id:
            raise ValueError("project_id cannot contain '/'")

        url = f"{self.base_url}/api/v2/projects/get"
        data = {
            "org_id": org_id,
            "project_id": project_id,
        }

        try:
            response = self.request("POST", url, json=data, timeout=timeout)
            response.raise_for_status()
            project_data = response.json()

            return Project(
                client=self,
                org_id=org_id,
                project_id=project_id,
                description=project_data.get("description", ""),
                # Server API uses "config" but Project class uses "configuration" for consistency
                configuration=project_data.get("config"),
                # Server does not return "metadata" in ProjectResponse, so we pass None
                metadata=None,
            )
        except requests.HTTPError as e:
            if e.response.status_code == 422:
                # Try to get detailed error message from response
                try:
                    error_detail = e.response.json()
                    error_msg = f"Validation error (422): {error_detail}"
                except Exception:
                    error_msg = "Validation error (422): Invalid org_id or project_id format. Ensure they don't contain '/' and are non-empty strings."
                logger.exception(
                    "Failed to get project %s/%s: %s", org_id, project_id, error_msg
                )
                raise ValueError(error_msg) from e
            logger.exception("Failed to get project %s/%s", org_id, project_id)
            raise
        except requests.RequestException:
            logger.exception("Failed to get project %s/%s", org_id, project_id)
            raise

    def _validate_project_ids(self, org_id: str, project_id: str) -> None:
        """Validate org_id and project_id."""
        if not org_id or not isinstance(org_id, str):
            raise ValueError("org_id must be a non-empty string")
        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string")
        if "/" in org_id:
            raise ValueError("org_id cannot contain '/'")
        if "/" in project_id:
            raise ValueError("project_id cannot contain '/'")

    def _create_project_with_retry(
        self,
        org_id: str,
        project_id: str,
        description: str,
        embedder: str,
        reranker: str,
        timeout: int | None,
    ) -> Project:
        """Create project, handling concurrent creation (409) by fetching existing."""
        try:
            return self.create_project(
                org_id=org_id,
                project_id=project_id,
                description=description,
                embedder=embedder,
                reranker=reranker,
                timeout=timeout,
            )
        except requests.HTTPError as create_error:
            # If project was created between our get and create calls (409),
            # fetch the existing project
            if create_error.response.status_code == 409:
                logger.debug(
                    "Project %s/%s was created concurrently, fetching existing project",
                    org_id,
                    project_id,
                )
                return self.get_project(
                    org_id=org_id, project_id=project_id, timeout=timeout
                )
            # Re-raise other HTTP errors from create
            raise

    def get_or_create_project(
        self,
        org_id: str,
        project_id: str,
        description: str = "",
        embedder: str = "",
        reranker: str = "",
        timeout: int | None = None,
    ) -> Project:
        """
        Get an existing project or create it if it doesn't exist.

        This method first attempts to get the project. If it doesn't exist (404),
        it will create the project with the provided parameters. If the project
        already exists during creation (409), it will fetch the existing project.

        Args:
            org_id: Organization identifier (required)
            project_id: Project identifier (required)
            description: Optional description for the project (default: "").
                        Only used if project needs to be created.
            embedder: Embedder model name to use (default: "").
                     Only used if project needs to be created.
                     Use "" to let server use its configured defaults, or specify a model name like "default".
            reranker: Reranker model name to use (default: "").
                     Only used if project needs to be created.
                     Use "" to let server use its configured defaults, or specify a model name like "default".
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            Project instance representing the project (existing or newly created)

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed
            ValueError: If validation fails

        Example:
            ```python
            client = MemMachineClient(base_url="http://localhost:8080")
            # This will get the project if it exists, or create it if it doesn't
            project = client.get_or_create_project(
                org_id="my_org",
                project_id="my_project",
                description="My project description"
            )
            memory = project.memory(user_id="user123")
            ```

        """
        if self._closed:
            raise RuntimeError("Cannot get or create project: client has been closed")

        self._validate_project_ids(org_id, project_id)

        # First, try to get the project
        try:
            return self.get_project(
                org_id=org_id, project_id=project_id, timeout=timeout
            )
        except requests.HTTPError as e:
            # If project doesn't exist (404), create it
            if e.response.status_code == 404:
                return self._create_project_with_retry(
                    org_id, project_id, description, embedder, reranker, timeout
                )
            # Re-raise other HTTP errors from get
            raise

    def health_check(self, timeout: int | None = None) -> dict[str, Any]:
        """
        Check the health status of the MemMachine server.

        Args:
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            Dictionary containing health status information

        Raises:
            requests.RequestException: If the health check fails

        """
        request_timeout = timeout if timeout is not None else self.timeout
        try:
            response = self._session.get(
                f"{self.base_url}/api/v2/health",
                timeout=request_timeout,
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

    @property
    def closed(self) -> bool:
        """Check if the client has been closed."""
        return self._closed
