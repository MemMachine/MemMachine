"""
MemMachine HTTP client for API operations.
"""

import logging
from typing import Any, Dict, Optional

import requests

from ..schemas import DeleteRequest, MemoryEpisode, SearchQuery
from ..settings import settings


class MemMachineClient:
    """HTTP client for MemMachine API operations."""

    def __init__(
        self,
        base_url: str,
        timeout: int = settings.request_timeout,
        verify_ssl: bool = True,
    ):
        """Initialize the MemMachine client.

        Args:
            base_url: Base URL for the MemMachine API
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates (default: True)
                       Set to False for self-signed certificates
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.logger = logging.getLogger(f"{__name__}.MemMachineClient")

        # Log SSL verification setting for debugging
        if not self.verify_ssl:
            self.logger.warning(
                "SSL certificate verification is DISABLED. This should only be used for development/testing with self-signed certificates."
            )

    def _make_request(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an HTTP request to the MemMachine API.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path
            data: Request data for POST/PUT requests

        Returns:
            Response data as dictionary

        Raises:
            requests.exceptions.RequestException: For HTTP errors
            Exception: For other errors
        """
        url = f"{self.base_url}{endpoint}"

        try:
            self.logger.debug(f"Making {method} request to {url}")
            if data:
                self.logger.debug(f"Request data: {data}")

            response = requests.request(
                method=method,
                json=data,
                url=url,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            response.raise_for_status()

            return response.json() if response.content else {}
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error in {method} {url}: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    response_content = e.response.text
                    self.logger.error(f"Response status code: {e.response.status_code}")
                    self.logger.error(f"Response content: {response_content}")
                except Exception as parse_error:
                    self.logger.error(f"Could not parse error response: {parse_error}")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP error in {method} {url}: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    response_content = e.response.text
                    self.logger.error(f"Response content: {response_content}")
                except Exception as parse_error:
                    self.logger.error(f"Could not parse error response: {parse_error}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in {method} {url}: {e}")
            raise

    def add_memory(self, episode_data: MemoryEpisode) -> Dict[str, Any]:
        """Add a memory episode to MemMachine.

        Args:
            episode_data: Memory episode data

        Returns:
            Response data from the API
        """
        self.logger.info(f"Adding memory - Episode data: {episode_data.model_dump()}")
        return self._make_request(
            method="POST", endpoint="/v1/memories", data=episode_data.model_dump()
        )

    def search_memory(self, search_data: SearchQuery) -> Dict[str, Any]:
        """Search for memories in MemMachine.

        Args:
            search_data: Search query data

        Returns:
            Search results from the API
        """
        return self._make_request(
            method="POST", endpoint="/v1/memories/search", data=search_data.model_dump()
        )

    # def get_episodic_memory(self, user_id: str) -> Dict[str, Any]:
    #     """Get episodic memory from MemMachine.

    #     Args:
    #         episodic_data: Episodic memory data

    #     Returns:
    #         Episodic memory from the API
    #     """
    #     return self._make_request(
    #         method="GET", endpoint="/v1/memories/episodic/search", data=episodic_data.model_dump()
    #     )

    def get_profile_memory(self, user_id: str, limit: int) -> Dict[str, Any]:
        """Get profile memory from MemMachine.

        Returns:
            Profile memory from the API
        """
        return self._make_request(
            method="GET",
            endpoint="/v1/memories/profile/search",
            params={"user_id": user_id, "limit": limit},
        )

    def delete_session_memory(self, delete_data: DeleteRequest) -> Dict[str, Any]:
        """Delete all memories for a session.

        Args:
            delete_data: Delete request data

        Returns:
            Response data from the API
        """
        return self._make_request(
            method="DELETE", endpoint="/v1/memories", data=delete_data.model_dump()
        )

    def get_sessions_for_user(self, user_id: str) -> Dict[str, Any]:
        """Get all sessions for a specific user.

        Args:
            user_id: User identifier

        Returns:
            Response data containing user sessions
        """
        self.logger.info(f"Getting sessions for user: {user_id}")
        return self._make_request(
            method="GET", endpoint=f"/v1/users/{user_id}/sessions"
        )


# Create the global client instance
memmachine_client = MemMachineClient(
    settings.mm_backend_url, verify_ssl=settings.verify_ssl
)
