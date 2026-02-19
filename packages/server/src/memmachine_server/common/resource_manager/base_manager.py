"""Base class for resource managers with common status tracking functionality."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from asyncio import Lock
from typing import TypeVar

from memmachine_common.api.config_spec import ResourceStatus

logger = logging.getLogger(__name__)

# Type variable for the resource type (Embedder, LanguageModel, Reranker, etc.)
ResourceT = TypeVar("ResourceT")


class BaseResourceManager[ResourceT](ABC):
    """
    Base class providing common resource status tracking and retry functionality.

    Subclasses must implement:
    - `_is_configured(name)`: Check if a resource is configured
    - `_get_resource(name, validate)`: Build/retrieve a resource
    - `_get_not_found_error(name)`: Return the appropriate error for unknown resources
    - `get_all_names()`: Return all configured resource names
    """

    def __init__(self) -> None:
        """Initialize the base manager with error tracking."""
        self._build_errors: dict[str, Exception] = {}
        self._resources: dict[str, ResourceT] = {}
        self._lock = Lock()
        self._resource_locks: dict[str, Lock] = {}

    @abstractmethod
    def _is_configured(self, name: str) -> bool:
        """Check if a resource with the given name is configured."""
        raise NotImplementedError

    @abstractmethod
    async def _build_resource(self, name: str, validate: bool = False) -> ResourceT:
        """Build a resource by name. Called within the locking context."""
        raise NotImplementedError

    @abstractmethod
    def _get_not_found_error(self, name: str) -> Exception:
        """Return the appropriate 'not found' error for the resource type."""
        raise NotImplementedError

    @abstractmethod
    def get_all_names(self) -> set[str]:
        """Return all configured resource names."""
        raise NotImplementedError

    @property
    @abstractmethod
    def _resource_type_name(self) -> str:
        """Return the human-readable name of the resource type (e.g., 'embedder')."""
        raise NotImplementedError

    def get_resource_status(self, name: str) -> ResourceStatus:
        """Return the status of a named resource."""
        if name in self._resources:
            return ResourceStatus.READY
        if name in self._build_errors:
            return ResourceStatus.FAILED
        if self._is_configured(name):
            return ResourceStatus.PENDING
        raise self._get_not_found_error(name)

    def get_resource_error(self, name: str) -> Exception | None:
        """Return the build error for a failed resource, or None if not failed."""
        return self._build_errors.get(name)

    async def _try_build_with_error_tracking(self, name: str) -> None:
        """Attempt to build a resource, tracking failures in _build_errors."""
        try:
            await self._get_resource_with_locking(name)
        except Exception as e:
            logger.warning(
                "Failed to build %s '%s': %s", self._resource_type_name, name, e
            )
            self._build_errors[name] = e

    async def _get_resource_with_locking(
        self, name: str, validate: bool = False
    ) -> ResourceT:
        """
        Get a resource with proper locking and error checking.

        This implements the common pattern of:
        1. Check cache
        2. Check for previous build errors
        3. Acquire lock
        4. Double-check cache and errors
        5. Build resource
        """
        from memmachine_server.common.errors import ResourceNotReadyError

        # Return cached if already built
        if name in self._resources:
            return self._resources[name]

        # Check if this resource previously failed to build
        if name in self._build_errors:
            raise ResourceNotReadyError(
                f"{self._resource_type_name.capitalize()} '{name}' is not ready: "
                f"{self._build_errors[name]}"
            )

        # Ensure a lock exists for this resource
        if name not in self._resource_locks:
            async with self._lock:
                self._resource_locks.setdefault(name, Lock())

        async with self._resource_locks[name]:
            # Double-checked locking
            if name in self._resources:
                return self._resources[name]

            # Check again after acquiring lock
            if name in self._build_errors:
                raise ResourceNotReadyError(
                    f"{self._resource_type_name.capitalize()} '{name}' is not ready: "
                    f"{self._build_errors[name]}"
                )

            # Build the resource
            resource = await self._build_resource(name, validate=validate)
            self._resources[name] = resource
            return resource

    async def retry_build(self, name: str, validate: bool = False) -> ResourceT:
        """
        Retry building a failed resource.

        Clears the previous error and attempts to build again.
        """
        if (
            name not in self._build_errors
            and name not in self._resources
            and not self._is_configured(name)
        ):
            raise self._get_not_found_error(name)
        # Clear error to allow retry
        self._build_errors.pop(name, None)
        return await self._get_resource_with_locking(name, validate=validate)

    async def build_all_with_error_tracking(
        self, names: set[str] | list[str]
    ) -> dict[str, ResourceT]:
        """
        Build all resources, catching and logging failures.

        Failures are tracked in _build_errors, allowing startup to continue
        even if some resources fail to initialize.
        """
        await asyncio.gather(
            *[self._try_build_with_error_tracking(name) for name in names]
        )
        return self._resources

    def _remove_from_cache(self, name: str) -> bool:
        """
        Remove a resource from cache and error tracking.

        Returns True if anything was removed, False otherwise.
        Subclasses should call this and also handle config removal.
        """
        removed = False
        if name in self._resources:
            del self._resources[name]
            removed = True
        if name in self._build_errors:
            del self._build_errors[name]
            removed = True
        return removed

    def clear_build_error(self, name: str) -> None:
        """Clear any build error for a resource, allowing it to be retried."""
        self._build_errors.pop(name, None)
