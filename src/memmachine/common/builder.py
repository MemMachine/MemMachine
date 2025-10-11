"""
Abstract base class for a builder that construct resources
based on their definitions and dependencies.
"""

from abc import ABC, abstractmethod
from typing import Any


class Builder(ABC):
    """
    Abstract base class for a builder that construct resources
    based on their definitions and dependencies.
    """

    @staticmethod
    @abstractmethod
    def get_dependency_ids(variant: str, config: dict[str, Any]) -> set[str]:
        """
        Get the set of dependency IDs
        required for building the resource.

        Args:
            variant (str):
                The variant of the resource to build.
            config (dict[str, Any]):
                The configuration dictionary for the resource.

        Returns:
            set[str]:
                A set of dependency IDs
                required for building the resource.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build(variant: str, config: dict[str, Any], injections: dict[str, Any]) -> Any:
        """
        Build the resource
        based on its variant,
        configuration,
        and injected dependencies.

        Args:
            variant (str):
                The variant of the resource to build.
            config (dict[str, Any]):
                The configuration dictionary for the resource.
            injections (dict[str, Any]):
                A dictionary of injected dependencies,
                where keys are dependency IDs
                and values are the corresponding resource instances.
        """
        raise NotImplementedError
