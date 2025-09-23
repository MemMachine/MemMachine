"""
Abstract base class for a derivative mutator.

Defines an interface for creating mutations from a given derivative.
"""

from abc import ABC, abstractmethod

from ..data_types import Derivative, EpisodeCluster


class DerivativeMutator(ABC):
    """
    Abstract base class for a derivative mutator.
    """

    @abstractmethod
    async def mutate(
        self,
        derivative: Derivative,
        source_episode_cluster: EpisodeCluster,
    ) -> list[Derivative]:
        """
        Mutate a given derivative to create new derivatives.

        Args:
            derivative (Derivative):
                The input derivative to mutate.
            source_episode_cluster (EpisodeCluster):
                The source episode cluster
                of the derivative provided for context.

        Returns:
            list[Derivative]:
                A list of mutated derivatives.
        """
        raise NotImplementedError
