"""
Abstract base class for a derivative deriver.

Defines an interface for deriving derivatives
from a given episode cluster.
"""

from abc import ABC, abstractmethod

from ..data_types import Derivative, EpisodeCluster


class DerivativeDeriver(ABC):
    """
    Abstract base class for a derivative deriver.
    """

    @abstractmethod
    async def derive(self, episode_cluster: EpisodeCluster) -> list[Derivative]:
        """
        Derive derivatives from a given episode cluster.

        Args:
            episode_cluster (EpisodeCluster):
                The input episode cluster
                from which to derive derivatives.

        Returns:
            list[Derivative]:
                A list of derived derivatives.
        """
        raise NotImplementedError
