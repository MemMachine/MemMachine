"""
A derivative mutator implementation that does not mutate derivatives.

This can be used as a default implementation
when mutation is not required.
"""

from uuid import uuid4

from ..data_types import Derivative, EpisodeCluster
from .derivative_mutator import DerivativeMutator


class IdentityDerivativeMutator(DerivativeMutator):
    """
    Derivative mutator that returns a copy of the original derivative.
    """

    async def mutate(
        self,
        derivative: Derivative,
        source_episode_cluster: EpisodeCluster,
    ) -> list[Derivative]:
        return [
            Derivative(
                uuid=uuid4(),
                derivative_type=derivative.derivative_type,
                content_type=derivative.content_type,
                content=derivative.content,
                timestamp=derivative.timestamp,
                filterable_properties=derivative.filterable_properties,
                user_metadata=derivative.user_metadata,
            )
        ]
