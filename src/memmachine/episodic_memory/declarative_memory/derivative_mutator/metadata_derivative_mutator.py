"""
A derivative mutator implementation
that formats the derivative content using metadata.

This can be used to improve searchability
by embedding metadata directly into the derivative content.
"""

from string import Template
from uuid import uuid4

from pydantic import BaseModel, Field

from ..data_types import Derivative, EpisodeCluster
from .derivative_mutator import DerivativeMutator


class MetadataDerivativeMutatorParams(BaseModel):
    """
    Parameters for MetadataDerivativeMutator.

    Attributes:
        template (str):
            Template string supporting $-substitutions
            for augmenting derivative content with metadata
            (default: "[$timestamp] $content").
    """

    template: str = Field(
        default="[$timestamp] $content",
        description=(
            "Template string supporting $-substitutions "
            "for augmenting derivative content with metadata"
        ),
    )


class MetadataDerivativeMutator(DerivativeMutator):
    """
    Derivative mutator that returns a metadata-formatted version
    of the original derivative.
    """

    def __init__(self, params: MetadataDerivativeMutatorParams):
        """
        Initialize a MetadataDerivativeMutator
        with the provided parameters.

        Args:
            params (MetadataDerivativeMutatorParams):
                Parameters for the MetadataDerivativeMutator.
        """
        super().__init__()

        self._template = Template(params.template)

    async def mutate(
        self,
        derivative: Derivative,
        source_episode_cluster: EpisodeCluster,
    ) -> list[Derivative]:
        mutated_content = self._template.safe_substitute(
            {
                "derivative_type": derivative.derivative_type,
                "content_type": derivative.content_type.value,
                "content": derivative.content,
                "timestamp": derivative.timestamp,
                "filterable_properties": derivative.filterable_properties,
                "user_metadata": derivative.user_metadata,
            },
            **{
                key: value
                for key, value in {
                    **derivative.filterable_properties,
                    **(
                        derivative.user_metadata
                        if isinstance(derivative.user_metadata, dict)
                        else {}
                    ),
                }.items()
            },
        )

        return [
            Derivative(
                uuid=uuid4(),
                derivative_type=derivative.derivative_type,
                content_type=derivative.content_type,
                content=mutated_content,
                timestamp=derivative.timestamp,
                filterable_properties=derivative.filterable_properties,
                user_metadata=derivative.user_metadata,
            )
        ]
