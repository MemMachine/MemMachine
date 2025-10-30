from datetime import datetime
from uuid import uuid4

import pytest

from memmachine.episodic_memory.declarative_memory.data_types import (
    ContentType,
    Derivative,
    Episode,
    EpisodeCluster,
)
from memmachine.episodic_memory.declarative_memory.derivative_mutator.metadata_derivative_mutator import (
    MetadataDerivativeMutator,
    MetadataDerivativeMutatorParams,
)


@pytest.mark.asyncio
async def test_metadata_derivative_mutator():
    mutator = MetadataDerivativeMutator(
        MetadataDerivativeMutatorParams(
            template='{"timestamp": $timestamp, "content": "$content", "prop": "$prop"}',
        )
    )

    source_episode_cluster = EpisodeCluster(
        uuid=uuid4(),
        episodes=[
            Episode(
                uuid=uuid4(),
                episode_type="test",
                content="One episode.",
                content_type=ContentType.STRING,
                timestamp=datetime.now(),
                filterable_properties={
                    "prop": "value1",
                },
                user_metadata={"key": "value1"},
            ),
            Episode(
                uuid=uuid4(),
                episode_type="test",
                content="Another episode.",
                content_type=ContentType.STRING,
                timestamp=datetime.now(),
                filterable_properties={
                    "prop": "value2",
                },
                user_metadata={"key": "value2"},
            ),
        ],
    )

    derivatives = [
        Derivative(
            uuid=uuid4(),
            derivative_type="test",
            content_type=ContentType.STRING,
            content="One episode.",
            timestamp=datetime.now(),
            filterable_properties={
                "prop": "value1",
            },
            user_metadata={"key": "value1"},
        ),
        Derivative(
            uuid=uuid4(),
            derivative_type="test",
            content_type=ContentType.STRING,
            content="Another episode.",
            timestamp=datetime.now(),
            filterable_properties={
                "prop": "value2",
            },
            user_metadata={"key": "value2"},
        ),
    ]

    for derivative in derivatives:
        mutated_derivatives = await mutator.mutate(
            derivative=derivative,
            source_episode_cluster=source_episode_cluster,
        )

        assert len(mutated_derivatives) == 1
        mutated_derivative = mutated_derivatives[0]

        assert mutated_derivative.derivative_type == derivative.derivative_type
        assert mutated_derivative.content_type == ContentType.STRING

        prop_value = derivative.filterable_properties.get("prop")
        assert (
            mutated_derivative.content
            == f'{{"timestamp": {derivative.timestamp}, "content": "{derivative.content}", "prop": "{prop_value}"}}'
        )
        assert mutated_derivative.timestamp == derivative.timestamp
        assert (
            mutated_derivative.filterable_properties == derivative.filterable_properties
        )
        assert mutated_derivative.user_metadata == derivative.user_metadata
