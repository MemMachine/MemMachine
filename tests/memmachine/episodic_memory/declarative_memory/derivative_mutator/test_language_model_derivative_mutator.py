from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from memmachine.common.language_model import LanguageModel
from memmachine.episodic_memory.declarative_memory.data_types import (
    ContentType,
    Derivative,
    Episode,
    EpisodeCluster,
)
from memmachine.episodic_memory.declarative_memory.derivative_mutator.language_model_derivative_mutator import (
    LanguageModelDerivativeMutator,
    LanguageModelDerivativeMutatorParams,
)


@pytest.mark.asyncio
async def test_language_model_derivative_mutator():
    language_model = MagicMock(spec=LanguageModel)
    language_model.generate_response = AsyncMock(
        return_value=(
            "Rewritten derivative content.",
            None,
        )
    )

    mutator = LanguageModelDerivativeMutator(
        LanguageModelDerivativeMutatorParams(
            language_model=language_model,
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
                    "prop1": "value1",
                    "prop2": "value1",
                },
                user_metadata={
                    "key1": "value1",
                    "key2": "value1",
                },
            ),
            Episode(
                uuid=uuid4(),
                episode_type="test",
                content="Another episode.",
                content_type=ContentType.STRING,
                timestamp=datetime.now(),
                filterable_properties={
                    "prop1": "value2",
                    "prop2": "value1",
                },
                user_metadata={
                    "key1": "value2",
                    "key2": "value1",
                },
            ),
        ],
        filterable_properties={
            "prop2": "value1",
        },
        user_metadata={
            "key2": "value1",
        },
    )

    derivatives = [
        Derivative(
            uuid=uuid4(),
            derivative_type="test",
            content_type=ContentType.STRING,
            content="One episode.",
            timestamp=datetime.now(),
            filterable_properties={
                "prop1": "value1",
                "prop2": "value1",
            },
            user_metadata={
                "key1": "value1",
                "key2": "value1",
            },
        ),
        Derivative(
            uuid=uuid4(),
            derivative_type="test",
            content_type=ContentType.STRING,
            content="Another episode.",
            timestamp=datetime.now(),
            filterable_properties={
                "prop1": "value2",
                "prop2": "value1",
            },
            user_metadata={
                "key1": "value2",
                "key2": "value1",
            },
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
        assert mutated_derivative.content == "Rewritten derivative content."
        assert mutated_derivative.timestamp == derivative.timestamp
        assert mutated_derivative.filterable_properties == {"prop2": "value1"}
        assert mutated_derivative.user_metadata == derivative.user_metadata
