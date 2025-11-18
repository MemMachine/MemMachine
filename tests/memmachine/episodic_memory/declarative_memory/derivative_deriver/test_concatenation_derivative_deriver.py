from datetime import datetime
from uuid import uuid4

import pytest

from memmachine.episodic_memory.declarative_memory.data_types import (
    ContentType,
    Episode,
    EpisodeCluster,
)
from memmachine.episodic_memory.declarative_memory.derivative_deriver.concatenation_derivative_deriver import (
    ConcatenationDerivativeDeriver,
    ConcatenationDerivativeDeriverParams,
)


@pytest.mark.asyncio
async def test_concatenation_derivative_deriver():
    deriver = ConcatenationDerivativeDeriver(ConcatenationDerivativeDeriverParams())
    episodes = [
        Episode(
            uuid=uuid4(),
            episode_type="test",
            content_type=ContentType.STRING,
            content="One episode.",
            timestamp=datetime.now(),
            filterable_properties={
                "prop1": "value1",
                "prop2": "value1",
            },
        ),
        Episode(
            uuid=uuid4(),
            episode_type="test",
            content_type=ContentType.STRING,
            content="Another episode.",
            timestamp=datetime.now(),
            filterable_properties={
                "prop1": "value1",
                "prop2": "value2",
            },
        ),
    ]

    episode_cluster = EpisodeCluster(
        uuid=uuid4(),
        episodes=episodes,
        filterable_properties={
            "prop1": "value1",
        },
    )

    derivatives = await deriver.derive(episode_cluster)
    assert len(derivatives) == 1
    assert derivatives[0].content == "One episode.\nAnother episode."
    assert derivatives[0].filterable_properties.get("prop1") == "value1"
    assert derivatives[0].filterable_properties.get("prop2") is None
