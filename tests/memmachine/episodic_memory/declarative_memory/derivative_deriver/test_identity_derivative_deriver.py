from datetime import datetime
from uuid import uuid4

import pytest

from memmachine.episodic_memory.declarative_memory.data_types import (
    ContentType,
    Episode,
    EpisodeCluster,
)
from memmachine.episodic_memory.declarative_memory.derivative_deriver.identity_derivative_deriver import (
    IdentityDerivativeDeriver,
    IdentityDerivativeDeriverParams,
)


@pytest.mark.asyncio
async def test_identity_derivative_deriver():
    deriver = IdentityDerivativeDeriver(IdentityDerivativeDeriverParams())
    episodes = [
        Episode(
            uuid=uuid4(),
            episode_type="test",
            content_type=ContentType.STRING,
            content="One episode.",
            timestamp=datetime.now(),
            filterable_properties={
                "prop": "value1",
            },
        ),
        Episode(
            uuid=uuid4(),
            episode_type="test",
            content_type=ContentType.STRING,
            content="Another episode.",
            timestamp=datetime.now(),
            filterable_properties={
                "prop": "value2",
            },
        ),
    ]

    episode_cluster = EpisodeCluster(
        uuid=uuid4(),
        episodes=episodes,
    )

    derivatives = await deriver.derive(episode_cluster)
    assert len(derivatives) == 2
    for derivative in derivatives:
        assert derivative.content in ("One episode.", "Another episode.")
        assert derivative.filterable_properties.get("prop") in ("value1", "value2")

    for content in ("One episode.", "Another episode."):
        assert any(derivative.content == content for derivative in derivatives)

    for prop_value in ("value1", "value2"):
        assert any(
            derivative.filterable_properties.get("prop") == prop_value
            for derivative in derivatives
        )
