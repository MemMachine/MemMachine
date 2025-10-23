from datetime import datetime
from uuid import uuid4

import pytest

from memmachine.episodic_memory.declarative_memory.data_types import (
    ContentType,
    Episode,
    EpisodeCluster,
)
from memmachine.episodic_memory.declarative_memory.derivative_deriver.sentence_derivative_deriver import (
    SentenceDerivativeDeriver,
    SentenceDerivativeDeriverParams,
)


@pytest.mark.asyncio
async def test_sentence_derivative_deriver():
    deriver = SentenceDerivativeDeriver(SentenceDerivativeDeriverParams())
    episodes = [
        Episode(
            uuid=uuid4(),
            episode_type="test",
            content_type=ContentType.STRING,
            content="This is a sentence. Is this another sentence?\nHere is one more.",
            timestamp=datetime.now(),
            filterable_properties={
                "prop": "value1",
            },
        ),
        Episode(
            uuid=uuid4(),
            episode_type="test",
            content_type=ContentType.STRING,
            content="Yet another sentence, but with a comma.",
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
    assert len(derivatives) == 4
    for derivative in derivatives:
        assert derivative.content in (
            "This is a sentence.",
            "Is this another sentence?",
            "Here is one more.",
            "Yet another sentence, but with a comma.",
        )
        assert derivative.filterable_properties.get("prop") in ("value1", "value2")

    for content in (
        "This is a sentence.",
        "Is this another sentence?",
        "Here is one more.",
        "Yet another sentence, but with a comma.",
    ):
        assert any(derivative.content == content for derivative in derivatives)
