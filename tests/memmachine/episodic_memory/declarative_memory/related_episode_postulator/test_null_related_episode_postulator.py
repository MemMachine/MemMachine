from datetime import datetime
from uuid import uuid4

import pytest

from memmachine.episodic_memory.declarative_memory.data_types import (
    ContentType,
    Episode,
)
from memmachine.episodic_memory.declarative_memory.related_episode_postulator.null_related_episode_postulator import (
    NullRelatedEpisodePostulator,
)


@pytest.mark.asyncio
async def test_null_related_episode_postulator():
    postulator = NullRelatedEpisodePostulator()
    episode = Episode(
        uuid=uuid4(),
        episode_type="test",
        content_type=ContentType.STRING,
        content="Hello, world!",
        timestamp=datetime.now(),
    )
    related_episodes = await postulator.postulate(episode)
    assert related_episodes == []
