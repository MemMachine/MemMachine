from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from memmachine.common.vector_graph_store import Node, VectorGraphStore
from memmachine.episodic_memory.declarative_memory.data_types import (
    ContentType,
    Episode,
    mangle_filterable_property_key,
)
from memmachine.episodic_memory.declarative_memory.related_episode_postulator.previous_related_episode_postulator import (
    PreviousRelatedEpisodePostulator,
    PreviousRelatedEpisodePostulatorParams,
)


@pytest.mark.asyncio
async def test_previous_related_episode_postulator():
    vector_graph_store = MagicMock(spec=VectorGraphStore)
    timestamp = datetime.now()

    def side_effect(*args, **kwargs):
        nodes = [
            Node(
                uuid=uuid4(),
                labels={"Episode"},
                properties={
                    "episode_type": "test",
                    "content_type": ContentType.STRING,
                    "content": "first",
                    "timestamp": timestamp,
                    mangle_filterable_property_key("user_id"): "user1",
                    "user_metadata": "null",
                },
            ),
            Node(
                uuid=uuid4(),
                labels={"Episode"},
                properties={
                    "episode_type": "test",
                    "content_type": ContentType.STRING,
                    "content": "second",
                    "timestamp": timestamp + timedelta(seconds=1),
                    mangle_filterable_property_key("user_id"): "user2",
                    "user_metadata": "null",
                },
            ),
        ]

        return [
            node
            for node in nodes
            if (
                node.properties["timestamp"] < kwargs.get("start_at_value")
                if "start_at_value" in kwargs
                else True
            )
            and all(
                node.properties.get(key) == value
                for key, value in kwargs.get("required_properties", {}).items()
            )
        ][: kwargs.get("limit", -1)]

    vector_graph_store.search_directional_nodes = AsyncMock(side_effect=side_effect)

    episode = Episode(
        uuid=uuid4(),
        episode_type="test",
        content_type=ContentType.STRING,
        content="Hello, world!",
        timestamp=timestamp + timedelta(seconds=2),
    )

    postulator = PreviousRelatedEpisodePostulator(
        PreviousRelatedEpisodePostulatorParams(
            vector_graph_store=vector_graph_store,
            search_limit=2,
            filterable_property_keys=[],
        )
    )

    related_episodes = await postulator.postulate(episode)
    assert len(related_episodes) == 2

    postulator = PreviousRelatedEpisodePostulator(
        PreviousRelatedEpisodePostulatorParams(
            vector_graph_store=vector_graph_store,
            search_limit=1,
            filterable_property_keys=[],
        )
    )

    related_episodes = await postulator.postulate(episode)
    assert len(related_episodes) == 1

    episode = Episode(
        uuid=uuid4(),
        episode_type="test",
        content_type=ContentType.STRING,
        content="Hello, world!",
        timestamp=timestamp + timedelta(seconds=1),
    )

    postulator = PreviousRelatedEpisodePostulator(
        PreviousRelatedEpisodePostulatorParams(
            vector_graph_store=vector_graph_store,
            search_limit=2,
            filterable_property_keys=[],
        )
    )

    related_episodes = await postulator.postulate(episode)
    assert len(related_episodes) == 1

    episode = Episode(
        uuid=uuid4(),
        episode_type="test",
        content_type=ContentType.STRING,
        content="Hello, world!",
        timestamp=timestamp + timedelta(seconds=2),
        filterable_properties={"user_id": "user1"},
    )

    postulator = PreviousRelatedEpisodePostulator(
        PreviousRelatedEpisodePostulatorParams(
            vector_graph_store=vector_graph_store,
            search_limit=2,
            filterable_property_keys={"user_id"},
        )
    )

    related_episodes = await postulator.postulate(episode)
    assert len(related_episodes) == 1
    assert related_episodes[0].content == "first"

    episode = Episode(
        uuid=uuid4(),
        episode_type="test",
        content_type=ContentType.STRING,
        content="Hello, world!",
        timestamp=timestamp + timedelta(seconds=2),
        filterable_properties={"user_id": "user2"},
    )

    postulator = PreviousRelatedEpisodePostulator(
        PreviousRelatedEpisodePostulatorParams(
            vector_graph_store=vector_graph_store,
            search_limit=2,
            filterable_property_keys={"user_id"},
        )
    )

    related_episodes = await postulator.postulate(episode)
    assert len(related_episodes) == 1
    assert related_episodes[0].content == "second"
