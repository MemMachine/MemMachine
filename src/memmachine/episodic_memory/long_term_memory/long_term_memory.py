from typing import Any, cast

from ..data_types import ContentType, Episode
from .declarative_memory import ContentType as DeclarativeMemoryContentType
from .declarative_memory import DeclarativeMemory
from .declarative_memory import Episode as DeclarativeMemoryEpisode

content_type_to_declarative_memory_content_type_map = {
    ContentType.STRING: DeclarativeMemoryContentType.STRING,
}

declarative_memory_content_type_to_content_type_map = {
    DeclarativeMemoryContentType.STRING: ContentType.STRING,
}


class LongTermMemory:
    def __init__(self, config: dict[str, Any]):
        declarative_memory = config.get("declarative_memory")
        if not isinstance(declarative_memory, DeclarativeMemory):
            raise TypeError(
                "declarative_memory must be an instance of DeclarativeMemory"
            )
        self._declarative_memory = declarative_memory

    async def add_episode(self, episode: Episode):
        declarative_memory_episode = DeclarativeMemoryEpisode(
            uuid=episode.uuid,
            episode_type="default",
            content_type=content_type_to_declarative_memory_content_type_map[
                episode.content_type
            ],
            content=episode.content,
            timestamp=episode.timestamp,
            filterable_properties={
                key: value
                for key, value in {
                    "group_id": episode.group_id,
                    "session_id": episode.session_id,
                    "producer_id": episode.producer_id,
                    "produced_for_id": episode.produced_for_id,
                }.items()
                if value is not None
            },
            user_metadata=episode.user_metadata,
        )
        await self._declarative_memory.add_episode(declarative_memory_episode)

    async def search(
        self,
        query: str,
        num_episodes_limit: int,
        id_filter: dict[str, str] = {},
    ):
        declarative_memory_episodes = await self._declarative_memory.search(
            query,
            num_episodes_limit=num_episodes_limit,
            filterable_properties=dict(id_filter),
        )
        return [
            Episode(
                uuid=declarative_memory_episode.uuid,
                episode_type=declarative_memory_episode.episode_type,
                content_type=(
                    declarative_memory_content_type_to_content_type_map[
                        declarative_memory_episode.content_type
                    ]
                ),
                content=declarative_memory_episode.content,
                timestamp=declarative_memory_episode.timestamp,
                group_id=cast(
                    str,
                    declarative_memory_episode.filterable_properties.get(
                        "group_id", ""
                    ),
                ),
                session_id=cast(
                    str,
                    declarative_memory_episode.filterable_properties.get(
                        "session_id", ""
                    ),
                ),
                producer_id=cast(
                    str,
                    (
                        declarative_memory_episode.filterable_properties.get(
                            "producer_id", ""
                        )
                    ),
                ),
                produced_for_id=cast(
                    str | None,
                    (
                        declarative_memory_episode.filterable_properties.get(
                            "produced_for_id", ""
                        )
                    ),
                ),
                user_metadata=declarative_memory_episode.user_metadata,
            )
            for declarative_memory_episode in declarative_memory_episodes
        ]

    async def clear(self):
        self._declarative_memory.forget_all()

    async def forget_session(self, group_id: str, session_id: str):
        await self._declarative_memory.forget_filtered_episodes(
            filterable_properties={
                "group_id": group_id,
                "session_id": session_id,
            }
        )
