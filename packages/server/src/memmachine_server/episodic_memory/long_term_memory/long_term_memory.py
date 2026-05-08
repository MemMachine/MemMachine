"""Long-term memory facade with declarative + event backends."""

import datetime
from collections.abc import Iterable
from typing import Annotated, Literal, cast
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, Field, InstanceOf, JsonValue

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.episode_store import (
    ContentType,
    Episode,
    EpisodeStorage,
    EpisodeType,
)
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
    map_filter_fields,
    normalize_filter_field,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.vector_graph_store import VectorGraphStore
from memmachine_server.common.vector_store import (
    VectorStore,
    VectorStoreCollection,
)
from memmachine_server.episodic_memory.declarative_memory import (
    DeclarativeMemory,
    DeclarativeMemoryParams,
)
from memmachine_server.episodic_memory.declarative_memory.data_types import (
    ContentType as DeclarativeMemoryContentType,
)
from memmachine_server.episodic_memory.declarative_memory.data_types import (
    Episode as DeclarativeMemoryEpisode,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    NullContext,
    ProducerContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStore,
    SegmentStorePartition,
)
from memmachine_server.episodic_memory.event_memory.segmenter import Segmenter

# Stable namespace for deterministic Episode.uid -> Event.uuid mapping. Do not
# change without a data migration.
_EVENT_UUID_NAMESPACE = UUID("8c2c0e0a-3a2f-4b9c-9d1f-9b6c2a3a4f7e")

# Reserved system-defined property keys on the event-backend. Stored on
# event.properties with the leading underscore so EventMemory's existing
# `_to_vector_record_property` translation (bare client-API field -> `_field`)
# matches the storage layout transparently.
_EPISODE_UID_FIELD = "_episode_uid"
_SESSION_KEY_FIELD = "_session_key"
_PRODUCER_ID_FIELD = "_producer_id"
_PRODUCER_ROLE_FIELD = "_producer_role"
_PRODUCED_FOR_ID_FIELD = "_produced_for_id"
_SEQUENCE_NUM_FIELD = "_sequence_num"
_EPISODE_TYPE_FIELD = "_episode_type"
_CONTENT_TYPE_FIELD = "_content_type"
_CREATED_AT_FIELD = "_created_at"

EVENT_BACKEND_SYSTEM_FIELDS: dict[str, type[PropertyValue]] = {
    _EPISODE_UID_FIELD: str,
    _SESSION_KEY_FIELD: str,
    _PRODUCER_ID_FIELD: str,
    _PRODUCER_ROLE_FIELD: str,
    _PRODUCED_FOR_ID_FIELD: str,
    _SEQUENCE_NUM_FIELD: int,
    _EPISODE_TYPE_FIELD: str,
    _CONTENT_TYPE_FIELD: str,
    _CREATED_AT_FIELD: datetime.datetime,
}

# Filterable-metadata sentinel: Episode.filterable_metadata=None vs {} carry
# different semantics in the declarative backend; preserve that here too.
_FILTERABLE_METADATA_NONE_FLAG = "_filterable_metadata_none"

# Multiplier applied to `num_episodes_limit` when over-fetching from EventMemory
# so that dedup-by-`_episode_uid` has enough headroom to return that many
# distinct episodes even when a single episode produces multiple segments
# (e.g., under TextSegmenter with chunking).
_EVENT_BACKEND_DEDUP_OVERFETCH = 4


class DeclarativeBackendParams(BaseModel):
    """Parameters for the declarative-backed LongTermMemory."""

    backend: Literal["declarative"] = "declarative"
    session_id: str = Field(..., description="Session identifier")
    vector_graph_store: InstanceOf[VectorGraphStore] = Field(...)
    embedder: InstanceOf[Embedder] = Field(...)
    reranker: InstanceOf[Reranker] = Field(...)
    message_sentence_chunking: bool = Field(False)


class EventBackendParams(BaseModel):
    """Parameters for the event-backed LongTermMemory."""

    backend: Literal["event"] = "event"
    session_id: str = Field(..., description="Session identifier")
    vector_store: InstanceOf[VectorStore] = Field(
        ...,
        description="Parent VectorStore (for partition lifecycle)",
    )
    vector_store_collection: InstanceOf[VectorStoreCollection] = Field(
        ...,
        description="Already-opened VectorStore collection",
    )
    vector_store_collection_namespace: str = Field(...)
    segment_store: InstanceOf[SegmentStore] = Field(
        ...,
        description="Parent SegmentStore (for partition lifecycle)",
    )
    segment_store_partition: InstanceOf[SegmentStorePartition] = Field(
        ...,
        description="Already-opened SegmentStorePartition",
    )
    partition_key: str = Field(...)
    episode_storage: InstanceOf[EpisodeStorage] = Field(
        ...,
        description="EpisodeStorage used to hydrate Episodes at query time",
    )
    embedder: InstanceOf[Embedder] = Field(...)
    reranker: InstanceOf[Reranker] | None = Field(default=None)
    segmenter: InstanceOf[Segmenter] = Field(...)
    deriver: InstanceOf[Deriver] = Field(...)


LongTermMemoryParams = Annotated[
    DeclarativeBackendParams | EventBackendParams,
    Field(discriminator="backend"),
]


class LongTermMemory:
    """Long-term memory facade dispatching to a declarative or event backend."""

    def __init__(
        self,
        params: DeclarativeBackendParams | EventBackendParams,
    ) -> None:
        """Wire up the chosen backend."""
        self._backend: Literal["declarative", "event"] = params.backend

        # Backend-specific state. Only the relevant slots are populated.
        self._declarative_memory: DeclarativeMemory | None = None
        self._event_memory: EventMemory | None = None
        self._vector_store: VectorStore | None = None
        self._vector_store_namespace: str | None = None
        self._segment_store: SegmentStore | None = None
        self._partition_key: str | None = None
        self._episode_storage: EpisodeStorage | None = None
        self._session_id: str = params.session_id

        match params:
            case DeclarativeBackendParams():
                self._declarative_memory = DeclarativeMemory(
                    DeclarativeMemoryParams(
                        session_id=params.session_id,
                        vector_graph_store=params.vector_graph_store,
                        embedder=params.embedder,
                        reranker=params.reranker,
                        message_sentence_chunking=params.message_sentence_chunking,
                    ),
                )
            case EventBackendParams():
                self._event_memory = EventMemory(
                    EventMemoryParams(
                        segment_store_partition=params.segment_store_partition,
                        vector_store_collection=params.vector_store_collection,
                        segmenter=params.segmenter,
                        deriver=params.deriver,
                        embedder=params.embedder,
                        reranker=params.reranker,
                    ),
                )
                self._vector_store = params.vector_store
                self._vector_store_namespace = params.vector_store_collection_namespace
                self._segment_store = params.segment_store
                self._partition_key = params.partition_key
                self._episode_storage = params.episode_storage

    async def add_episodes(self, episodes: Iterable[Episode]) -> None:
        episodes = list(episodes)
        if self._backend == "declarative":
            assert self._declarative_memory is not None
            await self._declarative_memory.add_episodes(
                LongTermMemory._declarative_memory_episode(e) for e in episodes
            )
            return

        assert self._event_memory is not None
        events = [LongTermMemory._episode_to_event(episode) for episode in episodes]
        await self._event_memory.encode_events(events)

    async def search_scored(
        self,
        query: str,
        *,
        num_episodes_limit: int,
        expand_context: int = 0,
        score_threshold: float = -float("inf"),
        property_filter: FilterExpr | None = None,
    ) -> list[tuple[float, Episode]]:
        if self._backend == "declarative":
            return await self._search_scored_declarative(
                query,
                num_episodes_limit=num_episodes_limit,
                expand_context=expand_context,
                score_threshold=score_threshold,
                property_filter=property_filter,
            )
        return await self._search_scored_event(
            query,
            num_episodes_limit=num_episodes_limit,
            expand_context=expand_context,
            score_threshold=score_threshold,
            property_filter=property_filter,
        )

    async def _search_scored_declarative(
        self,
        query: str,
        *,
        num_episodes_limit: int,
        expand_context: int,
        score_threshold: float,
        property_filter: FilterExpr | None,
    ) -> list[tuple[float, Episode]]:
        assert self._declarative_memory is not None
        scored = await self._declarative_memory.search_scored(
            query,
            max_num_episodes=num_episodes_limit,
            expand_context=expand_context,
            property_filter=LongTermMemory._sanitize_declarative_filter(
                property_filter
            ),
        )
        return [
            (
                score,
                LongTermMemory._episode_from_declarative_memory_episode(dm_episode),
            )
            for score, dm_episode in scored
            if score >= score_threshold
        ]

    async def _search_scored_event(
        self,
        query: str,
        *,
        num_episodes_limit: int,
        expand_context: int,
        score_threshold: float,
        property_filter: FilterExpr | None,
    ) -> list[tuple[float, Episode]]:
        assert self._event_memory is not None
        assert self._episode_storage is not None
        # Over-fetch from EventMemory: the per-segment results can have many
        # segments per episode under non-passthrough segmenters, and we dedup
        # them by `_episode_uid` below. Without headroom, the dedup loop can
        # return fewer than `num_episodes_limit` distinct episodes.
        vector_search_limit = max(
            num_episodes_limit * _EVENT_BACKEND_DEDUP_OVERFETCH,
            num_episodes_limit,
        )
        result = await self._event_memory.query(
            query,
            vector_search_limit=vector_search_limit,
            expand_context=expand_context,
            property_filter=property_filter,
        )

        # Map seed segment -> _episode_uid (system field already lives on
        # event/segment.properties under the underscore-prefixed key). Keep
        # first-seen score per episode_uid; preserve query result ordering.
        ordered_uids: list[str] = []
        scores_by_uid: dict[str, float] = {}
        for scored_context in result.scored_segment_contexts:
            if scored_context.score < score_threshold:
                continue
            episode_uid = LongTermMemory._scored_context_episode_uid(scored_context)
            if episode_uid is None or episode_uid in scores_by_uid:
                continue
            scores_by_uid[episode_uid] = scored_context.score
            ordered_uids.append(episode_uid)
            if len(ordered_uids) >= num_episodes_limit:
                break

        if not ordered_uids:
            return []

        episodes_by_uid: dict[str, Episode] = {}
        for uid in ordered_uids:
            episode = await self._episode_storage.get_episode(uid)
            if episode is not None:
                episodes_by_uid[uid] = episode

        return [
            (scores_by_uid[uid], episodes_by_uid[uid])
            for uid in ordered_uids
            if uid in episodes_by_uid
        ]

    async def delete_episodes(self, uids: Iterable[str]) -> None:
        uids = list(uids)
        if self._backend == "declarative":
            assert self._declarative_memory is not None
            await self._declarative_memory.delete_episodes(uids)
            return

        assert self._event_memory is not None
        event_uuids = {uuid5(_EVENT_UUID_NAMESPACE, uid) for uid in uids}
        await self._event_memory.forget_events(event_uuids)

    async def drop_session_partition(self) -> None:
        """Delete all data for this session/partition."""
        if self._backend == "declarative":
            assert self._declarative_memory is not None
            episodes = await self._declarative_memory.get_matching_episodes()
            await self._declarative_memory.delete_episodes(
                episode.uid for episode in episodes
            )
            return

        assert self._vector_store is not None
        assert self._vector_store_namespace is not None
        assert self._segment_store is not None
        assert self._partition_key is not None
        await self._vector_store.delete_collection(
            namespace=self._vector_store_namespace,
            name=self._partition_key,
        )
        await self._segment_store.delete_partition(self._partition_key)

    async def close(self) -> None:
        # Backends do not own resources we can close at this layer; the
        # ResourceManager handles SegmentStore/VectorStore lifecycle.
        return

    # --- Episode <-> declarative-memory translation (declarative backend) ---

    @staticmethod
    def _declarative_memory_episode(episode: Episode) -> DeclarativeMemoryEpisode:
        """Convert a top-level Episode into a DeclarativeMemoryEpisode."""
        filterable_properties: dict[str, PropertyValue] = {
            key: value
            for key, value in {
                "created_at": episode.created_at,
                "session_key": episode.session_key,
                "producer_id": episode.producer_id,
                "producer_role": episode.producer_role,
                "produced_for_id": episode.produced_for_id,
                "sequence_num": episode.sequence_num,
                "episode_type": episode.episode_type.value,
                "content_type": episode.content_type.value,
            }.items()
            if value is not None
        }
        if episode.filterable_metadata is not None:
            for key, value in episode.filterable_metadata.items():
                filterable_properties[
                    LongTermMemory._mangle_filterable_metadata_key(key)
                ] = value
        else:
            filterable_properties[_FILTERABLE_METADATA_NONE_FLAG] = True

        return DeclarativeMemoryEpisode(
            uid=episode.uid or str(uuid4()),
            timestamp=episode.created_at,
            source=episode.producer_id,
            content_type=LongTermMemory._declarative_memory_content_type_from_episode(
                episode,
            ),
            content=episode.content,
            filterable_properties=filterable_properties,
            user_metadata=episode.metadata,
        )

    @staticmethod
    def _declarative_memory_content_type_from_episode(
        episode: Episode,
    ) -> DeclarativeMemoryContentType:
        match episode.episode_type:
            case EpisodeType.MESSAGE:
                match episode.content_type:
                    case ContentType.STRING:
                        return DeclarativeMemoryContentType.MESSAGE
                    case _:
                        return DeclarativeMemoryContentType.TEXT
            case _:
                return DeclarativeMemoryContentType.TEXT

    @staticmethod
    def _episode_from_declarative_memory_episode(
        dm: DeclarativeMemoryEpisode,
    ) -> Episode:
        return Episode(
            uid=dm.uid,
            sequence_num=cast("int", dm.filterable_properties.get("sequence_num", 0)),
            session_key=cast("str", dm.filterable_properties.get("session_key", "")),
            episode_type=EpisodeType(
                cast("str", dm.filterable_properties.get("episode_type", "")),
            ),
            content_type=ContentType(
                cast("str", dm.filterable_properties.get("content_type", "")),
            ),
            content=dm.content,
            created_at=dm.timestamp,
            producer_id=cast("str", dm.filterable_properties.get("producer_id", "")),
            producer_role=cast(
                "str", dm.filterable_properties.get("producer_role", "")
            ),
            produced_for_id=cast(
                "str | None", dm.filterable_properties.get("produced_for_id")
            ),
            filterable_metadata={
                LongTermMemory._demangle_filterable_metadata_key(key): value
                for key, value in dm.filterable_properties.items()
                if LongTermMemory._is_mangled_filterable_metadata_key(key)
            }
            if _FILTERABLE_METADATA_NONE_FLAG not in dm.filterable_properties
            else None,
            metadata=cast("dict[str, JsonValue] | None", dm.user_metadata),
        )

    _MANGLE_FILTERABLE_METADATA_KEY_PREFIX = "metadata."

    @staticmethod
    def _mangle_filterable_metadata_key(key: str) -> str:
        return LongTermMemory._MANGLE_FILTERABLE_METADATA_KEY_PREFIX + key

    @staticmethod
    def _demangle_filterable_metadata_key(mangled_key: str) -> str:
        return mangled_key.removeprefix(
            LongTermMemory._MANGLE_FILTERABLE_METADATA_KEY_PREFIX
        )

    @staticmethod
    def _is_mangled_filterable_metadata_key(candidate_key: str) -> bool:
        return candidate_key.startswith(
            LongTermMemory._MANGLE_FILTERABLE_METADATA_KEY_PREFIX
        )

    @staticmethod
    def _sanitize_declarative_filter(
        property_filter: FilterExpr | None,
    ) -> FilterExpr | None:
        if property_filter is None:
            return None
        return map_filter_fields(
            property_filter,
            lambda field: normalize_filter_field(field)[0],
        )

    # --- Episode <-> Event translation (event backend) ---

    @staticmethod
    def _scored_context_episode_uid(scored_context: object) -> str | None:
        """Pull `_episode_uid` from the seed segment of a ScoredSegmentContext."""
        # We don't import ScoredSegmentContext here just for typing; the runtime
        # shape (`segments`, `seed_segment_uuid`) is what matters.
        segments = getattr(scored_context, "segments", [])
        seed_uuid = getattr(scored_context, "seed_segment_uuid", None)
        seed = next((s for s in segments if s.uuid == seed_uuid), None)
        if seed is None:
            return None
        return cast(str | None, seed.properties.get(_EPISODE_UID_FIELD))

    @staticmethod
    def _episode_to_event(episode: Episode) -> Event:
        """Translate an Episode into an event-memory Event.

        - Event.uuid = uuid5(NAMESPACE, episode.uid) so the mapping is
          deterministic and reversible (`_episode_uid` carries the original).
        - Context: ProducerContext for messages; NullContext otherwise.
        - One TextBlock per event (Episode.content is a string today).
        - Properties: system fields stored with `_` prefix, user filterable
          metadata stored bare. Matches EventMemory's `_to_vector_record_property`
          translation so the client-facing filter API (`producer_id`,
          `m.my_field`) Just Works.
        """
        properties: dict[str, PropertyValue] = {
            _EPISODE_UID_FIELD: episode.uid,
            _SESSION_KEY_FIELD: episode.session_key,
            _PRODUCER_ID_FIELD: episode.producer_id,
            _PRODUCER_ROLE_FIELD: episode.producer_role,
            _SEQUENCE_NUM_FIELD: episode.sequence_num,
            _EPISODE_TYPE_FIELD: episode.episode_type.value,
            _CONTENT_TYPE_FIELD: episode.content_type.value,
            _CREATED_AT_FIELD: episode.created_at,
        }
        if episode.produced_for_id is not None:
            properties[_PRODUCED_FOR_ID_FIELD] = episode.produced_for_id
        if episode.filterable_metadata is not None:
            properties.update(episode.filterable_metadata)

        if episode.episode_type == EpisodeType.MESSAGE:
            context = ProducerContext(producer=episode.producer_id)
        else:
            context = NullContext()

        return Event(
            uuid=uuid5(_EVENT_UUID_NAMESPACE, episode.uid),
            timestamp=episode.created_at,
            context=context,
            blocks=[TextBlock(text=episode.content)],
            properties=properties,
        )
