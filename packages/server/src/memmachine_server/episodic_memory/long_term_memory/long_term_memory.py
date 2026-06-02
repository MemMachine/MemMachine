"""Long-term memory facade with declarative + event backends."""

import datetime
import logging
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
    demangle_user_metadata_key,
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

logger = logging.getLogger(__name__)

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

# Bare client-API filter names for the system fields above (e.g. `producer_id`,
# not `_producer_id`). Used to validate filter expressions on the event backend
# so a typo'd field name surfaces as a ValueError rather than silently matching
# nothing in the storage layer.
_EVENT_BACKEND_SYSTEM_FIELD_CLIENT_NAMES: frozenset[str] = frozenset(
    key.removeprefix("_") for key in EVENT_BACKEND_SYSTEM_FIELDS
) | {"timestamp"}

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
    user_property_keys: frozenset[str] = Field(
        default_factory=frozenset,
        description=(
            "Configured user-property names (from properties_schema). When "
            "non-empty, filter expressions on `m.<key>` are validated against "
            "this set; empty means no validation (any user-metadata key "
            "accepted)."
        ),
    )


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
        # Event backend only: whether scores from `EventMemory.query` are
        # higher-is-better. Matches the same derivation inside EventMemory.query
        # (reranker scores are higher-is-better; raw vector scores depend on the
        # collection's similarity metric — cosine is higher-is-better, euclidean
        # is lower-is-better). Used to apply `score_threshold` in the correct
        # direction so it doesn't invert under euclidean with no reranker.
        self._score_higher_is_better: bool = True
        # Event backend only: configured user-property names from
        # properties_schema. Empty means "no validation"; non-empty means the
        # set is closed and filter expressions referencing `m.<unknown>` raise
        # ValueError at the LongTermMemory layer.
        self._user_property_keys: frozenset[str] = frozenset()
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
                self._score_higher_is_better = (
                    params.reranker is not None
                    or params.vector_store_collection.config.similarity_metric.higher_is_better
                )
                self._user_property_keys = params.user_property_keys

    async def add_episodes(self, episodes: Iterable[Episode]) -> None:
        episodes = list(episodes)
        if self._backend == "declarative":
            assert self._declarative_memory is not None
            await self._declarative_memory.add_episodes(
                LongTermMemory._declarative_memory_episode(e) for e in episodes
            )
            return

        event_memory = self._require_event_backend_live()
        events = [LongTermMemory._episode_to_event(episode) for episode in episodes]
        await event_memory.encode_events(events)

    async def search_scored(
        self,
        query: str,
        *,
        num_episodes_limit: int,
        expand_context: int = 0,
        score_threshold: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[tuple[float, Episode]]:
        """Score-thresholded query.

        `score_threshold=None` (default) keeps every result. With a numeric
        value, the comparison direction matches the scoring metric:
        higher-is-better metrics (cosine, dot, any reranker) drop scores BELOW
        the threshold; lower-is-better metrics (raw euclidean / manhattan with
        no reranker) drop scores ABOVE it. Avoids the prior `-inf` sentinel,
        which silently inverted to "drop everything" under euclidean.
        """
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
        score_threshold: float | None,
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
            if score_threshold is None or score >= score_threshold
        ]

    async def _search_scored_event(
        self,
        query: str,
        *,
        num_episodes_limit: int,
        expand_context: int,
        score_threshold: float | None,
        property_filter: FilterExpr | None,
    ) -> list[tuple[float, Episode]]:
        event_memory = self._require_event_backend_live()
        assert self._episode_storage is not None
        self._validate_event_backend_filter(property_filter)
        # Over-fetch from EventMemory: the per-segment results can have many
        # segments per episode under non-passthrough segmenters, and we dedup
        # them by `_episode_uid` below. Without headroom, the dedup loop can
        # return fewer than `num_episodes_limit` distinct episodes.
        vector_search_limit = max(
            num_episodes_limit * _EVENT_BACKEND_DEDUP_OVERFETCH,
            num_episodes_limit,
        )
        result = await event_memory.query(
            query,
            vector_search_limit=vector_search_limit,
            expand_context=expand_context,
            property_filter=property_filter,
        )

        # Map seed segment -> _episode_uid (system field already lives on
        # event/segment.properties under the underscore-prefixed key). Keep
        # first-seen score per episode_uid; preserve query result ordering.
        # The threshold comparison direction depends on the scoring metric:
        # higher-is-better (cosine + any reranker) → drop scores BELOW threshold;
        # lower-is-better (raw euclidean without a reranker) → drop scores
        # ABOVE threshold.
        ordered_uids: list[str] = []
        scores_by_uid: dict[str, float] = {}
        for scored_context in result.scored_segment_contexts:
            if not self._score_passes_threshold(scored_context.score, score_threshold):
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

        episodes = await self._episode_storage.get_episodes(ordered_uids)
        episodes_by_uid: dict[str, Episode] = {ep.uid: ep for ep in episodes}

        missing = [uid for uid in ordered_uids if uid not in episodes_by_uid]
        if missing:
            # Index/storage drift: the event index referenced these episode
            # UIDs, but they're absent from EpisodeStorage. The two stores
            # are not transactionally linked, so this can happen on partial
            # failures during add/delete. Surface it so operators notice;
            # continue with whatever did hydrate.
            logger.warning(
                "search_scored dropped %d episode(s) found in the event index "
                "but missing from EpisodeStorage (likely index/storage drift): %s",
                len(missing),
                missing,
            )

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

        event_memory = self._require_event_backend_live()
        event_uuids = {uuid5(_EVENT_UUID_NAMESPACE, uid) for uid in uids}
        await event_memory.forget_events(event_uuids)

    async def drop_session_partition(self) -> None:
        """Delete all data for this session/partition.

        On the event backend, this drops the underlying VectorStore collection
        and SegmentStore partition. After this returns the instance is no
        longer usable — `EventMemory` still holds handles to the deleted
        collection and partition, and any reuse would talk to deleted
        resources. We null those handles so subsequent calls fail loudly
        rather than silently corrupt state. If the caller needs the same
        session_id again, build a fresh LongTermMemory (which will open or
        create a new collection/partition).
        """
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
        # Drop references to the now-deleted resources so any further
        # add_episodes / search_scored / delete_episodes calls raise
        # rather than silently operating on stale handles.
        self._event_memory = None
        self._vector_store = None
        self._segment_store = None

    async def close(self) -> None:
        # Backends do not own resources we can close at this layer; the
        # ResourceManager handles SegmentStore/VectorStore lifecycle.
        return

    def _score_passes_threshold(
        self, score: float, score_threshold: float | None
    ) -> bool:
        """Apply `score_threshold` in the correct direction for the metric.

        higher-is-better → drop scores BELOW threshold;
        lower-is-better  → drop scores ABOVE threshold;
        None             → never drop.
        """
        if score_threshold is None:
            return True
        if self._score_higher_is_better:
            return score >= score_threshold
        return score <= score_threshold

    def _require_event_backend_live(self) -> EventMemory:
        """Return the EventMemory or raise if the instance was dropped."""
        if self._event_memory is None:
            raise RuntimeError(
                "LongTermMemory event backend is no longer usable: "
                "drop_session_partition() deleted the underlying collection "
                "and partition. Construct a new LongTermMemory to operate "
                "on this session again."
            )
        return self._event_memory

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

    def _validate_event_backend_filter(
        self,
        property_filter: FilterExpr | None,
    ) -> None:
        """Reject filter fields not known to the event-backend schema.

        Bare names are matched against system-defined fields
        (`_EVENT_BACKEND_SYSTEM_FIELD_CLIENT_NAMES`); `m.<key>` / `metadata.<key>`
        names are matched against `user_property_keys`, if non-empty.

        Validation lives here rather than in the segment store / vector store
        because this is the only layer that knows both the system field set
        and the configured user `properties_schema`. The stores themselves
        treat unknown property keys as empty matches (correct generic JSON
        semantics) — without this check a typo'd filter field would silently
        return zero results.
        """
        if property_filter is None:
            return

        def _check(field: str) -> str:
            internal_name, is_user_metadata = normalize_filter_field(field)
            if is_user_metadata:
                key = demangle_user_metadata_key(internal_name)
                if self._user_property_keys and key not in self._user_property_keys:
                    raise ValueError(
                        f"Unknown user-metadata filter field {field!r}. "
                        "Configured user properties: "
                        f"{sorted(self._user_property_keys)}"
                    )
                return field
            if field not in _EVENT_BACKEND_SYSTEM_FIELD_CLIENT_NAMES:
                raise ValueError(
                    f"Unknown filter field {field!r}. Valid system fields: "
                    f"{sorted(_EVENT_BACKEND_SYSTEM_FIELD_CLIENT_NAMES)}; "
                    "use 'm.<name>' for user metadata."
                )
            return field

        # `map_filter_fields` walks the whole tree; we use it for its side
        # effect of invoking `_check` on every leaf field. The returned tree
        # is discarded.
        map_filter_fields(property_filter, _check)

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

        Reject `_`-prefixed user metadata keys (event-backend only — the
        declarative backend mangles user keys with a `metadata.` prefix and
        is unaffected). Without this check a client could send
        `{"_producer_id": "victim", "_session_key": "other-session"}` and
        have its content indexed under those spoofed identities, enabling
        cross-producer / cross-session impersonation through
        `search_scored(property_filter=...)`. We raise loudly instead of
        silently dropping so the client sees the misuse.
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
            reserved = sorted(
                k for k in episode.filterable_metadata if k.startswith("_")
            )
            if reserved:
                raise ValueError(
                    "Episode filterable_metadata contains reserved "
                    f"`_`-prefixed keys (event backend only): {reserved}. "
                    "These collide with system-defined properties "
                    "(`_producer_id`, `_session_key`, `_episode_uid`, ...) "
                    "and are rejected to prevent cross-producer / "
                    "cross-session impersonation."
                )
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
