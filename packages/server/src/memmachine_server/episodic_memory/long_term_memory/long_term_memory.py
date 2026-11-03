"""Long-term memory facade with declarative + event backends."""

import datetime
import logging
from collections.abc import Iterable
from typing import Final, cast
from uuid import UUID, uuid5

from pydantic import JsonValue

from memmachine_server.common.data_types import PropertyType, PropertyValue
from memmachine_server.common.episode_store import (
    ContentType,
    Episode,
    EpisodeStorage,
    EpisodeType,
)
from memmachine_server.common.filter import (
    FilterExpr,
    Property,
    SystemField,
    map_filter_fields,
)
from memmachine_server.common.property_keys import (
    is_reserved_property_key,
    reserved_property_key,
    validate_user_property_key,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.episodic_memory.declarative_memory import DeclarativeMemory
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
    QueryResult,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemoryPartition,
)

logger = logging.getLogger(__name__)

# Stable namespace for deterministic Episode.uid -> Event.uuid mapping. Do not
# change without a data migration.
_EVENT_UUID_NAMESPACE = UUID("8c2c0e0a-3a2f-4b9c-9d1f-9b6c2a3a4f7e")

# The adapter's fields, stored on `event.properties` under the keys it
# reserves, by the bare name a client filters each by; `ltm` names this
# memory, short because the keys share the identifier budget.
_SYSTEM_FIELD_KEYS: Final[dict[str, str]] = {
    name: reserved_property_key("ltm", name)
    for name in (
        "episode_uid",
        "session_key",
        "producer_id",
        "producer_role",
        "produced_for_id",
        "episode_type",
        "content_type",
        "created_at",
    )
}
_EPISODE_UID_FIELD = _SYSTEM_FIELD_KEYS["episode_uid"]
_SESSION_KEY_FIELD = _SYSTEM_FIELD_KEYS["session_key"]
_PRODUCER_ID_FIELD = _SYSTEM_FIELD_KEYS["producer_id"]
_PRODUCER_ROLE_FIELD = _SYSTEM_FIELD_KEYS["producer_role"]
_PRODUCED_FOR_ID_FIELD = _SYSTEM_FIELD_KEYS["produced_for_id"]
_EPISODE_TYPE_FIELD = _SYSTEM_FIELD_KEYS["episode_type"]
_CONTENT_TYPE_FIELD = _SYSTEM_FIELD_KEYS["content_type"]
_CREATED_AT_FIELD = _SYSTEM_FIELD_KEYS["created_at"]

EVENT_BACKEND_SYSTEM_FIELDS: dict[str, type[PropertyValue]] = {
    _EPISODE_UID_FIELD: str,
    _SESSION_KEY_FIELD: str,
    _PRODUCER_ID_FIELD: str,
    _PRODUCER_ROLE_FIELD: str,
    _PRODUCED_FOR_ID_FIELD: str,
    _EPISODE_TYPE_FIELD: str,
    _CONTENT_TYPE_FIELD: str,
    _CREATED_AT_FIELD: datetime.datetime,
}

# Bare client-API filter names for the system fields above (e.g. `producer_id`,
# not `memmachine_ltm_producer_id`). Used to validate filter expressions on the
# event backend so a typo'd field name surfaces as a ValueError rather than
# silently matching nothing in the storage layer.
_EVENT_BACKEND_SYSTEM_FIELD_CLIENT_NAMES: frozenset[str] = frozenset(
    _SYSTEM_FIELD_KEYS
) | {"timestamp"}

# The type of each system field a filter may name on the event backend.
EVENT_BACKEND_FILTER_FIELD_TYPES: dict[str, PropertyType] = {
    name: EVENT_BACKEND_SYSTEM_FIELDS[key] for name, key in _SYSTEM_FIELD_KEYS.items()
} | {"timestamp": datetime.datetime}

# Filterable-metadata sentinel: Episode.filterable_metadata=None vs {} carry
# different semantics in the declarative backend; preserve that here too.
_FILTERABLE_METADATA_NONE_FLAG = "_filterable_metadata_none"

# Multiplier applied to `num_episodes_limit` when over-fetching from EventMemory
# so that dedup by episode uid has enough headroom to return that many
# distinct episodes even when a single episode produces multiple segments
# (e.g., under TextSegmenter with chunking).
_EVENT_BACKEND_DEDUP_OVERFETCH = 4


class LongTermMemory:
    """Long-term memory routing each request to a session's declarative memory or event memory partition."""

    def __init__(
        self,
        memory: DeclarativeMemory | EventMemoryPartition,
        *,
        episode_storage: EpisodeStorage | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        """
        Route requests to a session's memory.

        An EventMemoryPartition needs `episode_storage`, which turns its
        results into episodes; a `reranker` makes its scores higher-is-better.

        Raises:
            ValueError: If `memory` is an EventMemoryPartition and
                `episode_storage` is None.
        """
        self._memory = memory
        self._episode_storage = episode_storage
        # Event backend only: whether scores from `EventMemoryPartition.query`
        # are higher-is-better. Matches the same derivation inside it
        # (reranker scores are higher-is-better; raw vector scores depend on the
        # vector store's similarity metric — cosine is higher-is-better, euclidean
        # is lower-is-better). Used to apply `score_threshold` in the correct
        # direction so it doesn't invert under euclidean with no reranker.
        self._score_higher_is_better: bool = True

        if isinstance(memory, EventMemoryPartition):
            if episode_storage is None:
                raise ValueError(
                    "An event memory partition needs the episode storage "
                    "that turns its results into episodes"
                )
            self._score_higher_is_better = (
                reranker is not None or memory.similarity_metric.higher_is_better
            )

    async def add_episodes(self, episodes: Iterable[Episode]) -> None:
        episodes = list(episodes)
        match self._memory:
            case DeclarativeMemory() as declarative_memory:
                await declarative_memory.add_episodes(
                    LongTermMemory._declarative_memory_episode(e) for e in episodes
                )
            case EventMemoryPartition() as event_memory_partition:
                await event_memory_partition.encode_events(
                    [LongTermMemory._episode_to_event(episode) for episode in episodes]
                )

    async def query(
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

        A `num_episodes_limit` that is not positive asks for nothing, and is
        refused with ValueError before the query is embedded.
        """
        if not num_episodes_limit > 0:
            raise ValueError(
                f"num_episodes_limit is not positive: {num_episodes_limit}"
            )
        match self._memory:
            case DeclarativeMemory() as declarative_memory:
                return await self._query_declarative(
                    declarative_memory,
                    query,
                    num_episodes_limit=num_episodes_limit,
                    expand_context=expand_context,
                    score_threshold=score_threshold,
                    property_filter=property_filter,
                )
            case EventMemoryPartition() as event_memory_partition:
                return await self._query_event(
                    event_memory_partition,
                    query,
                    num_episodes_limit=num_episodes_limit,
                    expand_context=expand_context,
                    score_threshold=score_threshold,
                    property_filter=property_filter,
                )

    async def _query_declarative(
        self,
        declarative_memory: DeclarativeMemory,
        query: str,
        *,
        num_episodes_limit: int,
        expand_context: int,
        score_threshold: float | None,
        property_filter: FilterExpr | None,
    ) -> list[tuple[float, Episode]]:
        scored = await declarative_memory.search_scored(
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

    async def _query_event(
        self,
        event_memory_partition: EventMemoryPartition,
        query: str,
        *,
        num_episodes_limit: int,
        expand_context: int,
        score_threshold: float | None,
        property_filter: FilterExpr | None,
    ) -> list[tuple[float, Episode]]:
        assert self._episode_storage is not None
        event_memory_filter = LongTermMemory._event_memory_filter(property_filter)
        # `expand_context` is a window of segments, the unit EventMemory and
        # the segment store work in: neither knows episodes. Under the
        # passthrough segmenter one segment is one episode, and the window is
        # the declarative backend's window of neighbor episodes; under a
        # splitting segmenter the same window covers fewer episodes, the ones
        # its segments belong to. The window can never exceed the remaining
        # quota in either unit (a segment belongs to one episode; declarative
        # parity), and can never go negative: a negative `expand_context`
        # would ask the segment store for a negative window, which the
        # SegmentStorePartition contract does not define.
        expand_context = max(0, min(expand_context, num_episodes_limit - 1))
        # Over-fetch from EventMemory: the per-segment results can have many
        # segments per episode under non-passthrough segmenters, and we dedup
        # them by episode uid below. Without headroom, the dedup loop can
        # return fewer than `num_episodes_limit` distinct episodes.
        vector_search_limit = max(
            num_episodes_limit * _EVENT_BACKEND_DEDUP_OVERFETCH,
            num_episodes_limit,
        )
        result = await event_memory_partition.query(
            query,
            vector_search_limit=vector_search_limit,
            expand_context=expand_context,
            property_filter=event_memory_filter,
        )

        if expand_context > 0:
            # The expanded windows carry timeline-neighbor segments; fold
            # their episodes into the result the same way the declarative
            # backend folds neighbor episodes around its matches: contexts
            # of the best matches first, filled until the limit is met.
            return await self._unified_scored_event_episodes(
                result,
                num_episodes_limit=num_episodes_limit,
                score_threshold=score_threshold,
            )

        # Map seed segment -> episode uid (system field already lives on
        # event/segment.properties under its reserved key). Keep
        # first-seen score per episode_uid; preserve query result ordering.
        # The threshold comparison direction depends on the scoring metric:
        # higher-is-better (cosine + any reranker) → drop scores BELOW threshold;
        # lower-is-better (raw euclidean without a reranker) → drop scores
        # ABOVE threshold.
        ordered_uids: list[UUID] = []
        scores_by_uid: dict[UUID, float] = {}
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
        episodes_by_uid: dict[UUID, Episode] = {ep.uid: ep for ep in episodes}

        missing = [uid for uid in ordered_uids if uid not in episodes_by_uid]
        if missing:
            # Index/storage drift: the event index referenced these episode
            # UIDs, but they're absent from EpisodeStorage. The two stores
            # are not transactionally linked, so this can happen on partial
            # failures during add/delete. Surface it so operators notice;
            # continue with whatever did hydrate.
            logger.warning(
                "query dropped %d episode(s) found in the event index "
                "but missing from EpisodeStorage (likely index/storage drift): %s",
                len(missing),
                missing,
            )

        return [
            (scores_by_uid[uid], episodes_by_uid[uid])
            for uid in ordered_uids
            if uid in episodes_by_uid
        ]

    async def delete_episodes(self, uids: Iterable[UUID]) -> None:
        uids = list(uids)
        match self._memory:
            case DeclarativeMemory() as declarative_memory:
                await declarative_memory.delete_episodes(uids)
            case EventMemoryPartition() as event_memory_partition:
                await event_memory_partition.forget_events(
                    {uuid5(_EVENT_UUID_NAMESPACE, str(uid)) for uid in uids}
                )

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
            uid=episode.uid,
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
            property_filter, LongTermMemory._declarative_memory_field
        )

    @staticmethod
    def _declarative_memory_field(field: SystemField | Property) -> Property:
        """The filterable property that stores a filter's field on the declarative backend."""
        match field:
            case SystemField(name=name):
                return Property(name)
            case Property(key=key):
                return Property(LongTermMemory._mangle_filterable_metadata_key(key))

    # --- Episode <-> Event translation (event backend) ---

    @staticmethod
    def _event_memory_filter(
        property_filter: FilterExpr | None,
    ) -> FilterExpr | None:
        """The filter EventMemory takes in the place of a caller's.

        Each of the adapter's system fields becomes the property that stores
        it under its reserved key, with its declared type; any other field
        is the same field in both, and a caller's property in the reserved
        namespace raises ValueError.
        """
        if property_filter is None:
            return None
        return map_filter_fields(property_filter, LongTermMemory._event_memory_field)

    @staticmethod
    def _event_memory_field(field: SystemField | Property) -> SystemField | Property:
        match field:
            case SystemField(name="timestamp"):
                return field
            case SystemField(name=name) if name in _SYSTEM_FIELD_KEYS:
                key = _SYSTEM_FIELD_KEYS[name]
                return Property(key, EVENT_BACKEND_SYSTEM_FIELDS[key])
            case SystemField(name=name):
                raise ValueError(
                    f"Unknown filter field {name!r}. Valid system fields: "
                    f"{sorted(_EVENT_BACKEND_SYSTEM_FIELD_CLIENT_NAMES)}; "
                    "use 'm.<name>' for user metadata."
                )
            case Property(key=key) if is_reserved_property_key(key):
                raise ValueError(
                    f"Filter field {key!r} names a reserved key; filter "
                    f"a system field by its bare name: "
                    f"{sorted(_EVENT_BACKEND_SYSTEM_FIELD_CLIENT_NAMES)}."
                )
            case Property():
                return field

    async def _unified_scored_event_episodes(
        self,
        result: QueryResult,
        *,
        num_episodes_limit: int,
        score_threshold: float | None,
    ) -> list[tuple[float, Episode]]:
        """Fold expanded segment windows into a unified episode context.

        Event-backend analog of DeclarativeMemory's context unification:
        each scored window contributes the episodes its segments belong to
        (chronological within the window, the seed's episode as nucleus),
        best-scoring windows first, whole while they fit and by proximity to
        the nucleus when they no longer do. The unified context is returned
        chronologically, as the declarative backend returns its own.
        """
        assert self._episode_storage is not None
        scored_uid_contexts: list[tuple[float, UUID, list[UUID]]] = []
        for scored_context in result.scored_segment_contexts:
            if not self._score_passes_threshold(scored_context.score, score_threshold):
                continue
            nuclear_uid, context_uids = LongTermMemory._episode_uid_context(
                scored_context
            )
            if nuclear_uid is None:
                continue
            scored_uid_contexts.append(
                (scored_context.score, nuclear_uid, context_uids)
            )

        episode_scores = LongTermMemory._unify_scored_uid_contexts(
            scored_uid_contexts,
            max_num_episodes=num_episodes_limit,
        )
        if not episode_scores:
            return []

        episodes = await self._episode_storage.get_episodes(list(episode_scores))
        episodes_by_uid: dict[UUID, Episode] = {ep.uid: ep for ep in episodes}
        missing = [uid for uid in episode_scores if uid not in episodes_by_uid]
        if missing:
            logger.warning(
                "query dropped %d episode(s) found in the event index "
                "but missing from EpisodeStorage (likely index/storage drift): %s",
                len(missing),
                missing,
            )
        return sorted(
            (
                (episode_scores[uid], episodes_by_uid[uid])
                for uid in episode_scores
                if uid in episodes_by_uid
            ),
            key=lambda scored_episode: (
                scored_episode[1].created_at,
                scored_episode[1].uid,
            ),
        )

    @staticmethod
    def _episode_uid_context(
        scored_context: object,
    ) -> tuple[UUID | None, list[UUID]]:
        """Episode uids covered by one segment window.

        Returns the seed segment's episode uid (the nucleus) and the deduped
        episode uids of every segment in the window, in the window's
        chronological order.
        """
        segments = getattr(scored_context, "segments", [])
        seed_uuid = getattr(scored_context, "seed_segment_uuid", None)
        nuclear_uid: UUID | None = None
        context_uids: list[UUID] = []
        seen: set[UUID] = set()
        for segment in segments:
            episode_uid = segment.properties.get(_EPISODE_UID_FIELD)
            if episode_uid is None:
                continue
            episode_uid = UUID(str(episode_uid))
            if episode_uid not in seen:
                seen.add(episode_uid)
                context_uids.append(episode_uid)
            if segment.uuid == seed_uuid:
                nuclear_uid = episode_uid
        return nuclear_uid, context_uids

    @staticmethod
    def _unify_scored_uid_contexts(
        scored_uid_contexts: Iterable[tuple[float, UUID, list[UUID]]],
        max_num_episodes: int,
    ) -> dict[UUID, float]:
        """Unify episode-uid contexts into a limited set, best windows first.

        Mirror of DeclarativeMemory._unify_scored_anchored_episode_contexts:
        a window is taken whole while it fits within the limit; a window
        that would overflow contributes episodes by weighted index-proximity
        to its nucleus (forward recall preferred over backward) until the
        limit is met. An episode keeps the score of the first window that
        contributed it.
        """
        episode_scores: dict[UUID, float] = {}
        for score, nuclear_uid, context in scored_uid_contexts:
            if len(episode_scores) >= max_num_episodes:
                break
            if (len(episode_scores) + len(context)) <= max_num_episodes:
                for episode_uid in context:
                    episode_scores.setdefault(episode_uid, score)
                continue
            nuclear_index = context.index(nuclear_uid)

            def weighted_index_proximity(
                index: int, anchor: int = nuclear_index
            ) -> float:
                proximity = index - anchor
                if proximity >= 0:
                    # Forward recall is better than backward recall.
                    return (proximity - 0.5) / 2
                return float(-proximity)

            for index in sorted(range(len(context)), key=weighted_index_proximity):
                if len(episode_scores) >= max_num_episodes:
                    break
                episode_scores.setdefault(context[index], score)
        return episode_scores

    @staticmethod
    def _scored_context_episode_uid(scored_context: object) -> UUID | None:
        """Pull the episode uid from the seed segment of a ScoredSegmentContext."""
        # We don't import ScoredSegmentContext here just for typing; the runtime
        # shape (`segments`, `seed_segment_uuid`) is what matters.
        segments = getattr(scored_context, "segments", [])
        seed_uuid = getattr(scored_context, "seed_segment_uuid", None)
        seed = next((s for s in segments if s.uuid == seed_uuid), None)
        if seed is None:
            return None
        value = seed.properties.get(_EPISODE_UID_FIELD)
        return UUID(str(value)) if value is not None else None

    @staticmethod
    def _episode_to_event(episode: Episode) -> Event:
        """Translate an Episode into an event-memory Event.

        - Event.uuid = uuid5(NAMESPACE, episode.uid) so the mapping is
          deterministic and reversible (the episode uid field carries the
          original).
        - Context: ProducerContext for messages; NullContext otherwise.
        - One TextBlock per event (Episode.content is a string today).
        - Properties: system fields under the adapter's reserved keys, user
          filterable metadata under its own keys, as `_event_memory_filter`
          resolves a filter's fields.

        A user key must be a legal user property key
        (`validate_user_property_key`): one in the reserved namespace would
        claim a system field, so it is refused.
        """
        properties: dict[str, PropertyValue] = {
            _EPISODE_UID_FIELD: str(episode.uid),
            _SESSION_KEY_FIELD: episode.session_key,
            _PRODUCER_ID_FIELD: episode.producer_id,
            _PRODUCER_ROLE_FIELD: episode.producer_role,
            _EPISODE_TYPE_FIELD: episode.episode_type.value,
            _CONTENT_TYPE_FIELD: episode.content_type.value,
            _CREATED_AT_FIELD: episode.created_at,
        }
        if episode.produced_for_id is not None:
            properties[_PRODUCED_FOR_ID_FIELD] = episode.produced_for_id
        if episode.filterable_metadata is not None:
            for key in episode.filterable_metadata:
                validate_user_property_key(key)
            properties.update(episode.filterable_metadata)

        if episode.episode_type == EpisodeType.MESSAGE:
            context = ProducerContext(producer=episode.producer_id)
        else:
            context = NullContext()

        return Event(
            uuid=uuid5(_EVENT_UUID_NAMESPACE, str(episode.uid)),
            timestamp=episode.created_at,
            context=context,
            blocks=[TextBlock(text=episode.content)],
            properties=properties,
        )
