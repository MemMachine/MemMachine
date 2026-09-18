"""Long-term memory facade with declarative + event backends."""

import datetime
import logging
from collections.abc import Iterable
from typing import Annotated, Final, Literal, NamedTuple, cast
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
    And,
    Comparison,
    FilterExpr,
    In,
    map_filter_fields,
    normalize_filter_field,
)
from memmachine_server.common.metrics_factory import MetricsFactory
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
    Author,
    Context,
    DateTimeFormat,
    Event,
    QueryHit,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.event_memory_store import (
    EventMemoryStore,
    EventMemoryStorePartition,
)
from memmachine_server.episodic_memory.event_memory.segmenter import Segmenter

logger = logging.getLogger(__name__)

# Stable namespace for deterministic Episode.uid -> Event.uuid mapping. Do not
# change without a data migration.
_EVENT_UUID_NAMESPACE = UUID("8c2c0e0a-3a2f-4b9c-9d1f-9b6c2a3a4f7e")

DEFAULT_SESSION_ID: Final[str] = "memmachine_default"
"""The session of every event this API ingests.

The API carries no conversation id, so a partition's events are one
stream. This one name is reserved for it; a caller may name a session
anything else.
"""

# The adapter's fields, stored on `event.properties` under a leading
# underscore; the event memory store maps a bare client-API name to `_<name>`.
_EPISODE_UID_FIELD = "_episode_uid"
_SESSION_KEY_FIELD = "_session_key"
_PRODUCER_ID_FIELD = "_producer_id"
_PRODUCER_ROLE_FIELD = "_producer_role"
_PRODUCED_FOR_ID_FIELD = "_produced_for_id"
_SEQUENCE_NUM_FIELD = "_sequence_num"
_EPISODE_TYPE_FIELD = "_episode_type"
_CONTENT_TYPE_FIELD = "_content_type"
_CREATED_AT_FIELD = "_created_at"

# The fields the adapter writes into every event's properties. The segment
# store holds them with the caller's properties, and `property_filter`
# selects on them there; the vector store never sees them.
_EVENT_BACKEND_SYSTEM_FIELDS: frozenset[str] = frozenset(
    {
        _EPISODE_UID_FIELD,
        _SESSION_KEY_FIELD,
        _PRODUCER_ID_FIELD,
        _PRODUCER_ROLE_FIELD,
        _PRODUCED_FOR_ID_FIELD,
        _SEQUENCE_NUM_FIELD,
        _EPISODE_TYPE_FIELD,
        _CONTENT_TYPE_FIELD,
        _CREATED_AT_FIELD,
    }
)

# Bare client-API filter names for the system fields above (e.g. `producer_id`,
# not `_producer_id`). Used to validate filter expressions on the event backend
# so a typo'd field name surfaces as a ValueError rather than silently matching
# nothing in the storage layer.
_EVENT_BACKEND_SYSTEM_FIELD_CLIENT_NAMES: frozenset[str] = frozenset(
    key.removeprefix("_") for key in _EVENT_BACKEND_SYSTEM_FIELDS
) | {"timestamp"}

# Filterable-metadata sentinel: Episode.filterable_metadata=None vs {} carry
# different semantics in the declarative backend; preserve that here too.
_FILTERABLE_METADATA_NONE_FLAG = "_filterable_metadata_none"

# Multiplier applied to `num_episodes_limit` when over-fetching from EventMemory
# so that dedup-by-`_episode_uid` has enough headroom to return that many
# distinct episodes even when a single episode produces multiple segments
# (e.g., under TextSegmenter with chunking).
_EVENT_BACKEND_DEDUP_OVERFETCH = 4


class _LiftedFilters(NamedTuple):
    """A filter tree split into the memory's typed parameters and its post-filter."""

    since: datetime.datetime | None
    until: datetime.datetime | None
    source_ids: list[str] | None
    rest: FilterExpr | None


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
    event_memory_store: InstanceOf[EventMemoryStore] = Field(
        ...,
        description="Parent EventMemoryStore (for partition lifecycle)",
    )
    event_memory_store_partition: InstanceOf[EventMemoryStorePartition] = Field(
        ...,
        description="Already-opened EventMemoryStorePartition",
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
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        default=None,
        description=(
            "Metrics factory handed to EventMemory's OperationTracker. Without "
            "it the tracker discards every timing it takes, silently."
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
        self._event_memory_store: EventMemoryStore | None = None
        self._partition_key: str | None = None
        self._episode_storage: EpisodeStorage | None = None
        self._session_id: str = params.session_id
        # Event backend only: reranking is a stage LongTermMemory runs on
        # top of EventMemory's vector search.
        self._reranker: Reranker | None = None

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
                        event_memory_store_partition=params.event_memory_store_partition,
                        vector_store_collection=params.vector_store_collection,
                        segmenter=params.segmenter,
                        deriver=params.deriver,
                        embedder=params.embedder,
                        metrics_factory=params.metrics_factory,
                    ),
                )
                self._reranker = params.reranker
                self._vector_store = params.vector_store
                self._vector_store_namespace = params.vector_store_collection_namespace
                self._event_memory_store = params.event_memory_store
                self._partition_key = params.partition_key
                self._episode_storage = params.episode_storage

    @property
    def event_memory(self) -> EventMemory | None:
        """The memory of the event backend, for a caller that speaks events.

        None on the declarative backend, and once
        `drop_session_partition` has deleted the collection and the
        partition this memory held.
        """
        return self._event_memory

    @property
    def reranker(self) -> Reranker | None:
        """The reranker this memory scores hits with, or None when it has none."""
        return self._reranker

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

        `score_threshold=None` (default) keeps every result. A numeric value
        drops scores below it: every score here is a cosine similarity or a
        reranker score, and both are higher-is-better, so the comparison needs
        no direction. Avoids the prior `-inf` sentinel, which silently
        inverted to "drop everything" under a lower-is-better metric.
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
        # Context can never exceed the remaining quota (declarative parity),
        # and can never go negative: with `num_episodes_limit == 0` the quota
        # clamp on its own would ask the event memory store for a window of -1,
        # which the EventMemoryStorePartition contract does not define.
        expand_context = max(0, min(expand_context, num_episodes_limit - 1))
        # Over-fetch from EventMemory: a segmenter can cut an episode into
        # many segments, so the per-segment results can carry one episode
        # several times, and we dedup them by `_episode_uid` below. Without
        # headroom, the dedup loop can return fewer than `num_episodes_limit`
        # distinct episodes.
        vector_search_limit = max(
            num_episodes_limit * _EVENT_BACKEND_DEDUP_OVERFETCH,
            num_episodes_limit,
        )
        # The fields ingestion maps onto the event come back typed: the
        # memory filters by them at the vector stage, and the rest of the
        # tree is the event memory store's post-filter.
        lifted = LongTermMemory._lift_typed_filters(property_filter)
        hits = await event_memory.query(
            query,
            vector_search_limit=vector_search_limit,
            expand_context=expand_context,
            since=lifted.since,
            until=lifted.until,
            source_ids=lifted.source_ids,
            property_filter=lifted.rest,
        )
        if self._reranker is not None:
            hits = await EventMemory.rerank(
                query,
                hits,
                reranker=self._reranker,
                datetime_format=DateTimeFormat(time_style="short"),
            )

        if expand_context > 0:
            # The expanded windows carry timeline-neighbor segments; fold
            # their episodes into the result the same way the declarative
            # backend folds neighbor episodes around its matches: contexts
            # of the best matches first, filled until the limit is met.
            return await self._unified_scored_event_episodes(
                hits,
                num_episodes_limit=num_episodes_limit,
                score_threshold=score_threshold,
            )

        # Map seed segment -> _episode_uid (system field already lives on
        # event/segment.properties under the underscore-prefixed key). Keep
        # first-seen score per episode_uid; preserve query result ordering.
        # Cosine similarities and reranker scores are both higher-is-better,
        # so the threshold always drops scores below it.
        ordered_uids: list[str] = []
        scores_by_uid: dict[str, float] = {}
        for hit in hits:
            if not self._score_passes_threshold(hit.score, score_threshold):
                continue
            episode_uid = LongTermMemory._hit_episode_uid(hit)
            if episode_uid is None or episode_uid in scores_by_uid:
                continue
            scores_by_uid[episode_uid] = hit.score
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
        and EventMemoryStore partition. After this returns the instance is no
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
        assert self._event_memory_store is not None
        assert self._partition_key is not None
        # The segment partition first: its deletion waits for every write
        # in flight, whose records land before it commits, and blocks new
        # ones, so the collection deletion that follows removes every
        # record that could ever have landed.
        await self._event_memory_store.delete_partition(self._partition_key)
        await self._vector_store.delete_collection(
            namespace=self._vector_store_namespace,
            name=self._partition_key,
        )
        # Drop references to the now-deleted resources so any further
        # add_episodes / search_scored / delete_episodes calls raise
        # rather than silently operating on stale handles.
        self._event_memory = None
        self._vector_store = None
        self._event_memory_store = None
        # Physical reclamation is the sweeper's, which the resource manager
        # runs: the partition is unreachable now, and its rows are reclaimed
        # within the sweeper's interval.

    async def close(self) -> None:
        # Backends do not own resources we can close at this layer; the
        # ResourceManager handles EventMemoryStore/VectorStore lifecycle.
        return

    def _score_passes_threshold(
        self, score: float, score_threshold: float | None
    ) -> bool:
        """Drop scores below `score_threshold`; None never drops.

        Reranker scores and cosine similarities are both higher-is-better.
        """
        if score_threshold is None:
            return True
        return score >= score_threshold

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

    @staticmethod
    def _lift_typed_filters(property_filter: FilterExpr | None) -> _LiftedFilters:
        """Lift the conjuncts on ingestion-mapped fields out of a filter tree.

        `_episode_to_event` maps `created_at` onto the event's timestamp and
        `producer_id` onto its source, so a top-level `timestamp >= x` or
        `created_at >= x` is `since`, `< x` is `until` (the tightest of
        each), and `producer_id = x` or `producer_id IN [...]` is
        `source_ids` (the intersection; empty admits nothing). The tree of
        the remaining conjuncts is `rest`, or None when nothing remains. A
        predicate of another operator, or one under a disjunction or a
        negation, stays in the tree as a post-filter.
        """
        since: datetime.datetime | None = None
        until: datetime.datetime | None = None
        source_ids: set[str] | None = None
        rest: list[FilterExpr] = []
        for conjunct in LongTermMemory._conjuncts(property_filter):
            match conjunct:
                case Comparison(
                    field="timestamp" | "created_at",
                    op=">=",
                    value=datetime.datetime() as bound,
                ):
                    since = bound if since is None else max(since, bound)
                case Comparison(
                    field="timestamp" | "created_at",
                    op="<",
                    value=datetime.datetime() as bound,
                ):
                    until = bound if until is None else min(until, bound)
                case Comparison(field="producer_id", op="=", value=str() as source):
                    sources = {source}
                    source_ids = sources if source_ids is None else source_ids & sources
                case In(field="producer_id", values=values) if all(
                    isinstance(value, str) for value in values
                ):
                    sources = set(cast(list[str], values))
                    source_ids = sources if source_ids is None else source_ids & sources
                case _:
                    rest.append(conjunct)
        remaining: FilterExpr | None = None
        for conjunct in rest:
            remaining = (
                conjunct if remaining is None else And(left=remaining, right=conjunct)
            )
        return _LiftedFilters(
            since=since,
            until=until,
            source_ids=None if source_ids is None else sorted(source_ids),
            rest=remaining,
        )

    @staticmethod
    def _conjuncts(expr: FilterExpr | None) -> list[FilterExpr]:
        """The top-level conjuncts of a tree, nested conjunctions flattened."""
        if expr is None:
            return []
        if isinstance(expr, And):
            return [
                *LongTermMemory._conjuncts(expr.left),
                *LongTermMemory._conjuncts(expr.right),
            ]
        return [expr]

    def _validate_event_backend_filter(
        self,
        property_filter: FilterExpr | None,
    ) -> None:
        """Reject bare filter fields that name no system field.

        A `m.<key>` / `metadata.<key>` name may be any caller key; a bare
        name is a system field or a mistake.
        """
        if property_filter is None:
            return

        def _check(field: str) -> str:
            _internal_name, is_user_metadata = normalize_filter_field(field)
            if is_user_metadata:
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

    async def _unified_scored_event_episodes(
        self,
        hits: Iterable[QueryHit],
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
        scored_uid_contexts: list[tuple[float, str, list[str]]] = []
        for hit in hits:
            if not self._score_passes_threshold(hit.score, score_threshold):
                continue
            nuclear_uid, context_uids = LongTermMemory._episode_uid_context(hit)
            if nuclear_uid is None:
                continue
            scored_uid_contexts.append((hit.score, nuclear_uid, context_uids))

        episode_scores = LongTermMemory._unify_scored_uid_contexts(
            scored_uid_contexts,
            max_num_episodes=num_episodes_limit,
        )
        if not episode_scores:
            return []

        episodes = await self._episode_storage.get_episodes(list(episode_scores))
        episodes_by_uid: dict[str, Episode] = {ep.uid: ep for ep in episodes}
        missing = [uid for uid in episode_scores if uid not in episodes_by_uid]
        if missing:
            logger.warning(
                "search_scored dropped %d episode(s) found in the event index "
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
    def _episode_uid_context(hit: QueryHit) -> tuple[str | None, list[str]]:
        """Episode uids covered by one segment window.

        Returns the seed segment's episode uid (the nucleus) and the deduped
        episode uids of every segment in the window, in the window's
        chronological order.
        """
        nuclear_uid: str | None = None
        context_uids: list[str] = []
        seen: set[str] = set()
        for segment in hit.window():
            episode_uid = segment.properties.get(_EPISODE_UID_FIELD)
            if episode_uid is None:
                continue
            episode_uid = str(episode_uid)
            if episode_uid not in seen:
                seen.add(episode_uid)
                context_uids.append(episode_uid)
            if segment.uuid == hit.seed.uuid:
                nuclear_uid = episode_uid
        return nuclear_uid, context_uids

    @staticmethod
    def _unify_scored_uid_contexts(
        scored_uid_contexts: Iterable[tuple[float, str, list[str]]],
        max_num_episodes: int,
    ) -> dict[str, float]:
        """Unify episode-uid contexts into a limited set, best windows first.

        Mirror of DeclarativeMemory._unify_scored_anchored_episode_contexts:
        a window is taken whole while it fits within the limit; a window
        that would overflow contributes episodes by weighted index-proximity
        to its nucleus (forward recall preferred over backward) until the
        limit is met. An episode keeps the score of the first window that
        contributed it.
        """
        episode_scores: dict[str, float] = {}
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
    def _hit_episode_uid(hit: QueryHit) -> str | None:
        """Pull `_episode_uid` from the seed segment of a hit."""
        return cast(str | None, hit.seed.properties.get(_EPISODE_UID_FIELD))

    @staticmethod
    def _episode_to_event(episode: Episode) -> Event:
        """Translate an Episode into an event-memory Event.

        - Event.uuid = uuid5(NAMESPACE, episode.uid) so the mapping is
          deterministic and reversible (`_episode_uid` carries the original).
        - Event.session_id = DEFAULT_SESSION_ID: the API carries no
          conversation id, so a partition's events are one stream, under
          the one reserved session name; when the API carries one, it
          goes here.
        - The producer id is both the event's `source_id`, the one source
          an episode has, and an `Author` part in its context, as the
          `ProducerContext` was before, so a rendered segment keeps its
          `producer: text` shape. The server API carries no readable name,
          so the id stands in.
        - One TextBlock per event (Episode.content is a string today).
        - Properties: system fields stored with `_` prefix, user filterable
          metadata stored bare, the layout the event memory store maps a filter's
          bare name (`producer_id`) and `m.<key>` onto. A search lifts the
          fields mapped above (`producer_id`, `created_at`) back into the
          memory's typed filters (`_lift_typed_filters`).

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

        return Event(
            uuid=uuid5(_EVENT_UUID_NAMESPACE, episode.uid),
            timestamp=episode.created_at,
            session_id=DEFAULT_SESSION_ID,
            source_id=episode.producer_id,
            context=Context(Author(name=episode.producer_id)),
            blocks=[TextBlock(text=episode.content)],
            properties=properties,
        )
