# Event memory store

Existing component, `episodic_memory/event_memory/event_memory_store/`, as
shipped in #1548 (`design/event_memory_store_shared_tables.md`) and #1659
(the write transaction and the event rows; the name follows its
dependent since the store holds events, segments and links). Derived
data: the events held, their segments and derivative links, rebuildable
from the event store. This file lists what changes; everything not
listed stays.

## Constructed with

- `SQLAlchemyEventMemoryStore(engine: AsyncEngine,
  settings: EventMemoryStoreSettings)`;
  settings: `purge_max_segments`, `payload_codec` defaults,
  `property_indexes`.

## Storage, after the changes

`event_memory_store_pt`: `key UUID PK`, `config JSON`, `created_at`.
`event_memory_store_ev`: `key UUID`, `uuid UUID`, the event uuid; primary
key `(key, uuid)`, the row that holds an event once.
`event_memory_store_sg`: `key UUID`, `uuid UUID`,
`event_uuid UUID`, `event_position`, `index`, `offset`, `timestamp`,
`timestamp_timezone_offset`, `session_id VARCHAR(255)`, `source_id VARCHAR(255)`,
`context BLOB`, `block_kind VARCHAR(255)`, `block BLOB`, `properties JSON`;
primary key `(key, uuid)`, foreign key to the event row with cascade.
`event_memory_store_dv_ln`: `key UUID`, `uuid
UUID`, `segment_uuid UUID`, foreign key to the segment row with cascade.
`event_memory_store_gc`: `key UUID PK`, `enqueued_at`, the purge queue.

## API, after the changes

Two ABCs: the store, the resource and the only place a key is named
(lifecycle, and constructing handles); and the partition, the stateless
handle every data consumer holds (`server_redesign.md`, "Vocabulary").
No method on the partition takes a key, so a consumer cannot name a
wrong one, and it cannot reach lifecycle. Each backend implements both,
as the current code does with `EventMemoryStorePartition`, minus the
incarnation, the open and close, and the stale state. `partition(key)`
does no I/O and checks nothing; every operation through the handle
fences on the registry row exactly as the current code does.

```python
class EventMemoryStore(ABC):                  # the resource: lifecycle, and handles
    async def create_partition(self, key: UUID) -> None
    async def delete_partition(self, key: UUID) -> None
    async def purge_partition(self, key: UUID) -> Progress
    async def purge_deleted_partitions(self) -> bool     # library use only
    def partition(self, key: UUID) -> EventMemoryPartition   # stateless handle, no I/O
    @property
    def concurrency_scope(self) -> ConcurrencyScope

class EventMemoryPartition(ABC):              # data, bound to one key; no method takes a key
    @property
    def key(self) -> UUID
    def write(self, *, exclusive: bool = False
              ) -> AbstractAsyncContextManager[EventMemoryPartitionWriter]
    async def get_segments(self, segment_uuids: Iterable[UUID], *,
                           since: datetime | None, until: datetime | None,
                           session_ids: Iterable[str] | None,
                           source_ids: Iterable[str] | None,
                           block_kinds: Iterable[str] | None,
                           property_filter: FilterExpr | None) -> dict[UUID, Segment]
    async def get_segment_neighborhoods(self, seed_uuids: Iterable[UUID], *,
                                 before: int, after: int,
                                 since: datetime | None, until: datetime | None,
                                 session_ids: Iterable[str] | None,
                                 source_ids: Iterable[str] | None,
                                 block_kinds: Iterable[str] | None,
                                 property_filter: FilterExpr | None) -> dict[UUID, Neighborhood]
    async def get_derivative_uuids_by_event_uuids(self,
                           event_uuids: Iterable[UUID]) -> dict[UUID, list[UUID]]
    async def delete_events(self, event_uuids: Iterable[UUID]) -> None
    async def delete_derivatives(self, derivative_uuids: Iterable[UUID]) -> None

class EventMemoryPartitionWriter(ABC):        # one write transaction, inside write() only
    async def add_events(self,
                         events: Mapping[UUID, Mapping[Segment, Iterable[UUID]]]) -> None
    async def get_segment_uuids_by_derivative_uuids(self,
                         derivative_uuids: Iterable[UUID]) -> dict[UUID, UUID]
```

The write transaction. `write()` is a context manager: entering it
fences on the registry row and pins the partition against deletion,
normal exit commits, an exception rolls the block back, so a caller
makes a write conditional on work of its own inside it, which is how
`EpisodicMemory` writes its vector records: the segments commit only
once the vector store has acknowledged them. `add_events` is keyed by
event, and the partition holds an event once: a batch naming a held
event is rejected whole, before anything is stored, by the event row's
primary key, exactly under concurrency on both dialects.
`write(exclusive=True)` takes the row exclusively, waiting for every
write in flight and excluding new ones until the block exits; a reader
that must see the partition settled uses it, and read repair does.
`delete_events` cascades to segments and links and frees the uuid;
`delete_derivatives` and `delete_segments` (eviction) leave the event
held, so an evicted event is not resurrected by a replay.

The one total order. Segments within a key are ordered by
`(timestamp, event_position, index, offset)`: timestamp ties break by
the event's position in the event store, which is the order the events
were ingested in, and a segment's place within its event by index and
offset. Segment windows and expansion walk this order confined to the
seed's session by an equality predicate on its session id,
so they never cross into another conversation interleaved in time.
Every segment has a session, so every walk pins one. A session list
holds ids only, and a filter that names no session means every session.

The two reads have different jobs. `before`
and `after` count neighbors on each side of a given segment; `since` and `until` bound
the timestamp, inclusive and exclusive; `source_ids`, `block_kinds` and
`property_filter` select rows; every walk stays in its seed's session.
`get_segments` is the filtered lookup: the segments among the given uuids
that the partition holds and that pass every filter; a uuid that fails has
no entry. `get_segment_neighborhoods` is the walk: it locates each seed by
uuid whether or not it passes any filter, applies the filters to the
neighbors only, and never returns the seed; the result is two lists in the
store's order, `Neighborhood(before, after)`, with the seed's place between
them, so nothing in it can be mistaken for the seed. Filters select what a
read returns: a search fetches its seeds with the filters and walks from
the ones that pass; expansion walks from its seed segment and, when
sessions are named, first checks that the seed is in one of them. The
order is total and stable. That is the rule decided for MemMachine #1498:
during a search a seed that fails the filter is dropped before any window is
built, and after a search the neighborhood is kept even when its seed would
fail, in which case the seed is never returned. Both apply `since`, `until`,
`source_ids`, `block_kinds` and `property_filter` to the surrounding rows, so a
window or neighborhood is bounded by the same filters as the search that led
to it.

## Changes required

- Key type `UUID` (`sqlalchemy_event_memory_store.py:145`, `String(255)`),
  and `validate_partition_key`, `PARTITION_KEY_MAX_BYTES` and
  `partition_key_for_session` (`long_term_memory/service_locator.py:166`)
  go.
- The incarnation goes: the `incarnation` column of every table
  (`:146`, `:158`, `:201`, `:234`) becomes the key, the registry row's
  unique incarnation goes, the physical-key helper in `utils.py` goes,
  and the store mints nothing. Rationale in `server_redesign.md`,
  "Event memory store".
- The registry row and the purge queue stay as shipped, keyed by the
  key: the logical delete removes the row and enqueues the key in one
  transaction, and a key is in one of two conditions the store can
  observe, a row (live) or a queue entry (dropping). Those two give the
  same outward behavior as the key-registry stores' `live` and
  `dropping` without a second component: a row or an entry refuses
  create; no row refuses data operations; `purge` proceeds on an entry
  and raises `KeyLiveError` on a row with no entry.
- `EventMemoryStorePartition` (`event_memory_store.py:20`) becomes
  `EventMemoryPartition`: the same data operations, none taking a key,
  bound to the key at construction and stateless (no incarnation,
  nothing opened or closed, no stale state). `open_partition`,
  `open_or_create_partition` and `close_partition` (`:176`, `:191`,
  `:220`) go and `partition(key)`, which does no I/O, replaces them.
  The registry read that fences each operation returns the codec
  configuration; codec objects are cached process-wide by
  configuration.
- `create_partition` stays strict: `KeyExistsError` on a row in any
  state. The `config` parameter goes: the row records the store's
  `payload_codec` setting at create.
- `purge_partition(key) -> Progress` is added: this key's rows while
  its queue entry exists, bounded by `purge_max_segments`; `DONE` when
  none remain and the entry is removed. It is what the `sweep` job
  calls; `purge_deleted_partitions` stays for library users, keeps its
  `bool`, and the server does not run it.
- `Segment` gains `event_position`, copied from the `StoredEvent` the
  segmenter was given, and the row the column; the ordering index
  changes accordingly.
- `get_segment_windows` becomes `get_segments`, a filtered lookup by
  uuid: `since` and `until` on the real `timestamp` column, as on the
  reference branch (commit 27b3279b, where the pair is `since` and
  `before`), and the reserved timestamp property key goes from the
  segment side; `session_ids`, `source_ids` and `block_kinds`; no
  window counts. `session_ids` selects what a read returns; the walk's
  confinement to the given segment's session is a separate rule.
- `event_memory_store_sg` gains `block_kind`, the kind name of the segment's
  one block as a plain column, since the encoded block cannot be
  filtered (`blocks.md`).
- `get_segment_neighborhoods` is the one walk, over the ordering index,
  from seeds named by uuid (`max_backward_segments` and
  `max_forward_segments` become `before` and `after`), confined to the
  given segment's session by the store, returning the neighbors and
  never the segment. It takes no `session_ids`: confinement decides the
  session, and the lookup decides which segments are visible.
- `delete_derivatives` is added for eviction (`episodic_memory.md`):
  removes link rows by derivative uuid and leaves the segments.
- From #1659, keyed by the key instead of the incarnation: the event
  table and the segment table's foreign key to it; `write()` and the
  writer's `add_events`, which replaced `add_segments`;
  `delete_events`; `get_derivative_uuids_by_event_uuids`, which
  replaced `get_segment_uuids_by_event_uuids` and
  `get_derivative_uuids_by_segment_uuids`; the purge reclaiming event
  rows after the segments, on the same budget; the exclusive fence.
- The two ABCs stay two, `EventMemoryStore` and `EventMemoryPartition`, with
  the line between them redrawn: the store names keys, the partition
  never does.
- Errors: `EventMemoryStorePartitionHandleStaleError` becomes
  `KeyNotLiveError`; `EventMemoryStorePartitionAlreadyExistsError` becomes
  `KeyExistsError`; `EventMemoryStoreAttemptsExhaustedError` becomes
  `AttemptsExhaustedError`; `EventMemoryPartitionConfigMismatchError`
  goes with open-or-create.
- Fencing is unchanged: writes pin the registry row for the
  transaction (`FOR SHARE` on PostgreSQL; the self-checking registry
  `UPDATE` on SQLite, as shipped), an exclusive write and the logical
  delete take it exclusively, reads carry the liveness predicate, the
  row's existence.
- Segmenter and deriver contracts gain a clause: a segment carries a
  verbatim copy of its event's properties, session id, source id,
  context, timestamp with offset and position, and a derivative of its
  segment's; and the unhandled-kind clause of `blocks.md`.

## Schema, after the changes

`event_memory_store_pt`, the registry row (the fence):

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key |
| `config` | `JSON` (`JSONB` on PostgreSQL) | not null |
| `created_at` | `DateTime(timezone=True)` | not null, `func.now()` |

`event_memory_store_gc`, the purge queue: `key Uuid` primary key,
`enqueued_at DateTime(timezone=True)` not null `func.now()`, index
`event_memory_store_gc__enqueued_at`.

`event_memory_store_ev`, the events held: `key Uuid` and `uuid Uuid`, the
event uuid, primary key `(key, uuid)`; nothing else, the row is the
fact that the partition holds the event.

`event_memory_store_sg`, the segments:

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key part |
| `uuid` | `Uuid` | primary key part |
| `event_uuid` | `Uuid` | not null; foreign key `(key, event_uuid)` to `event_memory_store_ev (key, uuid)` `ON DELETE CASCADE` |
| `event_position` | `BigInteger` | not null; the event's position in the event store |
| `index` | `Integer` | not null |
| `offset` | `Integer` | not null |
| `timestamp` | `DateTime(timezone=True)` | not null, UTC |
| `timestamp_timezone_offset` | `Integer` | not null, minutes |
| `session_id` | `Text` | null; copied from the event |
| `source_id` | `Text` | null; copied from the event |
| `context` | `LargeBinary` | null, codec-encoded; copied from the event, for rendering |
| `block_kind` | `Text` | not null; `block.kind` projected out of the encoded block at write, for filtering |
| `block` | `LargeBinary` | not null, codec-encoded |
| `properties` | `JSON` (`JSONB` on PostgreSQL) | not null |

Indexes: `event_memory_store_sg__key_event (key, event_uuid)` for lookup by
event and the cascade from the event row; `event_memory_store_sg__key_order (key, session_id, timestamp,
event_position, index, offset)` for context windows, expansion and
`since` and `until`, which is the one total order the store exposes (a
walk filtered by source or kind scans past the session's other rows; a
pinned walk index is added when a workload shows such walks); expression indexes on `properties` for the keys a
deployment names in `event_memory_store.property_indexes`, created by the
schema command.

`event_memory_store_dv_ln`, the derivative links:

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key part |
| `uuid` | `Uuid` | primary key part, the derivative uuid |
| `segment_uuid` | `Uuid` | not null; foreign key `(key, segment_uuid)` to `event_memory_store_sg (key, uuid)` `ON DELETE CASCADE` |

Index: `event_memory_store_dv_ln__key_segment (key, segment_uuid)`, which the
cascade and `get_derivative_uuids_by_event_uuids` use.

No foreign key from the data tables to the registry row, so the logical
delete is O(1); the cascades from events to segments and from segments
to links are kept, and an engine that does not enforce them leaves rows
the purge removes with a warning, as today.
