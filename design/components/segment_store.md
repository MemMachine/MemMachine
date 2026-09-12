# Segment store

Existing component, `episodic_memory/event_memory/segment_store/`, as
shipped in #1548 (`design/segment_store_shared_tables.md`). Derived
data: segments and their derivative links, rebuildable from the event
store. This file lists what changes; everything not listed stays.

## Constructed with

- `SQLAlchemySegmentStore(engine: AsyncEngine,
  settings: SegmentStoreSettings)`;
  settings: `purge_max_segments`, `payload_codec` defaults,
  `property_indexes`.

## Storage, after the changes

`segment_store_pt`: `key UUID PK`, `config JSON`, `created_at`.
`segment_store_sg`: `key UUID`, `uuid UUID`,
`event_uuid UUID`, `event_position`, `index`, `offset`, `timestamp`,
`timestamp_timezone_offset`, `session_id TEXT`, `source_id TEXT`,
`context BLOB`, `block_kind TEXT`, `block BLOB`, `properties JSON`;
primary key `(key, uuid)`. `segment_store_dv_ln`: `key UUID`, `uuid
UUID`, `segment_uuid UUID`, foreign key to the segment row with cascade.
`segment_store_gc`: `key UUID PK`, `enqueued_at`, the purge queue.

## API, after the changes

Two ABCs: the store, the resource and the only place a key is named
(lifecycle, and constructing handles); and the partition, the stateless
handle every data consumer holds (`server_redesign.md`, "Vocabulary").
No method on the partition takes a key, so a consumer cannot name a
wrong one, and it cannot reach lifecycle. Each backend implements both,
as the current code does with `SegmentStorePartition`, minus the
incarnation, the open and close, and the stale state. `partition(key)`
does no I/O and checks nothing; every operation through the handle
fences on the registry row exactly as the current code does.

```python
class SegmentStore(ABC):                  # the resource: lifecycle, and handles
    async def create_partition(self, key: UUID) -> None
    async def delete_partition(self, key: UUID) -> None
    async def purge_partition(self, key: UUID) -> Progress
    async def purge_deleted_partitions(self) -> bool     # library use only
    def partition(self, key: UUID) -> SegmentPartition   # stateless handle, no I/O
    @property
    def concurrency_scope(self) -> ConcurrencyScope

class SegmentPartition(ABC):              # data, bound to one key; no method takes a key
    @property
    def key(self) -> UUID
    async def add_segments(self,
                           segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]]) -> None
    async def get_segments(self, segment_uuids: Iterable[UUID], *,
                           since: datetime | None, until: datetime | None,
                           session_ids: Iterable[str] | None,
                           source_ids: Iterable[str] | None,
                           block_kinds: Iterable[str] | None,
                           property_filter: FilterExpr | None) -> dict[UUID, Segment]
    async def get_segment_neighborhoods(self, seed_segment_uuids: Iterable[UUID], *,
                                 before: int, after: int,
                                 since: datetime | None, until: datetime | None,
                                 session_ids: Iterable[str] | None,
                                 source_ids: Iterable[str] | None,
                                 block_kinds: Iterable[str] | None,
                                 property_filter: FilterExpr | None) -> dict[UUID, Neighborhood]
    async def delete_derivatives(self, derivative_uuids: Iterable[UUID]) -> None
```

The one total order. Segments within a key are ordered by
`(timestamp, event_position, index, offset)`: timestamp ties break by
the event's position in the event store, which is the order the events
were ingested in, and a segment's place within its event by index and
offset. Segment windows and expansion walk this order confined to the
seed's or anchor's session by an equality predicate on its session id,
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
the ones that pass; expansion walks from its anchor and, when sessions are
named, first checks that the anchor is in one of them. The order is total
and stable, so a caller walks further by repeating the call from the first of
`before` or the last of `after`. That is the rule decided for MemMachine #1498:
during a search a seed that fails the filter is dropped before any window is
built, and after a search the neighborhood is kept even when its seed would
fail, in which case the seed is never returned. Both apply `since`, `until`,
`source_ids`, `block_kinds` and `property_filter` to the surrounding rows, so a
window or neighborhood is bounded by the same filters as the search that led
to it.

## Changes required

- Key type `UUID` (`sqlalchemy_segment_store.py:145`, `String(255)`),
  and `validate_partition_key`, `PARTITION_KEY_MAX_BYTES` and
  `partition_key_for_session` (`long_term_memory/service_locator.py:166`)
  go.
- The incarnation goes: the `incarnation` column of every table
  (`:146`, `:158`, `:201`, `:234`) becomes the key, the registry row's
  unique incarnation goes, the physical-key helper in `utils.py` goes,
  and the store mints nothing. Rationale in `server_redesign.md`,
  "Segment store".
- The registry row and the purge queue stay as shipped, keyed by the
  key: the logical delete removes the row and enqueues the key in one
  transaction, and a key is in one of two conditions the store can
  observe, a row (live) or a queue entry (dropping). Those two give the
  same outward behavior as the key-registry stores' `live` and
  `dropping` without a second component: a row or an entry refuses
  create; no row refuses data operations; `purge` proceeds on an entry
  and raises `KeyLiveError` on a row with no entry.
- `SegmentStorePartition` (`segment_store.py:20`) becomes
  `SegmentPartition`: the same data operations, none taking a key,
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
- `segment_store_sg` gains `block_kind`, the kind name of the segment's
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
- The two ABCs stay two, `SegmentStore` and `SegmentPartition`, with
  the line between them redrawn: the store names keys, the partition
  never does.
- Errors: `SegmentStorePartitionHandleStaleError` becomes
  `KeyNotLiveError`; `SegmentStorePartitionAlreadyExistsError` becomes
  `KeyExistsError`; `SegmentStoreAttemptsExhaustedError` becomes
  `AttemptsExhaustedError`; `SegmentPartitionConfigMismatchError`
  goes with open-or-create.
- Fencing is unchanged: writes pin the registry row for the
  transaction (`FOR SHARE` on PostgreSQL; the self-checking registry
  `UPDATE` on SQLite, as shipped), the logical delete takes it
  exclusively, reads carry the liveness predicate, the row's existence.
- Segmenter and deriver contracts gain a clause: a segment carries a
  verbatim copy of its event's properties, session id, source id,
  context, timestamp with offset and position, and a derivative of its
  segment's; and the unhandled-kind clause of `blocks.md`.

## Schema, after the changes

`segment_store_pt`, the registry row (the fence):

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key |
| `config` | `JSON` (`JSONB` on PostgreSQL) | not null |
| `created_at` | `DateTime(timezone=True)` | not null, `func.now()` |

`segment_store_gc`, the purge queue: `key Uuid` primary key,
`enqueued_at DateTime(timezone=True)` not null `func.now()`, index
`segment_store_gc__enqueued_at`.

`segment_store_sg`, the segments:

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key part |
| `uuid` | `Uuid` | primary key part |
| `event_uuid` | `Uuid` | not null |
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

Indexes: `segment_store_sg__key_event (key, event_uuid, index, offset)`
for lookup by event; `segment_store_sg__key_source (key, source_id)` for
`source_ids` on segment windows and expansion; `segment_store_sg__key_order
(key, session_id, timestamp, event_position, index, offset)` for context
windows, expansion and `since` and `until`, which is the one total order
the store exposes; expression indexes on `properties` for the keys a
deployment names in `segment_store.property_indexes`, created by the
schema command.

`segment_store_dv_ln`, the derivative links:

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key part |
| `uuid` | `Uuid` | primary key part, the derivative uuid |
| `segment_uuid` | `Uuid` | not null; foreign key `(key, segment_uuid)` to `segment_store_sg (key, uuid)` `ON DELETE CASCADE` |

Index: `segment_store_dv_ln__key_segment (key, segment_uuid)`, which the
cascade and `get_derivative_uuids_by_segment_uuids` use.

No foreign key from the data tables to the registry row, so the logical
delete is O(1); the link table's cascade from segments is kept, and an
engine that does not enforce it leaves link rows the purge removes with
a warning, as today.
