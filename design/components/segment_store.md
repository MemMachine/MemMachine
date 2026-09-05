# Segment store

Existing component, `episodic_memory/event_memory/segment_store/`, as
shipped in #1548 (`design/segment_store_shared_tables.md`). Derived
data: segments and their derivative links, rebuildable from the event
store. This file lists what changes; everything not listed stays.

## Constructed with

- `SQLAlchemySegmentStore(engine: AsyncEngine,
  settings: SegmentStoreSettings)`;
  settings: `purge_max_segments`, `purge_max_partitions`, `payload_codec`
  defaults.

## Storage, after the changes

`segment_store_pt`: `key UUID PK`, `config JSON`,
`created_at`. `segment_store_sg`: `key UUID`, `uuid UUID`, `event_uuid
UUID`, `index`, `offset`, `timestamp`, `timestamp_timezone_offset`,
`session_id TEXT`, `source_id TEXT`, `context BLOB`, `block_kind
TEXT`, `block BLOB`, `properties JSON`;
primary key `(key,
uuid)`. `segment_store_dv_ln`: `key UUID`, `uuid UUID`, `segment_uuid
UUID`, foreign key to the segment row with cascade. `segment_store_gc`:
`key UUID PK`, `enqueued_at`.

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
    async def create_partition(self, key: UUID, config: SegmentPartitionConfig) -> None
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
    async def get_segment_contexts(self, seed_segment_uuids: Iterable[UUID], *,
                                   max_backward_segments: int, max_forward_segments: int,
                                   since: datetime | None, before: datetime | None,
                                   source_ids: Iterable[str] | None,
                                   block_kinds: Iterable[str] | None,
                                   property_filter: FilterExpr | None) -> dict[UUID, list[Segment]]
    async def get_neighbours(self, anchor: UUID, *,
                             before: int, after: int,
                             session_ids: Iterable[str] | None,
                             source_ids: Iterable[str] | None,
                             block_kinds: Iterable[str] | None) -> list[Segment]
    async def get_segment_uuids_by_event_uuids(self,
                                               event_uuids: Iterable[UUID]) -> dict[UUID, list[UUID]]
    async def get_derivative_uuids_by_segment_uuids(self,
                                                    segment_uuids: Iterable[UUID]) -> dict[UUID, list[UUID]]
    async def find_segments(self, *,
                            since: datetime | None, before: datetime | None,
                            session_ids: Iterable[str] | None,
                            source_ids: Iterable[str] | None,
                            block_kinds: Iterable[str] | None,
                            property_filter: FilterExpr, limit: int) -> list[UUID]
    async def delete_segments(self, segment_uuids: Iterable[UUID]) -> None
```

`get_neighbours` serves expansion (`episodic_memory.md`): the segments
ordered by `(session_id, timestamp, event_uuid, index, offset)` within
the key, within the anchor's session, the
`before` segments preceding the anchor and the `after` following it,
optionally restricted to source ids and block kinds; the anchor itself
is included. The
order is total and stable, so a caller can walk by repeating the call
from the last segment returned.

## Changes required

- Key type `UUID` (`sqlalchemy_segment_store.py:145`, `String(255)`),
  and `validate_partition_key`, `PARTITION_KEY_MAX_BYTES` and
  `partition_key_for_session` (`long_term_memory/service_locator.py:166`)
  go.
- The incarnation goes: the `incarnation` column of every table
  (`:146`, `:158`, `:201`, `:234`) becomes the key, the registry row's
  unique incarnation goes, the purge queue is keyed by the key, the
  physical-key helper in `utils.py` goes, and the store mints nothing.
  Rationale in `server_redesign.md`, "Segment store".
- `SegmentStorePartition` (`segment_store.py:20`) becomes
  `SegmentPartition`: the same data operations, none taking a key,
  bound to the key at construction and stateless (no incarnation,
  nothing opened or closed, no stale state). `open_partition`,
  `open_or_create_partition` and `close_partition` (`:176`, `:191`,
  `:220`) go and `partition(key)`, which does no I/O, replaces them.
  The registry read that fences each operation returns the codec
  configuration; codec objects are cached process-wide by
  configuration.
- `create_partition` stays strict and also raises on a key whose purge
  is pending (a queue entry under the key).
- `purge_partition(key) -> Progress` is added: this key's dead rows,
  bounded by `purge_max_segments`; `DONE` when none remain. It is what
  the `sweep` job calls; `purge_deleted_partitions`
  stays for library users and the server does not run it.
- `get_segment_contexts` gains `since` and `before` on the real
  `timestamp` column, as on the reference branch (commit 27b3279b), and
  the reserved timestamp property key goes from the segment side; and
  `source_ids` and `block_kinds`, so a window is bounded by the same
  system filters as the hits it surrounds. `session_ids` is not needed
  there: the ordering index is per session, so a window never leaves
  its seed's session.
- `find_segments` is added for the selectivity probe under
  `filters_and_properties.md`: segments matching the system filters and
  a property filter, up to `limit + 1`, so the caller can tell
  "selective" from "broad".
- `segment_store_sg` gains `block_kind`, the kind name of the segment's
  one block as a plain column, since the encoded block cannot be
  filtered (`blocks.md`).
- `get_neighbours` is added for expansion, over the ordering index.
- The two ABCs stay two, `SegmentStore` and `SegmentPartition`, with
  the line between them redrawn: the store names keys, the partition
  never does.
- Errors: `SegmentStorePartitionHandleStaleError` becomes
  `KeyNotLiveError`; `SegmentStorePartitionAlreadyExistsError` becomes
  `KeyExistsError`; `SegmentStoreAttemptsExhaustedError` becomes
  `AttemptsExhaustedError`; `SegmentPartitionConfigMismatchError`
  goes with open-or-create.
- Fencing is unchanged: writes `FOR SHARE` the registry row for the
  transaction, the logical delete takes it `FOR UPDATE`, reads carry
  the liveness predicate; on SQLite `BEGIN IMMEDIATE`.
- Segmenter and deriver contracts gain a clause: a segment carries a
  verbatim copy of its event's properties, session id, source id and
  context, and a derivative of its segment's; and the unhandled-kind
  clause of `blocks.md`.

## Schema, after the changes

`segment_store_pt`, the registry row (the fence):

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key |
| `config` | `JSON` (`JSONB` on PostgreSQL) | not null |
| `created_at` | `DateTime(timezone=True)` | not null, `func.now()` |

`segment_store_sg`, the segments:

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key part |
| `uuid` | `Uuid` | primary key part |
| `event_uuid` | `Uuid` | not null |
| `index` | `Integer` | not null |
| `offset` | `Integer` | not null |
| `timestamp` | `DateTime(timezone=True)` | not null, UTC |
| `timestamp_timezone_offset` | `Integer` | not null, minutes |
| `session_id` | `Text` | null; copied from the event |
| `source_id` | `Text` | null; copied from the event |
| `context` | `LargeBinary` | null, codec-encoded; copied from the event, for rendering |
| `block_kind` | `Text` | not null; the block's kind name, for filtering |
| `block` | `LargeBinary` | not null, codec-encoded |
| `properties` | `JSON` (`JSONB` on PostgreSQL) | not null |

Indexes: `segment_store_sg__key_event (key, event_uuid, index, offset)` for
lookup by event; `segment_store_sg__key_source (key, source_id)` for
`source_ids` on context windows and expansion; `segment_store_sg__key_order
(key, session_id, timestamp, event_uuid, index, offset)` for context windows,
expansion and `since` and `before`, which is the one total order the store
exposes; a GIN index on `properties` on PostgreSQL, added by a deployment as
its undeclared-key filters need.

`segment_store_dv_ln`, the derivative links:

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key part |
| `uuid` | `Uuid` | primary key part, the derivative uuid |
| `segment_uuid` | `Uuid` | not null; foreign key `(key, segment_uuid)` to `segment_store_sg (key, uuid)` `ON DELETE CASCADE` |

Index: `segment_store_dv_ln__key_segment (key, segment_uuid)`, which the
cascade and `get_derivative_uuids_by_segment_uuids` use.

`segment_store_gc`, the purge queue: `key Uuid` primary key,
`enqueued_at DateTime(timezone=True)` not null `func.now()`, index
`segment_store_gc__enqueued_at`.

No foreign key from the data tables to the registry row, so the logical
delete is O(1); the link table's cascade from segments is kept, and an
engine that does not enforce it leaves link rows the purge removes with
a warning, as today.
