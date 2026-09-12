# Event store

New component. The system of record: a tenant's events as ingested,
plus a per-tenant log of additions and deletions in position order,
which memory subsystems replay. Never derived from anything.

## Constructed with

`SqlAlchemyEventStore(engine: AsyncEngine, settings: EventStoreSettings)`;
settings: `blocks.max_bytes`, `context.max_bytes`, `properties.max_keys`,
`properties.max_string_bytes`, `payload_codec` (codec settings for
blocks and context), `purge_batch` (rows per `purge` call).

## Types

`Event` from `event_memory/data_types.py` (`Event.uuid: UUID`), with
`Event.session_id: str` (required) and `Event.source_id: str | None` added
beside `timestamp`, each a string of at most
`properties.max_string_bytes`; `Event.blocks`, a list of the registered
block kinds (see `blocks.md`); and `Event.context: Context`, a mapping
from part kind to one registered part (see `context.md`), never `None`,
the empty mapping being no context. `Event.timestamp` is an aware
datetime and keeps the offset it was ingested with.

```python
class StoredEvent(Event):
    position: int                  # the position of the event's `added` entry

class IngestResult(BaseModel):
    stored: list[UUID]
    skipped: list[UUID]
    position: int                  # the key's head after the write

class LogEntry(BaseModel):
    position: int                  # monotonic and commit-ordered within a key
    kind: Literal["added", "deleted"]
    uuid: UUID
    event: StoredEvent | None      # the content for "added", unless superseded
```

`StoredEvent` exists because a subsystem needs the position: the
watermark is one, and the segment order breaks timestamp ties with it.
`IngestResult` is what ingest returns and what the API responds with.

## Schema

`event_store_pt`, the registry row (the fence):

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key |
| `config` | `JSON` | not null; the codec configuration at create |
| `next_position` | `BigInteger` | not null, default 1 |
| `created_at` | `DateTime(timezone=True)` | not null, `func.now()` |

`event_store_gc`, the purge queue: `key Uuid` primary key,
`enqueued_at DateTime(timezone=True)` not null `func.now()`, index
`event_store_gc__enqueued_at`. A key is in one of two conditions the
store can observe, a registry row (live) or a queue entry (dropping),
and those two give the store the same outward behavior as a
key-registry store without a second component: the row's existence is
the fence, checked in the same transaction as the data statement, and
the queue is the durable, ordered list of what remains to purge, written
in the same transaction as the delete.

`event_store_ev`, the events:

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key part |
| `uuid` | `Uuid` | primary key part |
| `position` | `BigInteger` | not null; unique `(key, position)`; the position of the event's `added` entry |
| `timestamp` | `DateTime(timezone=True)` | not null, UTC |
| `timestamp_timezone_offset` | `Integer` | not null, minutes; the offset as ingested |
| `session_id` | `Text` | null |
| `source_id` | `Text` | null |
| `context` | `LargeBinary` | null, codec-encoded |
| `properties` | `JSON` (`JSONB` on PostgreSQL) | not null |
| `blocks` | `LargeBinary` | not null, codec-encoded |
| `ingested_at` | `DateTime(timezone=True)` | not null, `func.now()` |

Indexes: `event_store_ev__key_timestamp (key, timestamp)` for
`list_events` by time; `event_store_ev__key_session (key, session_id,
timestamp)`; `event_store_ev__key_source (key, source_id)`; expression
indexes on `properties` for the keys a deployment names in
`event_store.property_indexes`, created by the schema command.

Round trip. Every field of an event is stored so that `get_events`
returns what `add_events` took: the timestamp with its original offset,
the session and source ids, the context and blocks byte for byte
through the codec, the properties with their types. A segment carries
the same fields copied from its event, so a segment round-trips too.

`event_store_lg`, the log:

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key part |
| `position` | `BigInteger` | primary key part |
| `kind` | `String(8)` | check in (`added`, `deleted`) |
| `uuid` | `Uuid` | not null |
| `at` | `DateTime(timezone=True)` | not null, `func.now()` |

No foreign key from `event_store_lg` or `event_store_ev` to the
registry row: the logical delete is one row flip and the purge purges
by key. No foreign key between the log and the events: an `added` entry
outlives its event once the event is deleted, by design.

## API

Two ABCs, as for every store: the store, the only place a key is named
(lifecycle, and constructing handles), and the partition, the stateless
handle bound to one key that every data caller holds, with no method
taking a key. The ingest service builds the partition for the tenant in
the request path and `EpisodicMemoryManager.replay` for the tenant of
its job, each at the one point the key is read.

```python
class EventStore(ABC):                    # the resource: lifecycle, and handles
    async def create_partition(self, key: UUID) -> None
    async def delete_partition(self, key: UUID) -> None
    async def purge_partition(self, key: UUID) -> Progress
    async def purge_deleted_partitions(self) -> bool       # library use only
    async def compact_log(self, key: UUID, below: int) -> Progress
    def partition(self, key: UUID) -> EventPartition       # stateless handle, no I/O
    @property
    def concurrency_scope(self) -> ConcurrencyScope

class EventPartition(ABC):                # data, bound to one key; no method takes a key
    @property
    def key(self) -> UUID
    async def add_events(self, events: Iterable[Event]) -> IngestResult
    async def delete_events(self, uuids: Iterable[UUID]) -> int      # the head after
    async def get_events(self, uuids: Iterable[UUID]) -> list[StoredEvent]
    async def list_events(self, *, filter: FilterExpr | None,
                          since: datetime | None, until: datetime | None,
                          after: int | None, limit: int) -> list[StoredEvent]
    async def read_log(self, after: int, limit: int) -> list[LogEntry]
    async def read_events_after(self, after: int, limit: int) -> list[StoredEvent]
    async def head(self) -> int                            # last position
```

- `create_partition`: strict; `KeyExistsError` on any row or queue
  entry under the key. The row records the store's `payload_codec`
  setting as the partition's configuration, so a later change of the
  setting applies to new partitions only.
- `delete_partition`: one transaction: lock the row, enqueue the key,
  remove the row. O(1), idempotent.
- `purge_partition`: with a queue entry, delete up to `purge_batch` of
  the key's rows across the three data tables; `MORE` while rows
  remain; remove the entry when none do and return `DONE`. With neither
  entry nor row, `DONE` after finding nothing under the key. With a row
  and no entry, the key is live: raise `KeyLiveError`. On SQLite the
  `DELETE` waits on the write lock up to `busy_timeout` and raises past
  it; the reconciler retries.
- `purge_deleted_partitions`: for library users without a tenant
  service: one `purge_partition` batch for the oldest queue entry;
  `True` while any entry remains. The server does not run it.
- `add_events`: one transaction that locks the registry row `FOR
  UPDATE` (on SQLite, the self-checking `UPDATE` of `next_position`
  serves as the lock, as in the segment store), so ingests to one
  tenant serialize at the event store and positions are commit-ordered
  within the key; validates every event (block bytes against
  `blocks.max_bytes`, context bytes against `context.max_bytes`,
  property keys and values, reserved keys, the session and source id
  lengths, raising `InvalidEventError` with the field and the reason);
  inserts each new event and one `added` log entry, taking positions
  from `next_position` and advancing it; events whose uuid exists under
  the key are skipped and returned in `skipped`. `position` is the
  key's head after the transaction, which is the position a caller
  waits on, and is the previous head when every event was skipped. The
  lock is exclusive rather than shared because positions must be
  commit-ordered; its cost is that a tenant's ingests do not overlap
  inside this one short transaction. This is the one place the bounds
  are enforced; the API relays the error as 422.
- `delete_events`: the same lock; remove the event rows; append one
  `deleted` log entry per uuid that existed. Idempotent: a uuid with no
  row appends nothing. Returns the head after the transaction.
- `get_events`: the events, in the order asked.
- `list_events`: the events in position order, with `position >
  after`, at most `limit`, filtered by `since` and `until` on the
  timestamp and by `filter` on the properties. The cursor is the last
  position returned; a caller passes it back as `after`. Positions are
  stable, so a listing is exact under concurrent ingest.
- `read_events_after`: events with `position > after`, ascending; the
  bootstrap and history source for a subsystem.
- `read_log`: entries with `position > after`, ascending, at most
  `limit`; an `added` entry carries the event whose `position` is the
  entry's own, so an entry superseded by a delete and a re-ingest of
  the same uuid carries `None`, as does one whose event has since been
  deleted, and a subsystem skips it.
- Reads carry `EXISTS (registry row)`; a read on a key with no row
  raises `KeyNotLiveError`.

## What positions are for

A position is the log's order within a tenant, monotonic and
commit-ordered; it is not contiguous once the log is compacted or an
event is deleted, and nothing depends on contiguity. It exists so that
a subsystem's progress is one integer per tenant (the watermark), so
that "what is left to process" is a range the subsystem reads from that
integer, so that lag is a subtraction (`head` minus watermark, the
status endpoint), so that an event's addition is always replayed before
its deletion, and so that two events with one timestamp have a fixed
order in the segment store. Positions are internal to the server:
events are addressed by uuid everywhere else, and a position is never a
request parameter except as the listing cursor. Commit order is what
makes "read after p" exact, and it is why `add_events` takes the key's
row exclusively rather than a sequence.

## Why a log

A subsystem replays the log from its watermark, so every addition and
deletion a client was acknowledged for reaches every subsystem at least
once without the client retrying: the request path's inline processing
is a latency optimization and `replay` is the guarantee. The log is the
per-tenant data queue; an entry and its event are one transaction,
which a broker could not join without an outbox on top.

## Compaction

Log entries below every subsystem's watermark are not needed for
replay, and `compact_log(key, below)` removes them in bounded steps; a
subsystem enabled on an existing tenant bootstraps from
`read_events_after(0, ...)`, the events in position order, not from the
log, which is why the event row carries its position, and switches to
`read_log` once its watermark is at or past the log's oldest entry.
`deleted` entries below every watermark are always removable; `added`
entries are too, since the events table is the bootstrap source. Nothing
in the server schedules compaction: `memmachine events compact
--settings PATH` runs it for every live tenant, with `below` the
minimum of the tenant's subsystem watermarks from the tenant service,
bounded per tenant per run, and a deployment runs it as it runs the
schema command. The log holds one short row per addition or deletion,
so an uncompacted log grows slower than the events table it indexes; a
scheduled duty can replace the command if that ever matters.

## Concurrency scope

`cluster` on PostgreSQL; `host` on a SQLite file; `process` in memory.

## Changes to existing code

Replaces `common/episode_store/` (`EpisodeStorage`, the `episodestore`
table, `CountCachingEpisodeStorage`) and the `Episode` model. Not a
rename: the event's shape is `event_memory/data_types.py`'s `Event`,
blocks and context are codec-encoded, and the log exists. Nothing is
carried over.
