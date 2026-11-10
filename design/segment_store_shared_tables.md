# Segment store: shared tables with incarnation-scoped tenant keys

Status: accepted 2026-08-31. Supersedes the per-tenant partitioned layout of
the SQLAlchemy segment store on all dialects.

## Problem

On PostgreSQL, every partition (tenant) had its own pair of child tables
under LIST-partitioned parents, created and dropped by per-tenant DDL. That
layout produced a family of defects, each diagnosed and measured on the way
here:

- `DROP ... CASCADE` of one tenant's child tables removed the store-wide
  parent-level foreign key, ending referential integrity for every other
  tenant (#1544).
- Any statement with the partition key as a bind parameter switched to a
  cached generic plan after five executions, and a generic plan locks every
  child partition per execution: 9 locks per read with a custom plan, 319
  with the generic plan at 316 partitions, and HTTP 500 `out of shared
  memory` under 128-way concurrency (#1546).
- PostgreSQL's foreign-key integrity triggers address the partitioned
  parents internally, so writes and deletes kept a per-backend lock spike
  proportional to partition count even after the client statements were
  fixed (10 -> 166 -> 10 locks across the generic-plan attempt).
- Create and delete DDL against the shared parents deadlocked with writers
  and with each other, on upstream and on every intermediate fix.
- Each fix added machinery: detach-before-drop, a store-wide management
  lock, per-partition ORM entities with a memoized cache, statement-cache
  tuning.

Two scaling requirements then ruled out every per-tenant-table layout:
tenant creation must stay cheap up to hundreds of thousands to millions of
tenants (a per-tenant table costs about 5 catalog relations plus disk
files, which PostgreSQL does not survive at those counts), and a single
tenant may hold tens of thousands to tens of millions of rows.

## Design

One physical schema on every dialect; the ORM models are the tables.

The store holds no tenant names. Its owner's [partition
registry](partition_registry.md) names each tenant, mints an `incarnation`
(a random UUID) for each of its lives, and never mints one again while any
trace of it remains; the store addresses everything by incarnation.

- `segment_store_pt_in`: one row per prepared incarnation, holding its
  payload codec config and, while the incarnation is purged, the purge's
  cursor, the last segment key purged.
- `segment_store_sg` and `segment_store_dv_ln`, the segment and
  derivative-link tables, shared by all tenants and not partitioned. Rows
  carry no tenant name; they are keyed by the incarnation alone, so
  addressing the wrong tenant is structurally impossible rather than
  guarded against. A 16-byte UUID also keeps index entries narrower than a
  composite string key, and random UUIDs are unique across nodes without
  coordination, so a tenant's rows move between databases verbatim.

The link table keeps its foreign key to the segment table (`ON DELETE
CASCADE`), both sides keyed by incarnation. The segment table has no
foreign key to the incarnation's row: the purge deletes an incarnation's
segments in batches, and its row last.

"Incarnation" is the established term for a per-lifetime random identifier
(cf. Kafka's broker incarnation ID). It is deliberately not "generation" or
"epoch", which would imply ordered, comparable values.

### Tenant lifecycle

- **Prepare**: insert the incarnation's row with its codec config. No DDL,
  no lock beyond the row, microseconds. Preparing again with the same
  config changes nothing; a different config raises the config-mismatch
  error.
- **Open**: read the incarnation's row. The handle captures the
  incarnation and the codec.
- **Delete** is the owner's: unregistering the tenant's name in its
  registry is O(1) regardless of tenant size, and the tenant is unreachable
  at once. The store is not called.
- **Re-create**: the owner registers a fresh incarnation, safe at any
  time; the old incarnation's rows are invisible to the successor even
  while its purge is still sweeping them.
- **Purge**: `purge_partition_incarnation(incarnation) -> bool` on the
  `SegmentStore` ABC. The owner's purge rounds call it for a deleted
  incarnation once the registry's retention has passed, until it returns
  False; how much one call does is implementation policy. The caller is
  promised only that a call does not noticeably degrade concurrent request
  serving, and every bound below exists to keep that promise. The deletion
  contract is "unreachable immediately, erased asynchronously".

  - One transaction per call, committing its progress or nothing, so
    committed progress survives interruption.
  - Segment rows are deleted in batches of up to
    `SQLAlchemySegmentStoreParams.purge_max_segments` per call. A batch
    continues from the incarnation's cursor: it reads the next keys after
    `purged_through` in primary-key order, deletes that key range, and
    records its last key in the same transaction. No batch reads a row an
    earlier batch deleted; PostgreSQL keeps deleted rows in the table and
    its indexes until vacuum, and a batch that started from the
    incarnation's first key would step over all of them, making each
    batch, and the call that finds the incarnation empty, cost in
    proportion to what was already purged. The cursor relies on no row
    being written into an incarnation once its purge starts, which the
    retention guarantees: every write finishes within a bound well under
    it. A row that lands anyway is found once the incarnation's row is
    gone, since a call without a cursor starts from the incarnation's
    first key. The link table follows by cascade, measured faster than
    deleting link rows manually at one and at four links per segment.
    Cascade deletion saturates around 3M link rows/s as density grows
    (380k segments/s at one link per segment, 46k at 64), so heavily
    linked partitions still purge in sub-second calls. Link fan-out is set
    by the deriver, not something the store can reject after derivation.
  - The incarnation's row goes only in a call whose own deletes found fewer
    rows than its bound, and a call returns whether it found anything, so a
    call that returns False proves the incarnation gone from the store.
    Link rows go with their segments by the foreign key's cascade and
    nothing else: the store requires foreign-key enforcement of its engine
    (on SQLite, per connection) and does not verify it, so an engine
    without enforcement leaves link rows behind.
  - Purgers of one incarnation, from any process, serialize on its row's
    lock (`FOR UPDATE`), so each reads the cursor its predecessor
    committed. On SQLite, which drops locking clauses, the call holds the
    database's write lock from BEGIN instead. With a cursor read before the
    lock is taken, a second purger could plan its batch from a cursor the
    first has since advanced and skip rows.
  - The store never schedules purging; the owner's registry does, through
    its purge rounds, and the server's resource manager runs them: bounded
    rounds a short pause apart while one ran, a full tick apart otherwise;
    failures are logged and retried next tick, and the task is cancelled
    on close.
  - Pacing, not deletion speed, bounds how fast a backlog drains: a call
    deletes at most `purge_max_segments` segment rows and the resource
    manager pauses between busy rounds, so one process drains fewer than
    `purge_max_segments` rows per pause (under 10,000 rows per second at
    the defaults, so upwards of 17 minutes for a ten-million-row tenant).
    Deletion itself measured about 147k rows/s on the benchmark box.

### Fencing

The store checks nothing about whether an incarnation is still live. Its
owner fences each operation by the incarnation's registration (see
[fencing](partition_registry.md#fencing)): an operation begun after the
tenant's deletion raises, and a write that lands under a dead incarnation
is reclaimed by the purge, which starts only after the retention.

### Locking model

Row locks only. Writers lock nothing beyond the rows they write;
`delete_segments` locks its segment rows in key order on PostgreSQL; the
purge takes `FOR UPDATE` on its incarnation's row, which only purgers of
the same incarnation contend for. No table-level lock, no DDL, and no lock
upgrade anywhere in the store, which removes the measured deadlock classes
at the root: reclamation touches only rows of an incarnation no request can
reach. On SQLite, which drops locking clauses, every write transaction of
the store takes the database's write lock at BEGIN (`BEGIN IMMEDIATE`,
through the `enable_sqlite_explicit_begin` setup the store requires of its
engine), which stands in for all of them.

## Alternatives considered

Measured 2026-08-31 (raw asyncpg, 40 tenants x 2000 row-pairs; scaled runs
with a 500k-pair tenant among 50 small tenants; pgvector:pg16, 2 CPUs):

| layout | ingest | seed read | tenant create | tenant delete | catalog cost |
|---|---|---|---|---|---|
| `PARTITION OF` children | 8.7-9.2k pairs/s | 0.43-0.46 ms | ~12 ms DDL | O(1) + detach | ~5 relations/tenant |
| standalone child tables | 8.0-8.4k pairs/s | 0.38-0.41 ms | ~6 ms DDL | O(1) | ~5 relations/tenant |
| shared tables | 15k pairs/s | 0.21 ms | 0.006 ms | O(1) logical | none |

At scale (1M pairs, big-tenant reads): shared 0.41 ms seed / 0.23 ms window
vs per-tenant 0.46 / 0.36. The serving path is point lookups and short
index ranges, which do not suffer from tenants interleaving in a shared
heap. Batched deletion measured faster on the shared layout (147k vs 93k
rows/s). Per-tenant tables win only small-tenant removal latency (9 ms drop
vs 58 ms row delete), which does not justify their catalog cost at the
required tenant counts. Both per-tenant options also keep the RI-trigger
lock fan-out of a parent-level foreign key, or need dialect-conditional
constraints on the shared models to avoid doubled foreign keys.

This is the industry's pool model: O(1) logical deletion via a registry
plus asynchronous batched reclamation. O(1) physical drops exist only in
per-tenant-resource (silo) architectures, which the tenant-count
requirement excludes.

## Consequences

- Removed: partitioned parents, per-tenant DDL, detach machinery, the
  store-wide management lock, per-partition ORM entities and their cache,
  the compiled-statement-cache tuning. Generic plans are benign on
  unpartitioned tables (a plan references one table), so no plan-mode
  tuning is needed.
- One architecture on every dialect. The remaining dialect splits are the
  LATERAL-vs-loop read strategy, the PostgreSQL-only ordered row locks in
  `delete_segments` (SQLAlchemy drops locking clauses on SQLite), and
  SQLite's engine setup: the foreign-key pragma, and transactions begun
  explicitly, a write transaction with `BEGIN IMMEDIATE`. Both are
  per-connection state that the engine's owner registers at engine
  creation. The store requires the pragma and does not verify it, which is
  the reference practice for SQLite foreign keys (SQLAlchemy's dialect
  docs, Django's and Rails' adapters); a listener added later by the store
  would miss connections already in the pool. An engine without the pragma
  leaves link rows that outlive their segments, and nothing reclaims them:
  the purge relies on the same cascade.
- Replication and sharding: three ordinary tables. Logical replication
  covers new tenants automatically (they are rows, not relations), and
  moving a tenant between nodes is an indexed row copy plus the owner
  registering a fresh incarnation, which fences all stale handles.
- Tenant deletion is O(rows) physically, in the background: a deleted
  partition is unreachable at once, and its rows are reclaimed by the
  owner's purge rounds, at the pace their bounds and pause set. A
  deployment must run the purge rounds.
- No migration from the partitioned layout is provided (the event backend
  is opt-in and pre-GA); existing databases recreate their schema.
- Datetime convention: filter nodes normalize datetime values to UTC-aware
  instants at construction (`FilterExpr`'s contract), so every consumer
  sees instants. The Neo4j compiler's `datetime.timestamp()` used to read
  a naive value in the server's local zone; it now parses ISO-string
  filter values and applies the naive-means-UTC rule the rest of the
  codebase uses. The in-memory short-term evaluator tags a naive stored
  metadata datetime the same way at comparison time, since stored user
  data is not rewritten. Storage write paths spell
  `ensure_tz_aware(...).astimezone(UTC)` as two explicit steps at each
  site. A composed `to_utc` helper was rejected: its name pins only the
  conversion, not the naive-means-UTC decision, and it could not prevent
  half-applied normalization anyway, because read paths need the tagging
  step alone (segment reads reapply the stored offset; cluster and episode
  reads only tag naive database values). The per-site repetition is
  deliberate.
