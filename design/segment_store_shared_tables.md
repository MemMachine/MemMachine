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

- `segment_store_pt`, the tenant registry: one row per live tenant, holding
  the logical partition key (primary key), an `incarnation` (a random UUID
  minted at creation, unique-constrained), and the payload codec config.
- `segment_store_sg` and `segment_store_dv_ln`, the segment and
  derivative-link tables, shared by all tenants and not partitioned. Rows
  carry no logical key; they are keyed by the incarnation alone, so a data
  query cannot be built without resolving the registry first, and
  addressing the wrong tenant is structurally impossible rather than
  guarded against. A 16-byte UUID also keeps index entries narrower than a
  composite string key, and random UUIDs are unique across nodes without
  coordination, so a tenant's rows move between databases verbatim.
- `segment_store_gc`, the purge queue: one row per dead incarnation, with
  the logical key kept for forensics.

The link table keeps its foreign key to the segment table (`ON DELETE
CASCADE`), both sides keyed by incarnation. The segment table has no
foreign key to the registry: registry rows and data rows are decoupled so
that registry deletion is O(1), and the rows left behind are exactly what
the purge queue tracks.

An incarnation value cannot be reused while any trace of it remains. A
collision with a live incarnation violates the registry's unique
constraint. A collision with one whose garbage is still awaiting purge is
caught by re-checking the purge queue inside the mint transaction, after
the registry insert: a concurrent deletion moving the colliding row to the
queue is visible by the time the insert's unique-index wait resolves, and
no new queue entry for the minted value can appear before commit, because
the only registry row carrying it is the uncommitted one. The queue's
primary key rejects double enqueueing. Across databases, uniqueness rests
on random-UUID collision resistance.

"Incarnation" is the established term for a per-lifetime random identifier
(cf. Kafka's broker incarnation ID). It is deliberately not "generation" or
"epoch", which would imply ordered, comparable values.

### Tenant lifecycle

- **Create**: validate the logical key, mint an incarnation, insert the
  registry row and re-check the purge queue in one transaction. No DDL, no
  management lock, microseconds. A committed row under the key raises the
  partition-exists error. Any other integrity rejection, or a minted
  incarnation found in the purge queue, is retried with a fresh incarnation
  up to `_MAX_MINT_ATTEMPTS`; a persistent failure raises
  `SegmentStoreAttemptsExhaustedError` with the database error chained.
- **Open**: read the registry row. The handle captures the logical key, the
  incarnation, and the codec.
- **Delete**: one transaction: lock the registry row (`SELECT ... FOR
  UPDATE`, which waits out in-flight writers holding `FOR SHARE` pins),
  insert the incarnation into the purge queue, delete the registry row.
  O(1) regardless of tenant size, and the tenant is unreachable at once
  because every operation resolves the registry first.
- **Re-create**: a new registry row with a fresh incarnation, safe at any
  time; the old incarnation's rows are invisible to the successor even
  while the purger is still sweeping them.
- **Purge**: `purge_deleted_partitions() -> bool` on the `SegmentStore`
  ABC, the sweeper: it claims the oldest entry across all keys, returns
  True while more work may remain, and the caller's whole protocol is
  "call until False"; how much one call does is implementation policy. The caller is promised only that a call
  does not noticeably degrade concurrent request serving, and every bound
  below exists to keep that promise. The deletion contract is "unreachable
  immediately, erased asynchronously".

  - One transaction per call, committing its progress or nothing, so
    committed progress survives interruption.
  - Segment rows are deleted in batches (`uuid IN (SELECT ... LIMIT n)`,
    portable across dialects) up to
    `SQLAlchemySegmentStoreParams.purge_max_segments` per call. The link
    table follows by cascade, measured faster than deleting link rows
    manually at one and at four links per segment. Cascade deletion
    saturates around 3M link rows/s as density grows (380k segments/s at
    one link per segment, 46k at 64), so heavily linked partitions still
    purge in sub-second calls. Link fan-out is set by the deriver, not
    something the store can reject after derivation.
  - Queue entries carry their own bound,
    `SQLAlchemySegmentStoreParams.purge_max_partitions`, because their cost
    is round trips rather than rows: empty partitions are cheap to create
    and delete, and a backlog of them must not turn one call into an
    unbounded transaction. Both bounds are set once at construction; the
    server currently uses the defaults, and wiring them into server
    configuration is future work.
  - Entries are claimed oldest-first (enqueue time stamped by the database
    clock, so every server's entries order on one clock, at its
    resolution) with `FOR UPDATE
    SKIP LOCKED`, one at a time, so a call neither materializes nor locks
    the rest of the backlog. Concurrent purgers, from any process, share
    the queue instead of contending, and only the claiming call touches a
    dead incarnation's rows (writers cannot, since the fence pins live
    incarnations only), so reclamation is deadlock-free by construction.
  - The delete path does not purge. An inline drain was tried twice,
    first of the global queue and then scoped to the deleted key, and
    removed: prompt physical erasure is not a promise the store can keep
    on every dialect (on SQLite any writer past the busy timeout fails),
    and a global drain made one deletion pay for other tenants' backlog.
    A deleted partition is unreachable at once and its rows are reclaimed
    within the sweeper's interval. Waking the sweeper from the delete
    path is future work with the tenant lifecycle layer.
  - An entry is retired only when the retiring call's own deletes found
    fewer rows than its remaining budget. Before retiring, the call also
    deletes any link rows still carrying the incarnation, in batches drawn
    from the same budget and logged as a warning: none when the cascade
    is enforced; an engine without foreign-key enforcement leaves them
    behind. A link row deletes about 3x cheaper than a segment
    row (1.0 vs 3.3 us per row, batched, on the benchmark box), so one
    budget calibrated on segment rows bounds the call without assuming a
    ratio; widening or indexing the link table revisits this.
  - On SQLite, which drops locking clauses and defers BEGIN to the first
    DML, the claim is a plain read and purgers serialize on the database
    write lock at the DELETE. Two purgers may claim the same entry; the
    second deletes whatever the first left (nothing, once the first retired
    it) and re-retires it. Duplicated round trips, never duplicated or
    missed reclamation.
  - The store never schedules purging; when and how often is the caller's
    policy, and implementations whose deletes reclaim physically return
    False. In the server, the resource manager runs one background task
    per store: bounded calls a short pause apart while the store reports
    more work (the pause yields the database to request serving), a full
    tick apart otherwise; failures are logged and retried next tick, and
    the task is cancelled on close. `LongTermMemory` does not purge on
    session drop; a deployment must run the sweeper somewhere, and the
    ABC says so.
  - Measured deletion throughput is about 147k rows/s on the benchmark
    box, so a ten-million-row tenant erases in about a minute of
    background work.

### Fencing (resolves #1549)

Every write pins the registry row with `SELECT ... WHERE incarnation =
:incarnation FOR SHARE` and raises a stale-handle error when no row
matches. Reads add the same predicate to their data statement as an
`EXISTS`, so one statement (one snapshot) checks liveness and reads: a
stale handle reads nothing, at no extra round trip, and a read that returns
no rows issues the registry check on its own to tell an empty partition
from a stale handle. Windowed reads span several statements, each with its
own snapshot, so they end with one registry read; a deletion committing
between their statements raises instead of returning seeds with empty
context.

On SQLite the driver defers BEGIN until the first write, so a SELECT-only
fence would run outside the write transaction. The proper primitive, BEGIN
IMMEDIATE, means taking over transaction management for the whole engine in
SQLAlchemy (`isolation_level=None` plus a begin-event hook), which the store
cannot do to a caller-owned, possibly shared engine. The write fence is
therefore a self-checking UPDATE of the registry row: it takes the same
write lock, scoped to the transaction, and its match count is the staleness
check. Deletion opens its transaction the same way, so racing deletions
serialize.

A handle held across delete, or across delete and re-create, fails loudly
instead of operating on the successor tenant, on every dialect. SQLite
previously had no mechanism for this at all.

### Locking model

Row locks only. Writers hold `FOR SHARE` on their registry row for the
write transaction; deletion takes `FOR UPDATE` on the same row; the purger
claims queue rows with `FOR UPDATE SKIP LOCKED` and never waits; the mint's
queue re-check takes `FOR SHARE` on a queue row only in the collision case.
No table-level lock, no DDL, and no lock upgrade anywhere in the store,
which removes the measured deadlock classes at the root: writers and
lifecycle operations share no lockable object except one tenant's registry
row, where blocking is the intended semantics, and reclamation touches only
rows no other operation can reach.

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
  SQLite's foreign-key pragma. The pragma is per-connection state that the
  engine's owner registers at engine creation. The store requires it and
  does not verify it, which is the reference practice for SQLite foreign
  keys (SQLAlchemy's dialect docs, Django's and Rails' adapters); a
  listener added later by the store would miss connections already in the
  pool. An engine without the pragma leaves link rows that outlive their
  segments until the partition is dropped, when the purge reclaims them
  with a warning.
- Replication and sharding: four ordinary tables. Logical replication
  covers new tenants automatically (they are rows, not relations), and
  moving a tenant between nodes is an indexed row copy plus a registry
  insert whose fresh incarnation fences all stale handles.
- Tenant deletion is O(rows) physically, in the background: a deleted
  partition is unreachable at once, and its rows are reclaimed by the
  sweeper within its interval. A deployment must run the sweeper.
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
