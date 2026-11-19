# Segment store: shared tables keyed by partition UUID

## Problem

On PostgreSQL, giving every partition (tenant) its own pair of child tables
under LIST-partitioned parents, created and dropped by per-tenant DDL,
produces a family of defects, each diagnosed and measured:

- `DROP ... CASCADE` of one tenant's child tables removes the store-wide
  parent-level foreign key, ending referential integrity for every other
  tenant (#1544).
- Any statement with the partition key as a bind parameter switches to a
  cached generic plan after five executions, and a generic plan locks every
  child partition per execution: 9 locks per read with a custom plan, 319
  with the generic plan at 316 partitions, and HTTP 500 `out of shared
  memory` under 128-way concurrency (#1546).
- PostgreSQL's foreign-key integrity triggers address the partitioned
  parents internally, so writes and deletes keep a per-backend lock spike
  proportional to partition count even with custom plans for the client
  statements (10 -> 166 -> 10 locks across the generic-plan attempt).
- Create and delete DDL against the shared parents deadlocks with writers
  and with each other, in every variant of the layout measured.
- Each workaround adds machinery: detach-before-drop, a store-wide
  management lock, per-partition ORM entities with a memoized cache,
  statement-cache tuning.

Two scaling requirements rule out every per-tenant-table layout:
tenant creation must stay cheap up to hundreds of thousands to millions of
tenants (a per-tenant table costs about 5 catalog relations plus disk
files, which PostgreSQL does not survive at those counts), and a single
tenant may hold tens of thousands to tens of millions of rows.

## Design

One physical schema on every dialect; the ORM models are the tables.

The store holds no tenant names. Its owner's [partition
registry](partition_registry.md) names each tenant's partition and mints its
`partition_uuid` (a random UUID), a new one each time the name is created
again; the store addresses everything by partition UUID.

- `segment_store_pt`: one row per created partition, holding its payload
  codec config and, while the partition is purged, the purge's cursor, the
  last segment key purged.
- `segment_store_sg` and `segment_store_dv_ln`, the segment and
  derivative-link tables, shared by all tenants and not partitioned. Rows
  carry no tenant name; they are keyed by the partition's UUID alone, so
  addressing the wrong tenant is structurally impossible rather than
  guarded against. A 16-byte UUID also keeps index entries narrower than a
  composite string key, and random UUIDs are unique across nodes without
  coordination, so a tenant's rows move between databases verbatim.

The link table keeps its foreign key to the segment table (`ON DELETE
CASCADE`), both sides keyed by partition UUID. The segment table has no
foreign key to the partition's row: the purge deletes a partition's
segments in batches, and the drop deletes its row once the purge finds
none.

A name with a UUID minted for each object created under it is the pattern
of a Kubernetes object's name and UID, and of a Kafka topic's name and topic
ID: an object deleted and created again under the same name is a new object
with a new UUID.

### Tenant lifecycle

- **Create**: insert the partition's row with the store's codec config. No
  DDL, no lock beyond the row. Creating again with the same config changes
  nothing; a different config raises the config-mismatch error.
- **Get**: read the partition's row. The handle captures the partition's
  UUID and the codec.
- **Delete** is the owner's: deleting the tenant's name in its registry is
  O(1) regardless of tenant size, and the tenant is unreachable at once. The
  store is not called.
- **Re-create**: the owner creates a new partition under the name, under a
  fresh UUID its registry mints, safe at any time; the old partition's rows
  are invisible to the new partition even while the purge sweeps them.
- **Purge and drop**: `purge_partition(partition_uuid) -> bool` and
  `drop_partition(partition_uuid)`, the `PartitionedStore` steps. The
  owner's partition manager calls the purge for a deleted partition once
  the retention has passed, round after round, and calls the drop, which
  deletes the partition's row, once no store's purge finds records. How much
  one call does is the store's choice. This store bounds each call so that
  it does not noticeably degrade concurrent request serving, and every bound
  below exists for that. The deletion contract is "unreachable immediately,
  erased asynchronously".

  - One transaction per call, committing its progress or nothing, so
    committed progress survives interruption.
  - Segment rows are deleted in batches of up to
    `SQLAlchemySegmentStoreParams.purge_max_segments` per call. A batch
    continues from the partition's cursor: it reads the next keys after
    `purged_through` in primary-key order, deletes that key range, and,
    when it found a full batch, records its last key in the same
    transaction. PostgreSQL keeps deleted rows in the table and its
    indexes until vacuum, and a batch that started from the partition's
    first key would step over all of them, making each batch cost in
    proportion to what was already purged; a batch that continues from
    the cursor reads no row an earlier batch deleted. A walk from the
    cursor that reaches the end clears it, and only a call that searches
    from the partition's first key answers that nothing remains, so a row
    that lands behind the cursor is still found. A search from the first
    key steps over the deleted rows vacuum has not yet removed, so the
    call that finds the partition empty costs in proportion to them. Link
    rows are deleted by cascade from their segments, measured faster than
    deleting them manually at one and at four links per segment. A call's
    transaction therefore also scales with the links per segment, which
    the deriver sets and the store cannot reject after derivation.
  - A call returns False only when it searched from the partition's first
    key and found nothing, so a call that returns False proves the
    partition's segments gone from the store. Link rows are deleted with
    their segments by the foreign key's cascade and by nothing else: the
    store requires foreign-key enforcement of its engine (on SQLite, per
    connection) and does not verify it, so an engine without enforcement
    leaves link rows behind.
  - Purgers of one partition, from any process, serialize on its row's
    lock (`FOR UPDATE`) while the row exists, so each reads the cursor its
    predecessor committed. On SQLite, which drops locking clauses, the call
    holds the database's write lock from BEGIN instead. With a cursor read
    before the lock is taken, a second purger could plan its batch from a
    cursor the first has since advanced and read again the rows the first
    deleted.
  - The store never schedules purging; its owner's purge rounds call it,
    and the server's resource manager runs those rounds (see
    [the sweeper](vector_store_purge.md#the-sweeper)).
  - Pacing, not deletion speed, bounds how fast a backlog drains: a call
    deletes at most `purge_max_segments` segment rows and the resource
    manager pauses between busy rounds, so one process drains at most
    `purge_max_segments` rows per pause.

### Fencing

The store checks nothing about whether a partition is still live. Its
owner fences each operation by the partition's registration (see
[fencing](partition_registry.md#fencing)): an operation begun after the
tenant's deletion raises, and a write that lands in a deleted partition is
reclaimed by the purge, which starts only after the retention.

### Locking model

Row locks only. Writers lock nothing beyond the rows they write;
`delete_segments` deletes its segments one statement each in key order, so
every deletion takes its row locks in that order; the purge takes
`FOR UPDATE` on its partition's row, which only the purgers and the drop of
the same partition contend for. No table-level lock, no DDL,
and no lock upgrade anywhere in the store, which removes the measured
deadlock classes at the root: reclamation touches only rows of a partition
no request can reach. On SQLite, which drops locking clauses, every write
transaction of the store takes the database's write lock at BEGIN
(`BEGIN IMMEDIATE`, through the `enable_sqlite_explicit_begin` setup the
store requires of its engine), which stands in for the row locks.

### Lookups by key

A statement that reads a partition's rows for a list of keys (seed segments
by UUID, segments by event UUID, links by segment UUID or by derivative
UUID) costs in proportion to the number of keys on PostgreSQL, whatever the
partition's size. It is one statement: the keys are one array parameter,
and each joins a `LATERAL` subquery selecting that key's rows, so the index
is searched once per key, as an `IN` list searches it under a plan built
for the partition. The subquery's `OFFSET 0` keeps the planner from merging
it into the statement; it is a constant, which PostgreSQL plans no node
for. `delete_segments` runs one `DELETE` per segment on every dialect.

Given the keys as one `IN` list, PostgreSQL chooses between an index lookup
per key and one scan of the partition's whole range of an index led by the
partition UUID, filtered by the list. It prices that scan by the number of
rows it estimates the partition to hold. It has that number only for the
partitions among the column's most common values (100 at the default
statistics target), and only in a plan built for the statement's parameter
values. For any other partition, and in the generic plan a prepared
statement switches to after five executions, it uses the average over all
partitions. Where most tenants are small, it prices the scan below the
lookups once the list holds more than a few keys, and on a large tenant the
scan reads every row the tenant has.

Measured on PostgreSQL 18.6 (0.6 to 6 million rows, statistics from
`ANALYZE`, data in memory; times are the server's mean execution time over
1,500 executions):

- With one 100,000-segment tenant among 500,000 tenants of one segment
  each, the generic plan for the segments of 40 events read 100,748 blocks
  in 61 ms; the statement above read 280 blocks in 0.09 ms, as the `IN`
  list did under a plan built for the tenant. The seed read did the same
  (101,471 blocks in 45 ms against 160 in 0.06 ms), and the link reads read
  a tenant whole in a like distribution (about 100,700 blocks).
- With one 1,000,000-segment tenant among 20,000 tenants of 48, the generic
  plan read 14,713 blocks against 20 for the segments of 5 events, and
  16,326 against 2,000 for 500 seed segments.
- A plan built for the parameter values chose the scan as well for a large
  tenant outside the most common values (4,973 blocks against 80 for 20
  events of a 5,000-segment tenant, at `random_page_cost` 1.1), so
  `plan_cache_mode = force_custom_plan` corrects only the listed tenants.
- The cost of the scan grows in proportion to the tenant, and the cost of
  the statement above with the number of keys alone: 2 to 3 µs of
  execution per key.

For one key, the index on the partition UUID and the key answers in one
lookup at any estimate, and no scan of the partition's range is priced
below it, so the plan depends on neither the tenant's size, the plan mode,
nor the statistics. It held in every case measured on PostgreSQL 18: seven
distributions of tenant sizes, 1 to 500 keys, plans built for the values
and generic plans, with and without statistics, at `random_page_cost` 4 and
1.1. It held for three of the distributions on PostgreSQL 16 as well. The
statement's text is also the same for any number of keys, so one prepared
statement serves them all.

Against an `IN` list under a plan built for the partition, the statement
reads the same blocks up to 100 keys spread over a large tenant and about a
third more at 500, in the same time to a third more. It costs more where
the list's one scan passes the keys together: up to about two and a half
times the time where the keys are neighbors in the index, and, for a tenant
of 48 segments, whose whole range is one page, 44 µs against 17 for 40 keys
and 0.3 ms against 0.03 for 500.

Rewrites that keep the keys as one list (`= ANY` of an array, a plain join
to the unnested array) leave the planner the same choice, and it chose the
scan for large tenants there too. SQLite chooses the index on both columns
for an `IN` list of any length measured, with and without statistics, so it
keeps the list.

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

- No partitioned parents, per-tenant DDL, detach machinery, store-wide
  management lock, per-partition ORM entities or their cache, or
  compiled-statement-cache tuning. A generic plan references one table,
  so its locks do not grow with the tenant count. It prices a statement
  for an average tenant, so the statements whose best plan depends on the
  tenant's size take a form with one plan (see [lookups by
  key](#lookups-by-key)), and no plan-mode tuning is needed.
- One architecture on every dialect. The remaining dialect splits are the
  LATERAL-vs-loop read strategy, the lookup per key on PostgreSQL, and
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
  moving a tenant's rows between nodes is an indexed row copy: they keep
  the partition UUID its registry holds.
- Tenant deletion is O(rows) physically, in the background: a deleted
  partition is unreachable at once, and its rows are reclaimed by the
  registry's purge rounds, at the pace their bounds and pause set. A
  deployment must run the purge rounds.
- No schema migration is provided; existing databases recreate their
  schema.
- Datetime convention: filter nodes normalize datetime values to UTC-aware
  instants at construction (`FilterExpr`'s contract), so every consumer
  sees instants. The Neo4j compiler parses ISO-string filter values and
  applies the naive-means-UTC rule the rest of the
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
