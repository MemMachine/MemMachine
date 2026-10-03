# Vector store: tombstones and purge

Part of [vector store horizontal scaling](vector_store_horizontal_scaling.md).

## Problem

Deleting a partition of a Qdrant or Milvus store with one filter-delete,
issued when the partition is deleted, has three defects once more than one
process serves a backend:

- A write in flight during the deletion, from a handle in another process,
  lands after it and outlives it.
- One delete of a large tenant is one burst of work the backend applies at
  once, stalling other tenants (measured below).
- Nothing records that a deletion is incomplete, so a crash part way through
  leaves records no one will ever reclaim.

## Design

Deletion is logical and immediate; reclamation is physical, deferred, bounded
and retried.

### Tombstones

`unregister` removes the partition's row and queues the incarnation's
*tombstone* in one transaction (see [partition
registry](vector_store_partition_registry.md)). The partition is unreachable
when it commits. Its records stay in the backend until purge rounds reclaim
them.

A tombstone becomes *due* once `tombstone_retention_seconds` (per store;
86,400, one day, unless configured) has passed since its deletion.

- **Why wait.** A write that passed its liveness check before the deletion
  committed can land after it, and the backend cannot refuse it. Every such
  write has landed once the longest a request can be in flight has passed;
  `request_timeout_seconds` (30 unless configured) bounds the client's part.
- **What a round sees.** A round lists the incarnation's records with the
  store's own reads, which may lag writes by a delay the store states (see
  [consistency](vector_store_consistency.md)): at most `common.gracefulTime`
  (5 s by default) on Milvus at Bounded. The retention exceeds that delay too,
  so a round that runs after it and finds nothing proves the incarnation empty. A round that lists
  records an earlier round deleted, before the deletion is reflected, deletes
  them again: a repeated round, never a wrong result.
- **The retention decides nothing about validity.** A stale write is refused
  by the handle's check after it; the retention only has to outlast any write
  in flight. A write that lands after its tombstone is gone stays under a dead
  incarnation no partition reads: leaked storage, never a wrong result.
- **The configuration enforces a floor of 10 × `request_timeout_seconds` +
  300 s.** The request timeout is the only part of a write's time in flight
  the store knows. Neither Qdrant nor Milvus bounds how long a received write
  can wait before it is applied (their queues are bounded by count), both can
  apply a write after its client has given up, and Milvus's read delay
  (`common.gracefulTime`) is server configuration a client cannot read. These
  take seconds to minutes when no process is paused; the default retention, a
  day, also covers a process paused between its liveness check and its write.
- **Only the database's clock is used.** The queue stores the deletion's time,
  `enqueued_at`, written by the database's `now()`. The retention is applied
  by the database's own arithmetic when a claim is decided (`now() - interval`
  on PostgreSQL; `datetime('now', '-N seconds')` on SQLite, which yields the
  same text form as `CURRENT_TIMESTAMP` so stamps and cutoff compare in time
  order). A changed retention therefore reaches every tombstone already
  queued, and no client clock enters a decision.
- An incarnation is never re-minted while its tombstone exists, so no new
  partition can adopt, or have reclaimed out from under it, a dead life's
  records.

### The purge round

`VectorStore.purge_deleted_partitions()` is part of the contract: each call
does a bounded amount of work and returns whether it ran a round, `False` once
nothing is due, so a caller drains the queue by calling it until `False`. On
Qdrant and Milvus a call is one round on the tombstone that came due first;
both SQLite stores, whose deletion reclaims physically, return `False`.

`run_purge_round()` runs one round:

1. **Claim.** One range on the `(vector_store_name, enqueued_at)` index: the
   oldest due tombstone that is neither backing off nor dead-lettered,
   `LIMIT 1`. On PostgreSQL it is selected `FOR UPDATE SKIP LOCKED`: a
   concurrent purger skips a locked tombstone and takes the next, so purgers
   on every process split a backlog without coordinating. SQLite has no row
   locks, so there the claim is an `UPDATE` of the tombstone's row, which
   opens SQLite's write transaction: purgers serialize at the claim, and
   rounds run one at a time. SQLite's write lock covers the whole database
   file and is then held across the round's remote calls, each bounded by the
   store's request timeout (30 s by default). Every writer to that database
   waits for it: the registry's, and those of every store sharing the
   database, as the episode store, session manager, segment store and
   configuration database do under the configuration wizard's defaults. Past
   the driver's busy timeout (SQLite's 5 s default, which the server does not
   change) they fail with a locked-database error.
2. **Round.** The registry calls the store's round with the tombstone's
   incarnation. The round looks for records under the incarnation in the
   store's native collection, deletes what it finds (per backend, below), and
   returns whether it found any.
3. **Record.** In the claim's transaction: a round that found nothing removes
   the tombstone, which frees the incarnation; a round that found records keeps
   it due and clears its failed rounds.

The claim's transaction stays open for the whole round, holding the
tombstone's row lock, and sits idle on PostgreSQL while the backend deletes. A
deployment that sets PostgreSQL's `idle_in_transaction_session_timeout` (off by
default) must set it above a round's duration. A shorter one ends the claim's
session mid-round: the round fails, though its deletions in the backend stand,
and a tombstone whose rounds keep outlasting the timeout is dead-lettered after
10. The measured rounds take about 100 ms per Milvus batch and, on Qdrant,
whose single filter-delete makes the longest round, about 1.3 s per million
points (see the per-backend documents).

### Failed rounds: backoff and dead-lettering

A round that raises rolls back, then counts against its tombstone in a
transaction of its own: `failed_rounds + 1`, and `last_failed_at = now()` on
the database clock.

- **Backoff.** After its f-th consecutive failure, a tombstone is claimed
  again once `min(purge_retry_backoff_seconds * 2^(f-1),
  max_purge_retry_backoff_seconds)` has passed since `last_failed_at`: 30 s,
  60 s, 120 s and so on, at most 1 h. The tombstones behind it are claimed
  meanwhile, so one failing tombstone does not hold the queue.
- **Dead-lettering.** After 10 consecutive failures, about 3 hours of retries,
  the tombstone is dead-lettered: kept, its incarnation reserved, skipped by
  claims, and reported by an error log naming the incarnation, the last error, and the
  table. Setting its `failed_rounds` back to 0 returns it to the purge.
- The backoff is computed from recorded facts (the count and the time of the
  last failure), not stored as a time to retry at. Both durations are registry
  parameters in seconds with those defaults, so they can become configuration
  without changing the schema.

A dead-letter bound, rather than retrying forever, makes a tombstone that
never purges a visible problem instead of garbage that is quietly retried.

### Per-backend rounds

Each backend deletes the way it measured best: Qdrant with one filter-delete
of the whole incarnation per round, Milvus in bounded batches listed by the
incarnation. The measurements are in the [Qdrant](qdrant_vector_store.md) and
[Milvus](milvus_vector_store.md) documents. A native collection that is gone
holds nothing: the round finds nothing, and the tombstone goes.

### The sweeper

The store never schedules its own purge. The resource manager starts one
sweeper task per vector store the first time it hands the store out, and
`close()` cancels them. A sweeper calls `purge_deleted_partitions()` again
after 1 s when it ran a round and after 60 s when nothing was due; a round that
raises is logged and retried a tick later. Sweepers on other
processes need no coordination: the claim arbitrates.

### Measured cost of the claim

- Backoff (PostgreSQL 18.6 and SQLite 3.50.4; 1,000,000 tombstones not yet
  due; median of 30 claims): the claim reads each due tombstone that is
  backing off and none that is not yet due. It took 0.3 / 1.3 / 11 ms on
  PostgreSQL and 0.3 / 2.1 / 21 ms on SQLite with 1k / 10k / 100k tombstones
  backing off, and 0.15-0.3 ms with none.
- Interference (PostgreSQL 16, the claim's earlier two-statement form; 20,000
  live registry rows and 20,000 tombstones): with two sweepers running rounds
  back to back, about 78 per second, beside 16 interactive workers,
  interactive throughput and the p99 of the handles' liveness lookup were
  unchanged within run-to-run noise on PostgreSQL. On SQLite, where the
  sweepers write in the same process, throughput dropped 2.5-14% at that
  rate, as much as with sweepers
  that only commit a one-row write per round, and not measurably at the
  resource manager's pace.

## Alternatives considered

- **Delete immediately, by filter.** Rejected for the
  three defects above.
- **Order the claim by failures first**, sinking a failing tombstone behind
  untried ones. Rejected: it retains a failing tombstone's records indefinitely
  while the queue has other work, and hides the failure. The backoff gives the
  tombstones behind it their turn without reordering the queue.
- **A `retry_at` column.** Rejected in favor of computing from recorded facts,
  so a changed backoff reaches tombstones already failing.
- **Count the backoff from when the tombstone came due**, with no new column.
  Measured: a tombstone due for days retries at once, its backoff spent before
  it first fails, so after an outage it fails as fast as before.
- **An index on `(enqueued_at, failed_rounds)`.** It helps only skipping
  dead-lettered tombstones, so the claim keeps `(enqueued_at)`.
- **A separate dead-letter table.** A counter on the queue row does the same
  with no move between tables.
- **Batched deletes on Qdrant; one filter-delete on Milvus; listing Milvus
  keys by primary-key range.** Measured worse; see the per-backend documents.
- **A stronger read level for the purge than for the store's other reads**
  (Strong on Milvus). It would let a round see the round before it, sparing a
  repeated listing, but the design needs no more than the retention already
  guarantees.

## Consequences

- Deleted records stay in the backend at least the retention: storage is
  reclaimed a day after deletion by default.
- An operator watches for the dead-letter error log. A dead-lettered
  tombstone's records stay until someone resets its `failed_rounds`.
- The purge loads the backend in bounded rounds from every process's sweeper;
  on PostgreSQL the rounds of different tombstones proceed in parallel.
