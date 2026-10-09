# Vector store: tombstones and purge

Part of [vector store horizontal scaling](vector_store_horizontal_scaling.md).

## Problem

Deleting a Qdrant or Milvus collection with one filter-delete by name, issued
when the collection is deleted, has three defects once more than one process
serves a backend:

- A write in flight during the deletion, from a handle in another process,
  lands after it and outlives it.
- One delete of a large tenant is one burst of work the backend applies at
  once, stalling other tenants (measured below).
- Nothing records that a deletion is incomplete, so a crash part way through
  leaves records no one will ever reclaim.

## Design

Deletion is logical and immediate; reclamation is physical, deferred, bounded,
and retried.

### Tombstones

`unregister` removes the collection's row and queues the incarnation's
*tombstone* in one transaction (see [collection
registry](vector_store_collection_registry.md)). The collection is unreachable
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
  incarnation no collection reads: leaked storage, never a wrong result.
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
  collection can adopt, or have reclaimed out from under it, a dead life's
  records.

### The purge round

`VectorStore.purge_deleted_collections()` is part of the contract: each call
does a bounded amount of work and returns whether it ran a round, `False` once
nothing is due, so a caller drains the queue by calling it until `False`. On
Qdrant and Milvus a call is one round on the tombstone that came due first;
both SQLite stores, whose deletion reclaims physically, return `False`.

`run_purge_round()` runs one round:

1. **Claim.** One `UPDATE ... RETURNING`, committed at once, takes the oldest
   due tombstone that is neither backing off, dead-lettered, nor held by another
   claim: it counts an attempt without progress, stamps the tombstone's
   `claimed_at` with the database's `now()`, and increments its
   `claim_generation`. The tombstone is picked by one range on the
   `(vector_store_name, enqueued_at)` index, `LIMIT 1`. On PostgreSQL the pick
   is `FOR UPDATE SKIP LOCKED`, so a concurrent claim skips a row another claim
   is taking and purgers on every process split a backlog without coordinating.
   SQLite drops the locking clause; there the claim's write serializes claims,
   for one statement.
2. **Round.** The registry calls the store's round with the tombstone's
   namespace, configuration, and incarnation, with no transaction open. The
   round looks for records under the incarnation where the namespace and
   configuration locate them, deletes what it finds (per backend, below), and
   returns whether it found any.
3. **Record.** In a short transaction: a round that found nothing removes the
   tombstone, which frees the incarnation; a round that found records ends its
   claim, keeping the tombstone due, and resets its attempts without
   progress.

### The lease

A claim holds its tombstone until its round ends, or until
`purge_lease_seconds` (a registry parameter; 300, five minutes, by default)
has passed since `claimed_at` on the database clock. No transaction is open
while the round runs, so no lock is held across its remote calls. On SQLite,
whose write lock covers the whole database file, every store sharing the
database can write during a round. On PostgreSQL, no session sits idle in a
transaction, holding back vacuum or meeting
`idle_in_transaction_session_timeout`.

- **The lease spreads work; correctness does not rest on it.** A round is safe
  to repeat and to run on two purgers at once, and a round that finds the
  incarnation empty after the retention has finished its purge, under any
  claim. The lease lets purgers split a backlog, each taking a different
  tombstone, where without it they would all take the oldest. A repeated round
  costs Qdrant a scroll and a filter-delete that matches nothing; on Milvus it
  repeats a listing and deletes keys already deleted.
- **A cancelled round takes back its attempt; a round whose purger died
  keeps it.** A cancelled round ends its claim and takes back the attempt its
  claim counted, in a write shielded from the cancellation, so its tombstone
  is claimable at once. A round whose purger died writes nothing after its
  claim, which already counted the attempt; its tombstone is claimed again
  once the lease, and then the backoff, have passed.
- **The lease should exceed a round.** A round's remote calls run one after
  another, each bounded by the store's request timeout. A round that outlasts
  its lease may run beside a later round on the same tombstone, which is
  harmless but repeats work. When it ends, it logs a warning naming the
  incarnation and the lease. The measured rounds take about 100 ms per Milvus
  batch and, on Qdrant, whose single filter-delete makes the longest round,
  about 1.3 s per million points (see the per-backend documents).
- **The generation fences the claim's own writes.** A round ends its claim,
  dates its failure, or resets the attempts, only while the tombstone's
  `claim_generation` is still its claim's. So a round that outlasted its lease
  neither ends the claim taken after it, which would let a third purger in
  during that claim's round, nor records an outcome against it. A round that
  found nothing removes the tombstone under any claim: what it found holds
  whoever found it. The fence covers the registry's writes only: the backend
  cannot condition a delete on a value held in the registry's database, so a
  round that outlasted its lease still sends its deletes, which reach only its
  own dead incarnation.
- **The lease runs on the database's clock.** `claimed_at` is the database's
  `now()`, and the lease is applied by the database's arithmetic when a claim
  is decided, as the retention and the backoff are, so a changed lease reaches
  claims already held. The generation is assigned by the database too; a
  purger supplies neither a time nor an identity.

### Failed rounds: backoff and dead-lettering

Each claim counts an attempt, `attempts_without_progress + 1`. A round that
finds records and deletes them made progress, and resets the count to 0. So
every round that does not finish uses an attempt, whether it raised or its
purger died, as job queues count attempts when work is taken. A round that
raises ends its claim, while the claim is the latest, and sets
`last_failed_at = now()` on the database clock; its error carries a note naming
the incarnation and the attempt. A round that never ended writes nothing.

- **Backoff.** After its a-th attempt fails, a tombstone is claimed again once
  `min(base_purge_retry_backoff_seconds * 2^(a-1),
  max_purge_retry_backoff_seconds)` has passed since the failure: since
  `last_failed_at` for a round that raised, and since its lease passed for one
  that never ended. That is 30 s, 60 s, 120 s, and so on, at most 1 h. The
  tombstones behind it are claimed meanwhile, so one failing tombstone does
  not hold the queue.
- **Retries are logged.** A claim of a second or later attempt logs a warning
  naming the incarnation and the attempt ("attempt 3 of 10"). Each log line
  is about the event at its time: nothing is logged later about an attempt
  that never ended.
- **Dead-lettering.** After 10 attempts without progress, about 3 hours of
  retries, claims skip the tombstone: it is kept, and its incarnation reserved.
  A last attempt that raises is reported by an error log naming the incarnation,
  the last error, and the table. A last attempt whose purger died is not, since
  nothing runs on the tombstone after it; its claim's warning, "attempt 10 of
  10", is the last line about it. Setting its `attempts_without_progress` back
  to 0 returns it to the purge at once.
- The backoff is computed from recorded facts (the attempts, and when the last
  failure raised or its claim was taken), not stored as a time to retry at. Both
  durations are registry parameters in seconds with those defaults, so they can
  become configuration without changing the schema.

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
`close()` cancels them. A sweeper calls `purge_deleted_collections()` again
after 1 s when it ran a round and after 60 s when nothing was due; a round that
raises is logged and retried a tick later. Sweepers on other
processes need no coordination: the claim arbitrates.

### Measured cost of the claim

Measured 2026-10-06 on AC power, against PostgreSQL in a container capped at
2 CPUs and SQLite in a file.

- Backoff (commit 96a8b5bf6's claim; PostgreSQL 18.6 and SQLite 3.50.4;
  1,000,000 tombstones not yet due; median of 50 calls): a call that claims
  nothing reads each due tombstone that is backing off and none that is not
  yet due. It took 1.5 / 2.9 / 20 ms on PostgreSQL and 1.0 / 4.5 / 40 ms on
  SQLite with 1k / 10k / 100k tombstones backing off, and 1.3 ms on PostgreSQL
  and 0.7 ms on SQLite with none due.
- Interference (commit 503687c35's lease, whose rounds cost the same database
  time as commit 96a8b5bf6's; 20,000 live collections and 20,000 due
  tombstones; 16 interactive workers checking handles' liveness beside two
  sweepers, three phases each): on PostgreSQL 16, with rounds back to back
  (about 520 per second) or of 20 ms (about 70 per second), interactive
  throughput and the p99 of the liveness lookup were unchanged within
  run-to-run noise. On SQLite, where the sweepers write in the same process,
  20 ms rounds (about 60 per second) left throughput unchanged and raised the
  p99 from 6-8 ms to 8-9 ms; rounds back to back (about 110 per second)
  lowered throughput about 7% and raised the p99 to 13-15 ms. The resource
  manager's sweepers run at most one round a second.

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
- **An index on `(enqueued_at, attempts_without_progress)`.** It helps only
  skipping dead-lettered tombstones, so the claim keeps `(enqueued_at)`.
- **A separate dead-letter table.** A counter on the queue row does the same
  with no move between tables.
- **Count a round that never ended when a later claim finds its lease
  passed**, instead of counting every attempt in its claim. That claim would
  need the earlier claim's state and time after its own write replaced them,
  and `RETURNING` gives only new values on SQLite and on PostgreSQL before
  18: it takes a column kept for the purpose, a claim that writes each column
  one of two ways, and a second kind of claim result that runs no round. Its
  report would also describe, at the later claim's time, an attempt made
  minutes earlier. Counting in the claim needs none of that; it gives up only
  an error log when a purger dies on the last attempt.
- **Batched deletes on Qdrant; one filter-delete on Milvus; listing Milvus
  keys by primary-key range.** Measured worse; see the per-backend documents.
- **Hold the claim's transaction across the round**: a row lock on
  PostgreSQL, SQLite's write lock on SQLite. Rejected: SQLite's lock covers the
  whole database file, so every writer to the database waited out the round's
  remote calls, each bounded by the request timeout, and failed past its busy
  timeout (SQLite's 5 s by default) with a locked-database error. On
  PostgreSQL the session sat idle in a transaction for the round, holding
  back vacuum.
- **Renew the lease while a round runs.** A round is bounded by its requests'
  timeouts, and overlapping rounds are harmless, so renewal would add a task
  per round to save work only when a round outlasts a lease set above that
  bound.
- **A token the purger mints for each claim**, in place of the generation.
  Either fences the claim's writes; the generation needs nothing from the
  purger, since the database assigns it.
- **Fence the tombstone's removal too.** An incarnation found empty after the
  retention needs no more rounds, whoever found it, so the fence would only
  discard the finding and repeat the round.
- **Store when the lease ends (`claimed_until`).** Rejected, as `retry_at` is,
  in favor of computing from the recorded `claimed_at`, so a changed lease
  reaches claims already held.
- **A progress cursor on the tombstone**, as the segment store's queue keeps.
  Qdrant's round deletes the whole incarnation at once and leaves nothing to
  resume. Milvus lists the incarnation's keys by its field, and its rounds
  stayed flat to the end of a 1M purge, where the segment store's batches
  needed the cursor to avoid stepping over every row already purged; listing
  Milvus keys by primary-key range, which a cursor needs, measured no faster.
  A cursor would also make the round carry a backend-specific position.
- **A lock service keyed by name.** It locks a key the caller already knows,
  while the claim picks the oldest due tombstone no claim holds, in one
  statement. With the lease in another table, picking and locking become
  separate steps that race, and a round's writes could not check the fence in
  the tombstone's own row.
- **A stronger read level for the purge than for the store's other reads**
  (Strong on Milvus). It would let a round see the round before it, sparing a
  repeated listing, but the design needs no more than the retention already
  guarantees.

## Consequences

- Deleted records stay in the backend at least the retention: storage is
  reclaimed a day after deletion by default.
- An operator watches for the dead-letter error log and for repeated retry
  warnings. A dead-lettered tombstone's records stay until someone resets its
  `attempts_without_progress`.
- The purge loads the backend in bounded rounds from every process's sweeper,
  and the rounds of different tombstones proceed in parallel.
- A tombstone whose round died waits out the lease and the backoff, its
  attempt counted by its claim: a tombstone whose rounds keep killing their
  purger is dead-lettered like one whose rounds keep raising, without the
  error log.
