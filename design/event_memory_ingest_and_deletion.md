# Event memory: ingest recovery and deletion across two stores

Event memory keeps events, their segments, and the segments' derivatives in
a SQL segment store, and the derivatives' vectors in a vector store. The
two share no transaction. This document is how an ingest completes once
its events are stored, and how a forget completes once it commits.

## Requirements

- **Deletion.** After a forget commits, the events, their segments, and
  their derivatives are deleted from SQL, and the background work deletes
  the derivatives' vectors, and again after a retention R.
- **Adding again.** A forgotten event can be added again under its UUID as
  soon as the forget commits.
- **Late writes.** A write to the vector store can be applied after its
  client stopped waiting for it, and after a delete sent later. A write
  landing more than R after its forget is outside this design, provided
  every vector store call ends within its client's configured timeouts and
  R exceeds them and any in-flight time by orders of magnitude.
- **Self-recovery.** An event its ingest stored is indexed in full even if
  the process stops after that and the client never retries.
- **Time bounds, not locks.** No lock is held across a vector store call,
  and no correctness argument rests on one.
- **Horizontal scaling.** Any process serves any request and any process
  takes up background work; opening a partition has no side effects.

## The protocol

**State.** The segment store keeps each segment's derivatives as derived,
from the ingest until the forget: what its index is written from, and the
segment a hit on one names. Besides those, it keeps an indexing job for
each segment, until the vector store has its derivatives, and a tombstone
for each forgotten segment, holding its derivatives' UUIDs until the
second delete, in its own database and transactions, all under the
partition's UUID. Both are a segment's, so
each holds a bounded amount of work when the segmenter bounds its
segments, and neither holds a value the caller chose.

**Ingest.**

1. Segment, derive and embed.
2. Check the partition's registration.
3. In one transaction, store the events, their segments, the derivatives,
   and the indexing jobs, which the ingest holds for its lease. A batch
   naming an event already stored is rejected whole.
4. Upsert the vectors.
5. Delete the indexing jobs.

The ingest succeeds once step 3 commits: its events are accepted, and
stored with everything their derived index is built from. Steps 4 and 5
only spare the background work: a failure there is logged, and a failure
or a stop after step 3 leaves the derivatives pending, for the background
work to write.

**Forget.** The forget checks the partition's registration. Then one
transaction locks the events and their segments, writes a
tombstone for each segment, copies the segment's derivative UUIDs into it,
and deletes the events, which cascades to the segments and to their
indexing jobs and derivatives. The background work then deletes the
vectors and records that first delete, and each tombstone awaits its
second delete, due R after that record, which deletes the tombstone and,
by cascade, its derivative UUIDs.

**Background work.** Any process claims, from the segment store:

- indexing jobs whose ingest's hold has passed: it embeds the segment's
  stored derivatives, renews its claim, and checks the claim's partition
  by its UUID, then writes them and deletes the record, or deletes the
  record without writing once the partition is deleted;
- tombstones due for a delete, the first once the forget commits and the
  second R after the first was recorded: it checks the claim's partition
  by its UUID, and deletes their derivatives from the vector store, or
  leaves them to the partition's purge once the partition is deleted.

Each record stores when a claim may next take it. A claim holds its record
for a lease, and one that never ends, as when its process stops, is taken
again once the lease and then the backoff have passed. A failed claim
backs off, doubling, and a record is no longer taken after repeated
failures, which is reported. The second delete's retention
is a condition of its own, so no schedule brings it forward.

The records name no vector store or embedder: a deployment has one
EventMemory, built from its configuration, and every partition in its
segment store is that EventMemory's. So the background work writes and
deletes with the EventMemory's own stores. EventMemory runs one indexing
job, or one tombstone's delete, when asked, and the server, like its
purges, runs each kind in a loop of its own from startup, whether or not
any session is opened, with no pause after a job and a short one after
finding none due.

**Further derived indexes.** Each derived index builds its own records
from the stored events, segments, and derivatives, keys them its own way,
and maps its hits to segments; it must accept a repeated write or delete
of a record. A segment's indexing job is deleted, and its tombstone's
first delete recorded, only once every index has done its part, so a
further index adds writes and deletes, not rows or columns. What an index
cannot rebuild under the same keys from what is stored is stored with the
segment, as the derivatives are: their UUIDs are minted at ingest, and a
deriver may not give the same derivatives twice. An index keys its records
by what an ingest mints, never by the event's UUID, so the deletes of a
forgotten event never reach the records of the event added again under
its UUID.

**Search.** A vector hit whose derivative is no longer stored is dropped
from the results; its forget's deletes remove the vector.

**Partition deletion.** Partitions are the [partition
registry](partition_registry.md)'s: a deleted partition is unreachable at
once, and purge rounds reclaim its records from both stores, its indexing
jobs and tombstones included, once the namespace's retention has passed
since the deletion. Until then, a claim on a deleted partition's indexing
job or tombstone ends without writing or deleting.

## Why it is safe

The design requires every segmenter to give each segment, and every deriver
each derivative, a UUID no other has. A tombstone's deletes then reach only
the vectors of the events it forgot, and never those of an event added
again under the same UUID.

Every vector write is one vector store call, made right after a moment
its derivative was stored and its event not forgotten: an ingest's commit,
or a recovery's renewal of its claim, which a forget of the same
derivatives must follow. Both embed before that moment, so only the call
runs between it and the call's requests, and, for a recovery, its check of
its partition, one read of the registry. The call sends its last request
within a time T that its client's configured timeouts bound, retries and
the registry read included, and a request lands within the in-flight time
d after it is sent. The forget commits after that moment, and its first delete is
recorded after the forget commits, so the write lands before that record
plus T plus d, and the second delete, R after the record, removes it
whenever

    R >= T + d

The deployment keeps R above ten times T, with a margin for d.

A partition's deletion rests on the same bound. Every write follows a check
of the partition that found it live: an ingest's and a forget's of its
registration, and a recovery's of its claim's partition by UUID. Each
checks after its own work, so only its store calls run between the check
and their requests, and every write lands within T plus d of the check, or
the segment store's own bound for its transactions. The purge waits out the
namespace's retention after the deletion, which exceeds that bound, so it
reclaims every write the check let through.

A claim's renewal holds its record's row, so the forget's cascade waits
for a renewal in progress: a renewal either commits before the forget or
finds its record gone. Indexing jobs and tombstones are separate tables,
so neither a claim on an indexing job nor the ingest's own delete of its
indexing jobs can reach a tombstone. On PostgreSQL, transactions run at
READ COMMITTED, where each statement reads its own snapshot, so the forget
acts only on the events its lock returned: a statement that rereads the
event list could delete an event committed after the segments were read,
cascading to segments nobody tombstoned. SQLite serializes write
transactions, so the same transactions are atomic there.

A recovery checks its claim's partition after renewing its claim, right
before its write. A UUID names one life of a partition, so the check
answers whether that life is live, whatever was registered under its name
since.

No transactions on PostgreSQL can wait on each other in a cycle, since
each is one of four kinds:

- **One row.** It locks one row at most: a claim, whose update takes the
  record it locked from a materialized CTE, which PostgreSQL evaluates
  once; a renewal or fail; an indexing job's finish; and a first
  delete's finish. It never holds a lock while it waits for another.
- **Never waits.** It takes every lock it could wait for with `SKIP
  LOCKED`: the ingest's delete of its indexing jobs. A row it skips is a
  forget's, which deletes it, or a claim's, whose work the background work
  completes.
- **One root.** It waits for one contested row before it holds any, and
  reaches the rest only through it, where it waits only on one-row and
  never-waiting transactions: an ingest of one event, which waits only on
  a transaction adding or deleting the same event; a forget of one event,
  which locks the event and reaches its segments, indexing jobs and
  derivatives through it; and the second delete, which deletes its
  tombstone before the tombstone's derivative UUIDs. A forget never waits
  on an ingest, whose rows it cannot see before they commit.
- **Takes turns.** An ingest or forget of several events can wait for one
  event while holding another, so it locks the partition's row in the
  segment store under `FOR NO KEY UPDATE` first, and such writes in a
  partition run one at a time. An ingest or forget of one event locks the
  row under `FOR KEY SHARE` first, which that lock does not block. The
  purge takes the row under `FOR UPDATE` before any row it deletes, which
  waits for both and blocks both, so a purge runs alone among the
  partition's writes of events and its other purges. A forget whose
  partition's row is gone, the partition dropped, deletes nothing, so a
  purge that finds no row to lock meets no forget.

A wait cycle needs two transactions that each hold a lock while waiting
for another. A one-row, never-waiting or one-root transaction never waits
while holding a lock another member of a cycle could want, and two
transactions that take turns never run at once. Each pair of operations
that can lock the same rows:

| Operations | Rows both lock | Why neither waits on the other in a cycle |
|---|---|---|
| Forgets of the same events | events | Forgets of several events take turns; one of one event waits for it before holding any |
| An ingest and a forget of the same event | the event | The forget never waits on the ingest |
| Ingests naming the same events | events | Ingests of several events take turns; one of one event waits before holding any |
| A write of several events and any other write of events | the partition's row | The other write's lock does not wait for the write of several events |
| A forget and the ingest's delete of its indexing jobs | indexing jobs | The delete never waits |
| A forget and a claim, renewal, fail or finish of an indexing job | an indexing job | The background statement locks one row |
| Two second deletes of a tombstone | the tombstone, its UUIDs | Both delete the tombstone first |
| A purge and a write of events, or another purge | the partition's row | Both take it before any other row |
| A purge and the ingest's delete of its indexing jobs | indexing jobs | The delete never waits |
| A purge and a claim, renewal, fail or finish | an indexing job or a tombstone | The background statement locks one row |
| A purge and a second delete | a tombstone, its UUIDs | Both delete the tombstone first |

SQLite serializes write transactions, each taking the write lock at its
`BEGIN`, so no cycle forms there.

The protocol was model-checked on small instances (an event encoded once,
with one or two recovery workers and a crash and a timeout at any step, or
encoded, forgotten and encoded again; a forget at any moment) for
deletion, findability of every stored vector, completion, durability and
erasure. Removing any one piece above breaks one of them.

## Alternatives

- **Hold the segment store's transaction across the upsert.** It holds a
  pooled connection, or SQLite's whole-database write lock, across every
  vector call, and it does not bound when an admitted write lands, so it
  still needs tombstones and R.
- **Write vectors first.** A write is then in the store before anything in
  SQL can find it; a crash there leaves it unreachable.
- **Look up, then delete in another transaction.** An ingest committing in
  between leaves vectors whose links the delete removed.
- **One delete.** A write admitted before it can land after it, as
  measured.
- **Keep the derivatives until the second delete**, the tombstone reading
  their UUIDs from them. It spares the forget a copy and a delete per
  derivative, but leaves the forgotten text in SQL for R, leaves the
  derivatives without a foreign key to their segment, and makes every
  read of derivatives by UUID join the segments to skip forgotten ones.
- **Keep the forgotten segments too, until the second delete.** Every read
  of segments would then have to skip forgotten ones.
- **One table for indexing jobs and tombstones.** The forget would turn a
  segment's row from one into the other, keeping only its key, and every
  statement on a row would have to name its kind as well as its
  generation; columns would mean something for one kind only.
- **Time the second delete from the forget.** The forget's time must then
  follow every renewal it waited for, which on PostgreSQL takes deleting
  the indexing jobs before taking the time, and taking it at the insert
  statement; timing the second delete from the first delete's record
  needs neither.
- **Claims that skip deleted partitions**, the segment store reading a
  partition's liveness in the claim's own statement, as from a row of its
  own per partition life. The partition registry keeps liveness, in a
  database that may be another, so a claim's holder checks the partition
  instead, and a deleted partition's records are claimed until its purge
  deletes them.
- **Store-side fences** (expiring credentials, insert-only markers). They
  are checked only at admission, so an admitted write still lands late.
- **One owner per tenant.** A paused former owner is the same problem as a
  paused process mid-write, and the stores cannot reject its writes, so
  the tombstones and R stay; ownership adds routing, leases and failover.
- **A deadline of event memory's own**, cancelling the upsert when it
  passes. It bounds the call whatever its client's timeouts, but needs
  every vector store to be safe to cancel mid-call.
- **Leave every index write to the background work**, the outbox's relay
  alone. One path writes the indexes, but every event waits for a claim,
  and ingest runs at the background work's claim rate rather than with
  the requests'. Writing in the request and falling back to the indexing
  job keeps the common case written when the request returns.
- **Delete the vectors in the forget's request** as well. The vectors go
  before the forget returns rather than at the next claim, but the request
  and a claim can then both do the first delete, which takes a hold on new
  tombstones and a generation bump to end a claim that overlaps the
  request. Results do not change, since a search drops hits whose
  derivatives are deleted.
- **Vectors in the SQL database.** Exact, with no late writes, but only for
  the backends that offer it.

## Consequences

- An ingest that fails before its events are stored, in the embedder, the
  segmenter, or the deriver, stores nothing and raises; the caller adds the
  events again.
- A deleted partition's indexing jobs and tombstones are claimed, and end
  without work, until its purge deletes them.
- An event is readable before its vectors exist, and searchable once they
  do. An ingest's success means its events are stored; when they are
  searchable is each derived index's delay, which no check reports.
- Forgotten vectors stay physically in the vector store until their
  deletes run; the events, segments, and derivatives are deleted from SQL
  when the forget commits, and a tombstone keeps only its derivatives'
  UUIDs until the second delete.
- A deployment's sessions share its vector store, segment store, embedder,
  reranker, segmenter, and deriver. Pointing the deployment at other stores moves no data.
- Recovery needs the embedder, the vector store, and the partition
  registry, and waits out their outages in its backoff. A record or tombstone that keeps failing is
  reported and kept, with a tombstone's derivative UUIDs, and needs an
  operator to retry it.
- A process paused for about R or longer between storing or renewing and
  its write's requests, and a restore of SQL from a backup, are outside
  this design.
