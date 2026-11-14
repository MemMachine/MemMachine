# Event memory: ingest recovery and deletion across two stores

Event memory keeps events, their segments, and the segments' derivatives in
a SQL segment store, and the derivatives' vectors in a vector store. The
two share no transaction. This document is how an ingest stays complete
and a forget stays complete anyway.

## Requirements

- **Deletion.** After a forget commits, the events, their segments, and
  their derivatives are deleted from SQL, and the derivatives' vectors are
  deleted at once and again after a retention R.
- **Adding again.** A forgotten event can be added again under its UUID as
  soon as the forget commits.
- **Late writes.** A write to the vector store can be applied after its
  client stopped waiting for it, and after a delete sent later. A write
  landing more than R after its forget is outside this design, provided no
  code path of ours sends after its deadline and R exceeds any in-flight
  time by orders of magnitude.
- **Self-recovery.** An event the server accepted is ingested in full even
  if the process stops midway and the client never retries.
- **Deadline, not lock.** No lock is held across a vector store call, and
  no correctness argument rests on one.
- **Horizontal scaling.** Any process serves any request and any process
  takes up background work; opening a partition has no side effects.

## The protocol

**State.** The segment store keeps each segment's derivatives as derived,
from the ingest until the forget: what its index is written from, and the
segment a hit on one names. Besides those, it keeps a pending write for
each segment, until the vector store has its derivatives, and a tombstone
for each forgotten segment, holding its derivatives' UUIDs until the
second delete, in its own database and transactions. Both are a segment's, so
each holds a bounded amount of work when the segmenter bounds its
segments, and neither holds a value the caller chose.

**Ingest.**

1. Segment, derive and embed.
2. Start the write budget, before anything is stored.
3. In one transaction, store the events, their segments, the derivatives,
   and the pending writes, which the ingest holds for its budget plus a
   margin. A batch naming an event already stored is rejected
   whole.
4. Upsert the vectors, cancelled when the budget runs out.
5. Delete the pending writes.

The ingest succeeds once step 3 commits: its events are accepted, and
stored with everything their derived index is built from. Steps 4 and 5
only spare the background work: a failure there is logged, and a failure
or a stop after step 3 leaves the derivatives pending, for the background
work to write.

**Forget.** One transaction locks the events and their segments, writes a
tombstone for each segment, held for the forget's lease, copies the
segment's derivative UUIDs into it, and deletes the events, which cascades
to the segments and to their pending writes and derivatives. The vectors
are then deleted once and the first delete recorded, and each tombstone
awaits its second delete, due R after that record, which deletes the
tombstone and, by cascade, its derivative UUIDs. A failure here leaves the
first delete to the background work, once the hold has passed.

**Background work.** Any process claims, from the segment store:

- pending writes whose ingest's hold has passed: it embeds the segment's
  stored derivatives, starts a budget of its own, and renews its claim,
  then writes them and deletes the record;
- tombstones due for a delete, the first once the forget's hold has
  passed and the second R after the first was recorded, and deletes their
  derivatives from the vector store.

Each record stores when a claim may next take it. A claim holds its record
for a lease, and one that never ends is taken again once the lease and
then the backoff have passed. A failed claim backs off, doubling, a
cancelled one takes back its attempt, and a record is no longer taken
after repeated failures, which is reported. The second delete's retention
is a condition of its own, so no schedule brings it forward.

The records name no vector store or embedder: a deployment has one
EventMemory, built from its configuration and started with the server,
and every partition in its segment store is that EventMemory's. So the
background work writes and deletes with the EventMemory's own stores, and
runs whether or not any session is opened.

**Further derived indexes.** Each derived index builds its own records
from the stored events, segments, and derivatives, keys them its own way,
and maps its hits to segments; it must accept a repeated write or delete
of a record. A segment's pending write is deleted, and its tombstone's
first delete recorded, only once every index has done its part, so a
further index adds writes and deletes, not rows or columns. What an index
cannot rebuild under the same keys from what is stored is stored with the
segment, as the derivatives are: their UUIDs are minted at ingest, and a
deriver may not give the same derivatives twice. An index keys its records
by what an ingest mints, never by the event's UUID, so the deletes of a
forgotten event never reach the records of the event added again under
its UUID.

**Search.** A vector hit whose derivative is no longer stored is dropped
from the results, and its vector is deleted without the search waiting
for it; if that delete fails, the next search that finds the hit deletes
it again.

**Partition deletion.** The segment store's partition is deleted first, so its
text is gone at once and no write starts on it; the vector store's
partition deletion then removes every vector, with its own retention for
writes in flight. Partitions are created in the other order, so a segment
store partition always has its vector store partition.

## Why it is safe

A segmenter gives every segment, and a deriver every derivative, a UUID no
other has, so a tombstone's deletes reach only the vectors of the events it
forgot, and never those of an event added again under the same UUID.

Every vector write is sent within the budget L of a moment its derivative
was stored and its event not forgotten: an ingest starts its budget before
its transaction, and a recovery starts its budget before renewing its
claim, which a forget of the same derivatives must follow. The forget
commits after that moment, and its first delete is recorded after the
forget commits, so the write lands before that record plus L plus the
in-flight time d, and the second delete, R after the record, removes it
whenever

    R >= L + d

The deployment keeps R above ten times L, with a margin for d.

Cancelling the upsert at the deadline bounds when the last request is
sent, whatever the vector store does inside one call. A request already
sent may still land; d covers it.

A claim's renewal holds its record's row, so the forget's cascade waits
for a renewal in progress: a renewal either commits before the forget or
finds its record gone. Pending writes and tombstones are separate tables,
so neither a claim on a pending write nor the writer's own delete of its
pending writes can reach a tombstone. On PostgreSQL, transactions run at
READ COMMITTED, where each statement reads its own snapshot, so the forget
acts only on the events its lock returned: a statement that rereads the
event list could delete an event committed after the segments were read,
cascading to segments nobody tombstoned. SQLite serializes write
transactions, so the same transactions are atomic there.

A recovery opens the vector store partition under the partition's key
before renewing its claim, and the renewal checks that the partition it
claimed from still exists: since deletion removes the segment store's
partition first, the vector store partition it opened belongs to the same
life. Embedding before the budget starts lets a recovery take as long to
embed as an ingest does.

No transactions on PostgreSQL can wait on each other in a cycle, since
each is one of five kinds:

- **One row.** It locks one row at most: a claim, whose update takes the
  record it locked from a materialized CTE, which PostgreSQL evaluates
  once; a renewal, fail or cancel; a pending write's finish; and a first
  delete's finish. It never holds a lock while it waits for another.
- **Never waits.** It takes every lock it could wait for with `SKIP
  LOCKED`: the writer's delete of its pending writes, and the request's
  record of a first delete. A row it skips is a forget's, which deletes
  it, or a one-row statement's, whose work the background work completes.
- **One root.** It waits for one contested row before it holds any, and
  reaches the rest only through it, where it waits only on one-row and
  never-waiting transactions: an ingest of one event, which waits only on
  a transaction adding or deleting the same event; a forget of one event,
  which locks the event and reaches its segments, pending writes and
  derivatives through it; and the second delete, which deletes its
  tombstone before the tombstone's derivative UUIDs. A forget never waits
  on an ingest, whose rows it cannot see before they commit.
- **Takes turns.** An ingest or forget of several events can wait for one
  event while holding another, so it holds the partition's registry row
  under `FOR NO KEY UPDATE`, and such writes in a partition run one at a
  time. Every other write pins the row under `FOR KEY SHARE`, which that
  lock does not block.
- **Fenced.** A transaction that locks more than one row of a partition
  holds a shared lock on the partition's registry row; the partition's
  deletion takes that row exclusively, and the purge reclaims only
  partitions whose deletion committed, so the purge meets only one-row
  statements, and other purges only at the queue, which it claims with
  `SKIP LOCKED`.

A wait cycle needs two transactions that each hold a lock while waiting
for another. A one-row, never-waiting or one-root transaction never waits
while holding a lock another member of a cycle could want, the purge
meets only one-row statements, and two writes of several events never run
at once. Each pair of operations that can lock the same rows:

| Operations | Rows both lock | Why neither waits on the other in a cycle |
|---|---|---|
| Forgets of the same events | events | Forgets of several events take turns; one of one event waits for it before holding any |
| An ingest and a forget of the same event | the event | The forget never waits on the ingest |
| Ingests naming the same events | events | Ingests of several events take turns; one of one event waits before holding any |
| A write of several events and any other write | the registry row | The other write's pin does not wait for the write lock |
| A forget and the writer's delete of its pending writes | pending writes | The delete never waits |
| A forget and a claim, renewal, fail, cancel or finish of a pending write | a pending write | The background statement locks one row |
| The request's first-delete record and a tombstone's claim, fail, cancel or first-delete finish | a tombstone | The record never waits |
| Two second deletes of a tombstone | the tombstone, its UUIDs | Both delete the tombstone first |
| A partition's deletion and a transaction locking several of its rows | the registry row | The deletion waits for the write lock and the pins |
| A purge and a one-row background statement | a deleted partition's row | The background statement locks one row |
| Two purges | a queue entry | The claim never waits |

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
- **One table for pending writes and tombstones.** The forget would turn a
  segment's row from one into the other, keeping only its key, and every
  statement on a row would have to name its kind as well as its
  generation; columns would mean something for one kind only.
- **Time the second delete from the forget.** The forget's time must then
  follow every renewal it waited for, which on PostgreSQL takes deleting
  the pending writes before taking the time, and taking it at the insert
  statement; timing the second delete from the first delete's record
  needs neither.
- **Store-side fences** (expiring credentials, insert-only markers). They
  are checked only at admission, so an admitted write still lands late.
- **One owner per tenant.** A paused former owner is the same problem as a
  paused writer, and the stores cannot reject its writes, so the deadline
  and the tombstones stay; ownership adds routing, leases and failover.
- **Leave every index write to the background work**, the outbox's relay
  alone. One path writes the indexes, but every event waits for a claim,
  and ingest runs at the background work's claim rate rather than with
  the requests'. Writing in the request and falling back to the pending
  write keeps the common case written when the request returns.
- **Vectors in the SQL database.** Exact, with no late writes, but only for
  the backends that offer it.

## Consequences

- An event is readable before its vectors exist, and searchable once they
  do. An ingest's success means its events are stored; when they are
  searchable is each derived index's delay, which no check reports.
- Forgotten vectors stay physically in the vector store until their
  deletes run; the events, segments, and derivatives are deleted from SQL
  when the forget commits, and a tombstone keeps only its derivatives'
  UUIDs until the second delete.
- A deployment's sessions share its vector store, segment store, and
  embedder. Pointing the deployment at other stores moves no data.
- Recovery needs the embedder and the vector store, and waits out their
  outages in its backoff. A record or tombstone that keeps failing is
  reported and kept, with a tombstone's derivative UUIDs, and needs an
  operator to retry it.
- A process paused for about R or longer after its budget starts, and a
  restore of SQL from a backup, are outside this design.
