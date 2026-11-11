# Event memory: ingest recovery and deletion across two stores

Event memory keeps events, their segments, and the segments' derivatives in
a SQL segment store, and the derivatives' vectors in a vector store. The
two share no transaction. This document is how an ingest stays complete
and a forget stays complete anyway.

## Requirements

- **Deletion.** After a forget commits, the events' text is gone from SQL,
  and their vectors are deleted at once and again after a retention R.
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

**State.** The segment store keeps each derivative with its segment, as
derived: what its index is written from, and the segment a hit on it
names. Besides those, it keeps a pending write for each segment with
derivatives, until the vector store has them, and a tombstone for each
forgotten segment with derivatives, naming them, in its own database and
transactions. Both are a segment's, so each holds a bounded amount of work
when the segmenter bounds its segments, and neither holds a value the
caller chose.

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

**Forget.** One transaction locks the events, reads their segments'
derivatives, writes a tombstone for each segment with derivatives, naming
them, and deletes the events, which cascades to segments, derivatives, and
pending writes. The vectors are then deleted once and the first delete
recorded, and each tombstone awaits its second delete, due R after that
record. A failure here leaves the first
delete to the background work.

**Background work.** Any process claims, from the segment store:

- pending writes whose ingest's hold has passed: it embeds the segment's
  stored derivatives, starts a budget of its own, and renews its claim,
  then writes them and deletes the record;
- tombstones due for a delete, at once for the first and R after the
  first was recorded for the second, and deletes their derivatives from
  the vector store.

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
deriver may not give the same derivatives twice.

**Search.** A vector hit whose derivative is gone is dropped from the
results and its vector deleted once, best effort and without the search
waiting for it; the next search that finds it tries again.

**Partition deletion.** The segment store's partition goes first, so its
text is gone at once and no write starts on it; the vector store's
partition deletion then removes every vector, with its own retention for
writes in flight. Partitions are created in the other order, so a segment
store partition always has its vector store partition.

## Why it is safe

A deriver gives every derivative a UUID no other derivative has, so a
tombstone's deletes reach only the vectors of the events it forgot, and
never those of an event added again under the same UUID.

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

A claim's renewal holds its record's row, so the forget's cascade waits for
a renewal in progress: a renewal either commits before the forget or finds
its record gone. On PostgreSQL, transactions run at READ COMMITTED, where
each statement reads its own snapshot, so the forget acts only on the
events its lock returned: a statement that rereads the event list could
delete an event committed after the derivatives were read, cascading to
derivatives nobody tombstoned. SQLite serializes write transactions, so the same
transactions are atomic there.

A recovery opens the vector store partition under the partition's key
before renewing its claim, and the renewal checks that the partition it
claimed from still exists: since deletion removes the segment store's
partition first, the vector store partition it opened belongs to the same
life. Embedding before the budget starts lets a recovery take as long to
embed as an ingest does.

The protocol was model-checked on small instances (an event encoded once
or twice, one or two recovery workers, a crash and a timeout at any step,
a forget at any moment) for deletion, findability of every stored vector,
completion, durability and erasure. Removing any one piece above breaks
one of them.

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
  deletes run; their text is gone from SQL when the forget commits.
- A deployment's sessions share its vector store, segment store, and
  embedder. Pointing the deployment at other stores moves no data.
- Recovery needs the embedder and the vector store, and waits out their
  outages in its backoff. A record or tombstone that keeps failing is
  reported and kept, and needs an operator to retry it.
- A process paused longer than R between its budget check and a send, and
  a restore of SQL from a backup, are outside this design.
