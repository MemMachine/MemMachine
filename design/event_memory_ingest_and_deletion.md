# Event memory: ingest recovery and deletion across two stores

Event memory keeps events, their segments, and the links from derivatives to
segments in a SQL event memory store, and the derivatives' vectors in a vector
store. The two share no transaction. This document is how an ingest stays
complete and a forget stays complete anyway.

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

**State.** Besides events, segments and links, the event memory store keeps
pending derivatives (one record per event, holding its derivatives until
the vector store has them) and tombstones (one per forget, naming the
derivatives it unlinked), in its own database and transactions. Neither
holds a value the caller chose.

**Ingest.**

1. Segment, derive and embed.
2. Start the write budget, before anything is stored.
3. In one transaction, store the events, their segments, the derivative
   links, and the pending derivatives. A batch naming an event already
   stored is rejected whole.
4. Upsert the vectors, cancelled when the budget runs out.
5. Delete the pending derivatives.

A failure or a stop after step 3 leaves the events stored with their
derivatives pending, and the background work writes them.

**Forget.** One transaction locks the events, reads their links, deletes
their pending derivatives, writes a tombstone naming the linked
derivatives, and deletes the events, which cascades to segments and links.
The vectors are then deleted once, and the tombstone awaits its second
delete. A failure here leaves the first delete to the background work.

**Background work.** Any process claims, from the event memory store:

- pending derivatives older than the budget plus a margin, starting a
  budget of its own before the claim, then writes them and deletes the
  record;
- tombstones due for a delete, at once for the first and after R for the
  second, and deletes their derivatives from the vector store.

A claim holds its record for a lease, fails into a backoff that doubles,
and is no longer taken after repeated failures, which is reported.

The records name no vector store or embedder: a deployment has one
EventMemory, built from its configuration and started with the server,
and every partition in its event memory store is that EventMemory's. A session
naming another vector store, event memory store, or embedder is refused. So the
background work writes and deletes with the EventMemory's own stores, and
runs whether or not any session is opened.

**Search.** A vector hit with no link is dropped from the results and its
vector deleted once, best effort; the next search that finds it tries
again.

**Partition deletion.** The event memory store's partition goes first, so its
text is gone at once and no write starts on it; the vector store's
partition deletion then removes every vector, with its own retention for
writes in flight. Partitions are created in the other order, so a segment
store partition always has its vector store partition.

## Why it is safe

Every vector write is sent within the budget L of a moment its derivative
was linked and its event not forgotten: an ingest starts its budget before
its transaction, and a recovery starts its budget before its claim, which a
forget of the same derivatives must follow. The forget's tombstone is
stamped after that moment, so the write lands before the tombstone's time
plus L plus the in-flight time d, and the second delete, after R, removes
it whenever

    R >= L + d

The deployment keeps R above ten times L, with a margin for d.

Cancelling the upsert at the deadline bounds when the last request is
sent, whatever the vector store does inside one call. A request already
sent may still land; d covers it.

On PostgreSQL, transactions run at READ COMMITTED, where each statement
reads its own snapshot, so the forget follows three rules:

1. **It acts only on the events its lock returned.** A statement that
   rereads the event list could delete an event committed after the links
   were read, cascading to links nobody tombstoned.
2. **The tombstone's time is that of its insert statement**, after the
   locks. The transaction's start time could precede an ingest the locks
   waited for.
3. **It deletes the pending derivatives before stamping the tombstone.** A
   claim holds its record, so the delete waits for a claim in progress: a
   claim either committed before the stamp or finds its record gone.

SQLite serializes write transactions, so the same transactions are
atomic there.

A recovery opens the vector store partition under the partition's key, so
it checks after opening that the partition it claimed from still exists:
since deletion removes the event memory store's partition first, the vector
store partition it opened belongs to the same life.

The protocol was model-checked on a small instance (two encodes of one
event, an evicting encode, two recovery workers, crashes and timeouts at
every step, forgets at any moment) for deletion, findability of every
stored vector, completion, durability and erasure. Removing any one piece
above breaks one of them.

## Alternatives

- **Hold the event memory store's transaction across the upsert.** It holds a
  pooled connection, or SQLite's whole-database write lock, across every
  vector call, and it does not bound when an admitted write lands, so it
  still needs tombstones and R.
- **Write vectors first.** A write is then in the store before anything in
  SQL can find it; a crash there leaves it unreachable.
- **Look up, then delete in another transaction.** An ingest committing in
  between leaves vectors whose links the delete removed.
- **One delete.** A write admitted before it can land after it, as
  measured.
- **Store-side fences** (expiring credentials, insert-only markers). They
  are checked only at admission, so an admitted write still lands late.
- **One owner per tenant.** A paused former owner is the same problem as a
  paused writer, and the stores cannot reject its writes, so the deadline
  and the tombstones stay; ownership adds routing, leases and failover.
- **Vectors in the SQL database.** Exact, with no late writes, but only for
  the backends that offer it.

## Consequences

- An event is readable before its vectors exist, and searchable once they
  do.
- Forgotten vectors stay physically in the vector store until their
  deletes run; their text is gone from SQL when the forget commits.
- A deployment's sessions share its vector store, event memory store, and
  embedder; a session stored with others is refused until its data is
  moved to the deployment's.
- Recovery needs the embedder and the vector store, and waits out their
  outages in its backoff. A record or tombstone that keeps failing is
  reported and kept, and needs an operator to retry it.
- A process paused longer than R between its budget check and a send, and
  a restore of SQL from a backup, are outside this design.
