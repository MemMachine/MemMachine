# Vector store: the collection registry

Part of [vector store horizontal scaling](vector_store_horizontal_scaling.md).

## Problem

A Qdrant or Milvus backend holds records and nothing that can arbitrate between
processes: no transactions, no unique constraints, and until Qdrant 1.16 no
conditional writes. The stores nevertheless kept their catalog inside the
backend, one `__registry` collection per namespace, and serialized their own
operations with process-local `asyncio` locks. Across processes that catalog
could not decide anything:

- **Creation raced.** Create was a read, then a write. Two processes creating
  the same name both succeeded: Qdrant's registry upsert let the last writer
  win, and Milvus's `insert` does not enforce primary-key uniqueness, so it
  kept two live entries.
- **Deletion did not end a collection.** The logical name was the tenant
  discriminator on every record, so a handle held in one process kept writing
  into a collection another process had deleted and re-created under the same
  name (#1563, reproducible sequentially, without any race).
- **Deletion raced with writes.** Delete was a filter-delete by name, and a
  write in flight during it outlived it.

The `VectorStore` contract answered all three by requiring that a collection
be managed by at most one process at a time, leaving the consumer to shard
names across processes, which no consumer did.

## Design

A collection's catalog entry lives where a primary key and a transaction can
arbitrate it: a relational database, reached through SQLAlchemy. The backend
keeps only records, each carrying the *incarnation* of the collection life that
wrote it.

### Roles

- `VectorStoreCollectionRegistry` (`common/vector_store/collection_registry/`)
  is the ABC, addressed by (namespace, name). Its operations are `startup`,
  `reserve`, `resolve`, `unregister`, and `run_purge_round`.
- `Reservation` and `Registration` are handles on one life of a collection
  (see [Reservations and registrations](#reservations-and-registrations)).
- `SQLAlchemyVectorStoreCollectionRegistry` is the one implementation. It
  supports PostgreSQL and SQLite 3.35 or newer (for `RETURNING`), as the
  segment store does, and its params refuse any other dialect or an older
  SQLite.
- `RegistryBackedVectorStore` (`common/vector_store/registry_backed_vector_store.py`)
  is the base of the Qdrant and Milvus stores and makes every registry call
  they make: create, open-or-create, open, delete, the purge round, and a
  handle's liveness fence. Its handle's `upsert`, `query`, and `delete` check
  their inputs and the handle's liveness around the backend call. A subclass
  supplies the backend steps: preparing a new collection's storage, the
  handle's backend calls (`_upsert`, `_query`, `_delete`), and one purge
  round over an incarnation's records. Nothing in the base assumes how the backend lays
  records out. Qdrant and Milvus share a native collection per namespace and
  configuration. A backend with a unit per collection, such as a Pinecone or
  turbopuffer namespace, a Chroma collection, or a Weaviate tenant, would make
  the collection's unit when it prepares its storage, which runs after the
  registration that mints the incarnation naming it, and its purge round
  would drop it.
- Every store whose client connects to the same backend data, the same Qdrant
  server or cluster or the same Milvus database, must share one registry, in
  any process; a store connected to other data must not. A store reclaims its
  registry's tombstones through its own client: a round run through a client
  that cannot reach the records finds none, removes the tombstone, and leaves
  the records behind.
- The caller owns the registry's lifecycle. The database manager builds and
  starts each Qdrant and Milvus store's registry before it opens the store's
  client, and hands the started registry to the store in its params. The store
  never starts or stops it: a store that started a registry it was given
  would own a lifecycle it does not control.

### Reservations and registrations

The registry is addressed by (namespace, name). What acts on one life of a
collection is a handle bound to that life's incarnation: the creator's
reservation, or a registration.

| Object | Operations | Answered by |
|---|---|---|
| `VectorStoreCollectionRegistry` | `reserve`, `resolve`, `unregister` (by name), `run_purge_round` | |
| `Reservation` | `confirm`, `cancel` | `reserve` |
| `Registration` | `require_current` | `resolve`, `confirm` |

- **Why handles.** Every operation on one life of a collection needs that
  life's incarnation: confirming it, cancelling it, and checking that it was
  not deleted. As methods of the registry, each would take the incarnation as
  an argument. The registry would then have two addressing schemes, names and
  incarnations, and every caller would carry the UUID `reserve` minted back
  into later calls, where any UUID fits: another collection's, a deleted
  one's, or one compared by hand against a lookup. On a handle, the
  incarnation is bound once, when the registry answers the handle, and no
  call takes it again. The store reads it to write records, and never passes
  it back.
- **Why two types.** What may be done with a collection depends on who holds
  it. Only its creator, holding the reservation, confirms or cancels it; a
  registration is checked, and its collection is deleted only by name. With
  one type, a registration would offer `confirm`, which could only fail, and
  `cancel`, which would delete a live collection outside the name-keyed path
  every deletion goes through. With two, each type offers its holder's
  operations and nothing else, and the wrong call is a type error.
- **Why role names, not state names.** A collection moves from pending to
  live only through its reservation's own `confirm`, but any caller can
  delete it at any time, through either handle's back. A name that states
  the collection's state ("pending", "live") is only true as of when the
  handle was answered; a name that states the holder's role stays true, and
  a deletion shows as the handle's methods raising. The words follow two
  known patterns: Try-Confirm/Cancel for the creator (reserve, then confirm
  or cancel), and the stale handle for everyone else (`require_current`,
  `VectorStoreCollectionHandleStaleError`).
- **Why no state fields.** A handle's fields (namespace, name, configuration,
  incarnation) belong to its life and never change. Whether the collection is
  pending, live, or deleted changes under every holder: a `live` field would
  be a snapshot, made stale by the creator's own `confirm`. The state is
  conveyed instead by the outcome of each call. `reserve` answers a
  reservation, and `confirm` a registration. `resolve` answers a
  registration or `None`, or raises when the collection is pending, and
  `require_current` raises once it is deleted. Each is one read or one
  write, so an open takes one round trip and the fence one.
- **What each part gains.** The registry's operations, beside `startup`, are
  all addressed by name. A store's collection handle is built from one
  registration, which carries everything the handle needs, and its fence is
  one call. The store's creation flow reads as the lifecycle it implements:
  reserve, prepare storage under the reservation's incarnation, then confirm
  the reservation, or cancel it if the preparation fails.

### Tables

The registries of every vector store share two tables, defined once at module
level as the segment store's are, and keyed by `vector_store_name`, the
backend's key under `resources.databases`. Two registry objects under one
name are one registry. One table pair with a key column, rather than a table
pair per vector store, keeps the schema static.

`collection_registry_ct`, the registered collections, pending or live:

| Column | Type | Role |
|---|---|---|
| `vector_store_name`, `namespace`, `name` | string, primary key | The collection's identity. The primary key arbitrates creation. |
| `incarnation` | UUID, unique | The collection's current life. Its records carry it. |
| `config` | JSON (JSONB on PostgreSQL) | The configuration the collection was created with. A handle is built from it; open-or-create compares against it. |
| `live` | boolean | Whether the collection's storage is prepared. A pending collection holds its name; only a live one is opened. |
| `registered_at` | timestamp | When the collection was registered, on the database clock. An operator finds a collection stuck pending by it, and opening a pending collection reports it. |

`collection_registry_gc`, the purge queue, one row per deleted incarnation
(its *tombstone*; see [purge](vector_store_purge.md)):

| Column | Type | Role |
|---|---|---|
| `incarnation` | UUID, primary key | The deleted life whose records remain. |
| `vector_store_name` | string | Whose purge claims the tombstone. |
| `namespace` | string | With `config`, locates the records in the store. |
| `name` | string | Kept for inspection; the purge does not read it. |
| `config` | JSON | With `namespace`, locates the records in the store. |
| `enqueued_at` | timestamp | When the deletion committed, on the database clock. |
| `attempts_without_progress` | integer | Purge rounds claimed since a round last found records and deleted them, the open one included; a cancelled round's attempt is taken back. |
| `last_failed_at` | timestamp, nullable | When a purge round last raised, on the database clock. |
| `claimed_at` | timestamp, nullable | When the open claim was taken, on the database clock; null when no claim is open. |
| `claim_generation` | integer | Incremented by each claim; a round's writes that end its claim are conditioned on it. |

Index `collection_registry_gc__vs_ea` on (`vector_store_name`, `enqueued_at`)
bounds the purge claim, equality before order. The index follows the
repository's naming scheme (table, two letters per column).

The queue carries `namespace` and `config` because nothing else locates a
dead incarnation's records once the collection is gone (on Qdrant and Milvus,
the native collection's name is derived from them). It carries `name` so an
operator looking at a tombstone, a dead-lettered one above all, can tell which
collection it was, as the segment store's queue carries its partition key.

### Operations

**`reserve(namespace, name, config) -> Reservation`.** Mints a random UUID
(version 4) and inserts the collection's row, pending. In the same
transaction, *after* the insert, a locking read checks that the incarnation is
not waiting in the purge queue. After the insert, a concurrent deletion that
moved a colliding row to the queue has already committed, because the insert
waited on it. After the check no queue row for this incarnation can appear
before commit, because the only row carrying it is this uncommitted one. A
primary-key violation with a row, pending or live, under the key raises
`VectorStoreCollectionAlreadyExistsError`. A rejected incarnation (unique
violation, or queued) is re-minted, up to 10 attempts, then
`VectorStoreAttemptsExhaustedError`. The loop and its bound are the segment
store's.

**`Reservation.confirm()`** marks the row live with an `UPDATE` conditional on
the reservation's incarnation, which is unique, and on the row being pending,
and answers the `Registration`; it raises `VectorStoreCollectionDeletedError`
when it matches nothing. A creation whose collection was deleted while its
storage was prepared matches nothing, so it cannot confirm a collection
reserved under the name since.

**`resolve(namespace, name)`** reads the collection's row once and answers by
its state: a `Registration` when it is live,
`VectorStoreCollectionPendingError`, with when the collection was reserved and
its configuration, when it is pending, and `None` when there is none. It
is how a handle is opened: callers address collections by name, and the name
is resolved to a live incarnation once, at open.

**`Registration.require_current()`** reads the row under the
registration's (namespace, name) and raises
`VectorStoreCollectionHandleStaleError` unless it carries the registration's
incarnation. It is how a handle is fenced (below).

**`unregister(namespace, name)`** is one transaction: `DELETE ... RETURNING`
the collection's row, pending or live, then insert its tombstone with
`enqueued_at = now()`. The collection is unreachable when it commits.
**`Reservation.cancel()`** does the same for the row carrying the
reservation's incarnation while it is pending: a creation whose storage
preparation raised takes back its own reservation that way, never one made
under the name since, and never its collection once confirmed, which only a
deletion by name ends, even when the confirmation committed but its answer
was lost. Racing deleters serialize on the row's write lock and the loser
deletes nothing, on PostgreSQL and SQLite alike, so a deletion is idempotent
and queues one tombstone.

**`run_purge_round()`** is described in
[purge](vector_store_purge.md).

### Creation

Creation is *reserved, prepared, then confirmed*. The store reserves the
collection's name, leaving it pending, prepares its storage (on Qdrant and
Milvus, the native collection its namespace and configuration share), and
confirms the reservation, which makes the collection live:

- The registry's primary key is the one arbiter: a racing creator on any
  process loses at the insert, never in the backend.
- A pending collection holds its name but is not opened: `open_collection`
  raises `VectorStoreCollectionPendingError`, which says since when it has
  been pending, a `create_collection` of the name raises
  `VectorStoreCollectionAlreadyExistsError`, and open-or-create waits for it.
  `None` from `open_collection` means only that no collection holds the name.
- A preparation or confirmation that raises, or is cancelled, cancels the
  reservation when the registry can, which frees the name and queues the
  incarnation's tombstone; a confirmation that committed before its failure
  was observed stands, since the cancel acts only on a pending collection.
  The reservation's cancellation is shielded, so a cancellation of the
  creation does not cut it short.
  Otherwise, and after a crash, the collection stays pending until it is
  deleted like any other.
- Whatever a failed or interrupted preparation leaves is recoverable. Shared
  storage is completed by the next preparation: each step is idempotent, so a
  creation that failed part way is completed by the next one as
  if it had never been attempted (the [Qdrant](qdrant_vector_store.md) and
  [Milvus](milvus_vector_store.md) documents say how). Storage of the
  collection's own is reclaimed by its incarnation's purge rounds once the
  pending collection is deleted.
- A collection deleted while its storage is prepared is not marked live, since
  the mark is conditional on its incarnation: the creation raises
  `VectorStoreCollectionDeletedError`.

**`open_or_create_collection`** is read-then-create, retried a second apart: a
live row is opened, or refused with `VectorStoreCollectionConfigMismatchError`
if its configuration differs, as is a pending row of another configuration; a
pending row of the same configuration is another creator's, so the loop waits
for it rather than registering; no row means create; losing the create means a
racing creator took the name; losing the mark means a racing deleter removed
the collection while its storage was prepared, so the loop creates again.
After 10 attempts it raises `VectorStoreCollectionPendingError` if the last
lookup found the collection pending, and `VectorStoreAttemptsExhaustedError`
otherwise. The event backend's service locator composes open and a strict
create itself: up to 10 attempts a second apart, each opening the collection and creating it when there is none;
losing the create, or finding the collection pending, moves to the next
attempt.

The contract tests (`collection_lifecycle_contract.py`) pin both outcomes of a
lost race on Qdrant and Milvus: the loser opens the winner's collection, or
refuses it when its configuration differs, and exactly one reservation is
lost.

### Handles and fencing

A handle is bound to the incarnation it was opened under. After that life is
deleted, every operation on the handle raises
`VectorStoreCollectionHandleStaleError`, and a collection created again under
the same name is a new life the old handle cannot reach.

- An upsert or query calls its registration's `require_current` once its
  inputs are checked and before its remote call; one with nothing to send (no
  records, no query vectors) checks too.
- An upsert checks again after the remote call, so an upsert that completed
  under an incarnation that died meanwhile raises instead of reporting
  success. A delete checks once, after its remote call: it adds nothing a
  purge must reclaim, and a stale handle's delete reaches only its own
  incarnation.
- A handle is given its registration alone.
- A read is not checked afterwards. A deleted collection's records stay until a
  purge round claims its tombstone, so a read in flight when the deletion
  commits returns a snapshot from before it, never a state halfway through a
  deletion: the answer it would have given had it run a moment earlier.
- No lock spans a remote call, and neither store holds a process-local lock.

An upsert can still land under a dead incarnation: between its two checks, or
after a check that never ran because the process died. Nothing can refuse it
at the backend, so the purge reclaims it. That is why a tombstone waits out a
retention before its purge starts.

### Differences from the segment store

The registry reuses the segment store's incarnation logic wherever it
can: the bounded mint loop, the in-transaction locking re-check of the queue,
the idempotent deletion that queues a tombstone, the claim under `FOR UPDATE
SKIP LOCKED`, and the retried read-then-create of open-or-create. It differs
where the backend is remote:

- deletion is `DELETE ... RETURNING` then the queue insert, since there is no
  write fence to pin the row with;
- a handle's upsert is fenced by a registry lookup before and after it (a
  query by one before, a delete by one after), not by a row lock held across
  the write, since the write is not in the database;
- the purge is one tombstone per call, each backend deleting the way it
  measured best;
- the purge claim is a lease, committed before the round, rather than the
  round's own transaction, since the round's work is remote;
- a tombstone waits out a retention, since a remote write can land after the
  deletion.

## Decisions

- **`unregister` is keyed by name.** Callers delete collections by name, and
  the name is resolved to its incarnation inside the deleting transaction.
  Keying it by incarnation alone would make every caller resolve the name
  first, in a separate transaction, and would still need the name-keyed path
  for a caller that holds no handle. A reservation's `cancel` serves the
  creation taking back its own.
- **The registry keeps the namespace** in the queue, because a dead
  incarnation's records are located by its namespace and configuration (on
  Qdrant and Milvus, the native collection they name).
- **`startup` keeps its name.** It creates the tables when missing, which is
  provisioning. Renaming it to `provision`, and taking provisioning out of
  runtime startup, is a change for every store at once, tracked in #1570.

## Alternatives considered

- **Keep the catalog in the backend.** Rejected: neither
  backend can arbitrate a create, a delete, or a claim.
- **Process-local locks.** They serialize one process only; the goal is any
  process serving any collection.
- **Storage first, registry last**, with no pending state. It suits storage a namespace and configuration's collections
  share, which a crash between the two leaves for the next creation to
  adopt, but storage named by a collection's incarnation cannot be made
  before the registry mints the incarnation. With the pending state, one
  preparation step, after registration, serves both; the cost is a pending
  row a crash leaves, deleted like any other.
- **One registry class for both levels**, with `mark_live(incarnation)`,
  `unregister_incarnation(incarnation)` and a lookup answering the state as
  fields. Callers carry incarnations back into the registry, any incarnation
  fits any call, and the state fields go stale; see
  [Reservations and registrations](#reservations-and-registrations).
- **One handle type for every state**, answered by `reserve` and the lookup
  alike. It would offer `confirm` and `cancel` on a live collection, and need
  a state field or an extra read to tell the two apart.
- **Handle types named by state** (`PendingRegistration`, `LiveRegistration`).
  A state name stops being true once someone deletes the collection; a role
  name does not.
- **A liveness method on the registration** (`is_live`) beside the lookup.
  Opening would take two reads instead of one.
- **A lock or coordination service** (etcd, ZooKeeper). It would arbitrate,
  but it would add a stateful dependency for a problem the deployment's
  relational database, which every SQL-backed component already requires,
  solves with a primary key.
- **The incarnation as the only key**, with names resolved elsewhere. Callers
  address collections by name, so the name lookup has to be arbitrated
  somewhere; the primary key on `(namespace, name)` is that place.

## Consequences

- The liveness check costs every operation one primary-key lookup on the
  registry's database (two for an upsert).
