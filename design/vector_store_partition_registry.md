# Vector store: the partition registry

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

A store is one collection, and a partition is one tenant's records in it,
addressed by a string key. A partition's catalog entry lives where a primary
key and a transaction can arbitrate it: a relational database, reached
through SQLAlchemy. The backend keeps only records, each carrying the
*incarnation* of the partition life that wrote it.

### Roles

- `VectorStorePartitionRegistry` (`common/vector_store/partition_registry/`)
  is the ABC. A registry belongs to one store and is addressed by partition
  key. Its operations are `startup`, `register`, `resolve`, `unregister` and
  `claim_purgeable_incarnation`.
- `PendingRegistration` and `LiveRegistration` are handles on one life of a
  partition (see [Registrations](#registrations)).
- `SQLAlchemyVectorStorePartitionRegistry` is the one implementation. It
  supports PostgreSQL and SQLite 3.35 or newer (for `RETURNING`), as the
  event memory store does, and its params refuse any other dialect or an older
  SQLite.
- `RegistryBackedVectorStore` (`common/vector_store/registry_backed_vector_store.py`)
  is the base of the Qdrant and Milvus stores and makes every registry call
  they make: create, open-or-create, get and delete, the purge claim, and a
  handle's liveness fence. Its handle's `upsert`, `query` and `delete` check
  their inputs and the handle's liveness around the backend call. A subclass
  supplies the backend steps: preparing the storage the store's partitions
  share, at startup; preparing the storage a new partition needs of its own;
  the handle's backend calls (`_upsert`, `_query`, `_delete`); and one purge
  round over an incarnation's records. Nothing in the base assumes how the
  backend lays records out. Qdrant and Milvus keep every partition of a store
  in its one native collection, so a partition needs no storage of its own. A
  backend with a unit per tenant, such as a Pinecone or turbopuffer
  namespace, a Chroma collection or a Weaviate tenant, would make the
  partition's unit when it prepares the partition's storage, which runs after
  the registration that mints the incarnation naming it, and its purge round
  would drop it.
- Every process serving a store must give it the same registry, and a
  registry serves one store. The registries of every store share two tables,
  keyed by the vector store name, so several stores keep their registries in
  one database. A store name therefore identifies one store among all the
  stores whose registries share a database, and whoever names a store keeps
  it so: the event backend names its store per backend and embedder, so one
  embedder's stores on two backends never share a registry. A store reclaims its registry's tombstones through its own
  client: a round run through a client that cannot reach the records finds
  none, removes the tombstone, and leaves the records behind.
- The caller owns the registry's lifecycle. The database manager builds and
  starts each Qdrant and Milvus store's registry before it builds the store,
  and hands the started registry to the store in its params. The store never
  starts or stops it: a store that started a registry it was given would own
  a lifecycle it does not control.

### Registrations

The registry is addressed by partition key. What acts on one life of a
partition is a registration: a handle bound to that life's incarnation.

| Object | Operations | Answered by |
|---|---|---|
| `VectorStorePartitionRegistry` | `register`, `resolve`, `unregister` (by key), `claim_purgeable_incarnation` | |
| `PendingRegistration` | `mark_live`, `unregister` (this life only) | `register` |
| `LiveRegistration` | `require_current` | `resolve`, `mark_live` |

- **Why handles.** Every operation on one life of a partition needs that
  life's incarnation: marking it live, abandoning it, and checking that it
  was not deleted. As methods of the registry, each would take the
  incarnation as an argument. The registry would then have two addressing
  schemes, keys and incarnations, and every caller would carry the UUID
  `register` minted back into later calls, where any UUID fits: another
  partition's, a deleted one's, or one compared by hand against a lookup.
  On a handle, the incarnation is bound once, when the registry answers the
  handle, and no call takes it again. The store reads it to write records,
  and never passes it back.
- **Why two types.** What may be done to a partition depends on its state.
  Only its creator marks a pending partition live or abandons it; a live
  partition is checked, and is deleted only by key. With one type, a live
  registration would offer `mark_live`, which could only fail, and
  `unregister`, which would delete a live partition outside the key-addressed
  path every deletion goes through. Split by state, each type offers what its
  state allows and nothing else, and the wrong call is a type error.
- **Why no state fields.** A registration's fields (partition key, schema,
  incarnation) belong to its life and never change. Whether the partition is
  pending, live or deleted changes under every holder: a `live` field would
  be a snapshot, made stale by the creator's own `mark_live`. The state is
  conveyed instead by the outcome of each call. `register` answers a pending
  registration, and `mark_live` a live one. `resolve` answers a live one or
  `None`, or raises when the partition is pending, and `require_current`
  raises once it is deleted. Each is one read or one write, so an open takes
  one round trip and the fence one.
- **What each part gains.** The registry's operations, beside `startup`, are
  all addressed by key. A store's partition handle is built from one live
  registration, and its fence is one call. The store's creation flow reads as
  the lifecycle it implements: register, prepare storage under the
  registration's incarnation, then mark it live, or unregister it if the
  preparation fails.

### Tables

The registries of every vector store share two tables, defined once at module
level as the event memory store's are, and keyed by `vector_store_name`. Two
registry objects under one name are one registry. One table pair with a key
column, rather than a table pair per vector store, keeps the schema static.

`partition_registry_pt`, the registered partitions, pending or live:

| Column | Type | Role |
|---|---|---|
| `vector_store_name`, `partition_key` | string, primary key | The partition's identity. The primary key arbitrates creation. |
| `incarnation` | UUID, unique | The partition's current life. Its records carry it. |
| `schema` | JSON (JSONB on PostgreSQL) | The dimensions and declared schema the partition was created under. A store built with others refuses the partition. |
| `live` | boolean | Whether the partition's storage is prepared. A pending partition holds its key; only a live one is opened. |
| `registered_at` | timestamp | When the partition was registered, on the database clock. An operator finds a partition stuck pending by it, and opening a pending partition reports it. |

`partition_registry_gc`, the purge queue, one row per deleted incarnation
(its *tombstone*; see [purge](vector_store_purge.md)):

| Column | Type | Role |
|---|---|---|
| `incarnation` | UUID, primary key | The deleted life whose records remain. |
| `vector_store_name` | string | Whose purge claims the tombstone. |
| `partition_key` | string | Kept for inspection; the purge does not read it. |
| `enqueued_at` | timestamp | When the deletion committed, on the database clock. |
| `failed_rounds` | integer | Consecutive purge rounds on the tombstone that raised. |
| `last_failed_at` | timestamp, nullable | When the last of them raised, on the database clock. |

Index `partition_registry_gc__vs_ea` on (`vector_store_name`, `enqueued_at`)
bounds the purge claim, equality before order. The index follows the
repository's naming scheme (table, two letters per column).

The incarnation alone locates a dead partition's records, since every
partition of a store is in its one native collection. The queue carries the
partition key so an operator looking at a tombstone, a dead-lettered one above
all, can tell which partition it was, as the event memory store's queue does.

### Operations

**`register(partition_key, schema) -> PendingRegistration`.** Mints a random
UUID (version 4) and inserts the partition's row, pending. In the same
transaction, *after* the insert, a locking read checks that the incarnation is
not waiting in the purge queue. After the insert, a concurrent deletion that
moved a colliding row to the queue has already committed, because the insert
waited on it. After the check no queue row for this incarnation can appear
before commit, because the only row carrying it is this uncommitted one. A
primary-key violation with a row, pending or live, under the key raises
`VectorStorePartitionAlreadyExistsError`. A rejected incarnation (unique
violation, or queued) is re-minted, up to 10 attempts, then
`VectorStoreAttemptsExhaustedError`. The loop and its bound are the segment
store's.

**`PendingRegistration.mark_live()`** marks the row live with an `UPDATE`
conditional on the registration's incarnation, which is unique, and on the row
being pending, and answers the `LiveRegistration`; it raises
`VectorStorePartitionDeletedError` when it matches nothing. A creation whose
partition was deleted while its storage was prepared matches nothing, so it
cannot mark live a partition registered under the key since.

**`resolve(partition_key)`** reads the partition's row once and answers by its
state: a `LiveRegistration` when it is live,
`VectorStorePartitionPendingError`, with when the partition was registered and
its schema, when it is pending, and `None` when there is none. It is how a
handle is opened: callers address partitions by key, and the key is resolved
to a live incarnation once, at open.

**`LiveRegistration.require_current()`** reads the row under the
registration's key and raises `VectorStorePartitionHandleStaleError` unless it
carries the registration's incarnation. It is how a handle is fenced (below).

**`unregister(partition_key)`** is one transaction: `DELETE ... RETURNING` the
partition's row, pending or live, then insert its tombstone with
`enqueued_at = now()`. The partition is unreachable when it commits.
**`PendingRegistration.unregister()`** does the same for the row carrying the
registration's incarnation: a creation whose storage preparation raised takes
back its own registration that way, never one registered under the key since.
Racing deleters serialize on the row's write lock and the loser deletes
nothing, on PostgreSQL and SQLite alike, so a deletion is idempotent and
queues one tombstone.

**`claim_purgeable_incarnation()`** is described in
[purge](vector_store_purge.md).

### Creation

Creation is *registered pending, prepared, then live*. The store registers the
partition, pending, prepares the storage it needs of its own (on Qdrant and
Milvus, none), and marks it live:

- The registry's primary key is the one arbiter: a racing creator on any
  process loses at the insert, never in the backend.
- A pending partition holds its key but is not opened: `get_partition` raises
  `VectorStorePartitionPendingError`, which says since when it has been
  pending, a `create_partition` of the key raises
  `VectorStorePartitionAlreadyExistsError`, and open-or-create waits for it.
  `None` from `get_partition` means only that no partition holds the key.
- A preparation that raises, or is cancelled, unregisters the pending
  partition when the registry can, which frees the key and queues the
  incarnation's tombstone. The unregistration is shielded, so a cancellation
  of the creation does not cut it short. Otherwise, and after a crash, the
  partition stays pending until it is deleted like any other.
- Whatever a failed or interrupted preparation leaves is recoverable: the
  partition's own storage is reclaimed by its incarnation's purge rounds once
  the pending partition is deleted. The storage a store's partitions share is
  prepared at startup, each step idempotent, so a startup that failed part way
  is completed by the next (the [Qdrant](qdrant_vector_store.md) and
  [Milvus](milvus_vector_store.md) documents say how).
- A partition deleted while its storage is prepared is not marked live, since
  the mark is conditional on its incarnation: the creation raises
  `VectorStorePartitionDeletedError`.
- A partition created under another schema than the store's, pending or
  live, is refused with `VectorStorePartitionSchemaMismatchError`, from
  `get_partition` and open-or-create, and from a `create_partition` whose key
  it holds.

**`open_or_create_partition`** is read-then-create, retried a second apart: a
live row is opened; a pending row is another creator's, so the loop waits for
it rather than registering; no row means create; losing the create means a
racing creator took the key; losing the mark means a racing deleter removed
the partition while its storage was prepared, so the loop creates again.
After 10 attempts it raises `VectorStorePartitionPendingError` if the last
lookup found the partition pending, and `VectorStoreAttemptsExhaustedError`
otherwise. The event backend's service locator opens a session's partition
with it.

The contract tests (`partition_lifecycle_contract.py`) pin both outcomes of a
lost race on Qdrant and Milvus: the loser opens the winner's partition, or
refuses it when its schema differs, and exactly one registration is lost.

### Handles and fencing

A handle is bound to the incarnation it was opened under. After that life is
deleted, every operation on the handle raises
`VectorStorePartitionHandleStaleError`, and a partition created again under
the same key is a new life the old handle cannot reach.

- An upsert or query calls its registration's `require_current` once its
  inputs are checked and before its remote call; one with nothing to send (no
  records, no query vectors, a limit of 0) checks too.
- An upsert checks again after the remote call, so an upsert that completed
  under an incarnation that died meanwhile raises instead of reporting
  success. A delete checks once, after its remote call: it adds nothing a
  purge must reclaim, and a stale handle's delete reaches only its own
  incarnation.
- A handle is given its live registration alone.
- A read is not checked afterwards. A deleted partition's records stay until
  a purge round claims its tombstone, so a read in flight when the deletion
  commits returns a snapshot from before it, never a state halfway through a
  deletion: the answer it would have given had it run a moment earlier.
- No lock spans a remote call, and neither store holds a process-local lock.

An upsert can still land under a dead incarnation: between its two checks, or
after a check that never ran because the process died. Nothing can refuse it
at the backend, so the purge reclaims it. That is why a tombstone waits out a
retention before its purge starts.

### Differences from the event memory store

The registry reuses the event memory store's incarnation logic wherever it can: the
bounded mint loop, the in-transaction locking re-check of the queue, the
idempotent deletion that queues a tombstone, the claim under `FOR UPDATE SKIP
LOCKED`, and the retried read-then-create of open-or-create. It differs where
the backend is remote:

- deletion is `DELETE ... RETURNING` then the queue insert, since there is no
  write fence to pin the row with;
- a handle's upsert is fenced by a registry lookup before and after it (a
  query by one before, a delete by one after), not by a row lock held across
  the write, since the write is not in the database;
- the purge is one tombstone per call, each backend deleting the way it
  measured best;
- a tombstone waits out a retention, since a remote write can land after the
  deletion.

## Decisions

- **A store is one collection.** The composition root builds one store per
  collection it needs, with its dimensions and declared schema fixed at
  construction, and partitions it by tenant. A partition key is all a
  caller names; the store, not each call, carries what the records share.
- **`unregister` is keyed by partition key.** Callers delete partitions by
  key, and the key is resolved to its incarnation inside the deleting
  transaction. Keying it by incarnation alone would make every caller resolve
  the key first, in a separate transaction, and would still need the
  key-addressed path for a caller that holds no handle. A pending
  registration's `unregister` serves the creation taking back its own.
- **`startup` creates the registry's tables when missing**, as every store's
  startup creates its own durable resources idempotently.

## Alternatives considered

- **Keep the catalog in the backend.** Rejected: neither backend can arbitrate
  a create, a delete or a claim.
- **Process-local locks.** They serialize one process only; the goal is any
  process serving any partition.
- **Storage first, registry last**, with no pending state. It suits storage a
  store's partitions share, which a crash between the two leaves for the next
  creation to adopt, but storage named by a partition's incarnation cannot be
  made before the registry mints the incarnation. With the pending state, one
  preparation step, after registration, serves both; the cost is a pending row
  a crash leaves, deleted like any other.
- **One registry class for both levels**, with `mark_live(incarnation)`,
  `unregister_incarnation(incarnation)` and a lookup answering the state as
  fields. Callers carry incarnations back into the registry, any incarnation
  fits any call, and the state fields go stale; see
  [Registrations](#registrations).
- **One registration type for every state**, answered by `register` and the
  lookup alike. It would offer `mark_live` and `unregister` on a live
  partition, and need a state field or an extra read to tell the two apart.
- **A liveness method on the registration** (`is_live`) beside the lookup.
  Opening would take two reads instead of one.
- **A lock or coordination service** (etcd, ZooKeeper). It would arbitrate,
  but it would add a stateful dependency for a problem the deployment's
  relational database, which every SQL-backed component already requires,
  solves with a primary key.
- **The incarnation as the only key**, with keys resolved elsewhere. Callers
  address partitions by key, so the key lookup has to be arbitrated somewhere;
  the primary key on `(vector_store_name, partition_key)` is that place.

## Consequences

- The liveness check costs every operation one primary-key lookup on the
  registry's database (two for an upsert).
