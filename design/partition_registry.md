# Partition registry: one identity and lifecycle for a memory system's tenants

## Problem

A memory system keeps each tenant's records in several stores, and some of
those backends hold records and nothing that can arbitrate between
processes. Qdrant and Milvus have no transactions, no unique constraints, and,
until Qdrant 1.16, no conditional writes. A catalog kept inside such a
backend, with process-local locks around it, cannot decide anything across
processes:

- **Creation races.** Create is a read, then a write. Two processes creating
  the same name both succeed: Qdrant's upsert lets the last writer win, and
  Milvus's `insert` does not enforce primary-key uniqueness, so it keeps two
  live entries.
- **Deletion does not end a partition.** When the name is the tenant
  discriminator on every record, a handle held in one process keeps writing
  into a partition another process deleted and created again under the same
  name (#1563, reproducible sequentially, without any race).
- **Deletion races with writes.** A filter-delete by name is outlived by a
  write in flight during it.

With a catalog in a relational database, each store is safe on its own. But
each store keeps its own catalog of partitions and mints its own incarnation
for each life of a name: the segment store, the registry behind the Qdrant
and Milvus stores, and both SQLite vector stores, each with its own mint
loop. Event memory keeps events, segments, and derivative links in the
segment store and the derivatives' vectors in a vector store, and further
derived indexes (text, a graph) would add more stores. A memory system then
composes its stores by name:

- **Deleting is not one step.** The memory deletes each store's partition in
  turn, so a process that stops between the deletes leaves the tenant half
  deleted, and nothing durable records the rest of the work.
- **Creating again can inherit the old life.** A name created while one
  store's partition of the old life survives opens that partition, records
  and all, beside the other stores' new ones.
- **The stores' orders and checks carry the correctness.** Creation goes
  one way and deletion the other, and background work that opens a store's
  partition by name checks afterwards that the partition it started from
  still exists.
- **Every derived index adds a catalog, a mint loop, and an ordering rule.**

## Design

A tenant's catalog entry lives where a primary key and a transaction can
arbitrate it: a relational database, reached through SQLAlchemy. One
registry per memory system holds the catalog of the memory's tenants. It
mints one incarnation for each life of a partition name, and every store of
the memory holds that life's records under that incarnation. Stores keep no
catalog of their own and mint nothing, and a backend keeps only records, each
carrying the *incarnation* of the life that wrote it.

### Roles

- **The registry** belongs to one *partition namespace*: the partitions of one
  memory system, such as `event_memory`, shared by every process serving it.
  It arbitrates creation, makes deletion one transaction, queues each deleted
  life as a tombstone, and runs purge rounds on due tombstones through a
  callback. It knows partition names, incarnations, and tombstones, and
  nothing about stores. `PartitionRegistry` (`common/partition_registry/`)
  is the ABC, and `SQLAlchemyPartitionRegistry` its one implementation. It
  supports PostgreSQL and SQLite 3.35 or newer (for `RETURNING`), and its
  params refuse any other dialect or an older SQLite.
- **The owner** is the memory system, the concept the registry binds to. It
  holds its stores, prepares each store's partition for a new life, opens
  each store's partition for a live one, fences its operations, and
  supplies the purge callback. It takes the place the registry-backed
  vector store (`common/vector_store/registry_backed_vector_store.py`) holds
  today: that base class of the Qdrant and Milvus stores reserves, prepares,
  confirms or cancels, opens, fences, deletes, and purges, and its
  subclasses supply the backend steps: preparing the storage the store's
  partitions share, at startup; preparing the storage a new partition needs
  of its own; the handle's backend calls (`_upsert`, `_query`, `_delete`);
  and one purge round over an incarnation's records. Here the memory system
  does the first, and its stores supply the steps.
- **A store** holds records under incarnations. It prepares a life's
  storage, opens a handle on a live life, and purges a dead life's records,
  each when its owner calls it. Nothing in the owner assumes how a backend
  lays records out. Qdrant and Milvus keep every partition of a store in its
  one native collection, so a partition needs no storage of its own. A
  backend with a unit per tenant, such as a Pinecone or turbopuffer
  namespace, a Chroma collection, or a Weaviate tenant, makes the partition's
  unit when it prepares the life's storage, which runs after the reservation
  that mints the incarnation naming it, and its purge drops it.
- **`Reservation` and `Registration`** are handles on one life of a
  partition (see [Reservations and
  registrations](#reservations-and-registrations)).
- **Whoever builds the owner owns the registry's lifecycle.** It builds and
  starts the registry and hands it over started, as the database manager
  does for each Qdrant and Milvus store. The owner never starts or stops it:
  an owner that started a registry it was given would own a lifecycle it
  does not control.

Memory systems do not share a partition namespace, and nothing ties one
memory system's registry to another's. The registries of every namespace
share two tables (see [Tables](#tables)), so several memory systems keep
their registries in one database, and two registry objects under one
namespace are one registry. A memory system's namespace is fixed in its code
(`event_memory`, `semantic_memory`), never derived from a configured
resource, and identifies one memory system among all those whose registries
share a database, so deployments that configure a memory system differently
keep their registries in different databases. Stores that still name native
resources, such as a vector store's collection, keep their own name for that
alone.

### Interface

```python
type PurgeRound = Callable[[UUID], Awaitable[bool]]


class PartitionRegistry(ABC):
    async def startup(self) -> None: ...
    async def reserve(self, partition_name: str) -> Reservation: ...
    async def resolve(self, partition_name: str) -> Registration | None: ...
    async def is_live(self, incarnation: UUID) -> bool: ...
    async def unregister(self, partition_name: str) -> None: ...
    async def run_purge_round(self, purge_round: PurgeRound) -> bool: ...


class Reservation:  # partition_name, incarnation
    async def confirm(self) -> Registration: ...
    async def cancel(self) -> None: ...


class Registration:  # partition_name, incarnation
    async def require_current(self) -> None: ...
```

A store's side of the lifecycle:

```python
async def create_partition(self, incarnation: UUID) -> None: ...
def open_partition(self, incarnation: UUID) -> StorePartition: ...
async def purge_partition(self, incarnation: UUID) -> bool: ...
```

`create_partition` prepares the storage a life needs of its own, raises when
it cannot, and is safe to repeat. `open_partition` answers a handle on a
life's records. `purge_partition` deletes a bounded batch of the incarnation's
records, is safe to repeat and to run on two processes at once, and returns
whether it found any. Stores lose `delete_partition` and their own purge:
deletion is the registry's.

### Reservations and registrations

The registry is addressed by partition name. What acts on one life of a
partition is a handle bound to that life's incarnation: the creator's
reservation, or a registration.

| Object | Operations | Answered by |
|---|---|---|
| `PartitionRegistry` | `reserve`, `resolve`, `unregister` (by name), `run_purge_round` | |
| `Reservation` | `confirm`, `cancel` | `reserve` |
| `Registration` | `require_current` | `resolve`, `confirm` |

- **Why handles.** Every operation on one life of a partition needs that
  life's incarnation: confirming it, cancelling it, and checking that it was
  not deleted. As methods of the registry, each would take the incarnation as
  an argument. The registry would then have two addressing schemes, names and
  incarnations, and every caller would carry the UUID `reserve` minted back
  into later calls, where any UUID fits: another partition's, a deleted
  one's, or one compared by hand against a lookup. On a handle, the
  incarnation is bound once, when the registry answers the handle, and no
  call takes it again. Its holder reads it to address the stores' records,
  and never passes it back.
- **Why two types.** What may be done with a partition depends on who holds
  it. Only its creator, holding the reservation, confirms or cancels it; a
  registration is checked, and its partition is deleted only by name. With one
  type, a registration would offer `confirm`, which could only fail, and
  `cancel`, which would delete a live partition outside the name-addressed
  path every deletion goes through. With two, each type offers its holder's
  operations and nothing else, and the wrong call is a type error.
- **Why role names, not state names.** A partition moves from pending to
  live only through its reservation's own `confirm`, but any caller can
  delete it at any time, through either handle's back. A name that states
  the partition's state ("pending", "live") is only true as of when the
  handle was answered; a name that states the holder's role stays true, and
  a deletion shows as the handle's methods raising. The words follow two
  known patterns: Try-Confirm/Cancel for the creator (reserve, then confirm
  or cancel), and the stale handle for everyone else (`require_current`,
  `PartitionHandleStaleError`).
- **Why no state fields.** A handle's fields (partition name and
  incarnation) belong to its life and never change. Whether the partition is
  pending, live, or deleted changes under every holder: a `live` field would
  be a snapshot, made stale by the creator's own `confirm`. The state is
  conveyed instead by the outcome of each call. `reserve` answers a
  reservation, and `confirm` a registration. `resolve` answers a registration
  or `None`, or raises when the partition is pending, and `require_current`
  raises once it is deleted. Each is one read or one write, so an open takes
  one round trip and the fence one.
- **What each part gains.** The registry's operations, beside `startup` and
  `is_live`, are all addressed by name. An owner's handle on a tenant is
  built from one registration, and its fence is one call. The owner's
  creation flow reads as the lifecycle it implements: reserve, prepare
  storage under the reservation's incarnation, then confirm the reservation,
  or cancel it if the preparation fails.

### Names and incarnations

Callers address partitions by name, and only the registry holds names. A
store addresses its partitions by incarnation, so the segment store's
32-byte identifier rule gives way to the registry's own rule: a name is
non-empty, at most 255 bytes, and free of NUL, which PostgreSQL's text type
cannot store. A caller's identifier, such as a session id, is the partition
name as it is.

Deletion is by name: `unregister` removes whatever life holds the name when
it runs. A memory system's lifecycle is its own, tied to no caller's, so a
caller that must delete one particular life names its partitions with ids it
mints for each life, such as a server's session ids, and a name never comes
back.

A setting that belongs to one life is kept by what reads it. No reader of a
per-tenant setting exists beyond the segment store's payload codec, which
the segment store passes and keeps itself, so the registry stores none.

### Tables

The registries of every namespace share two tables, defined once at module
level as the segment store's are, and keyed by `partition_namespace`. One
table pair with a namespace column, rather than a table pair per namespace,
keeps the schema static.

`partition_registry_pt`, the registered partitions, pending or live:

| Column | Type | Role |
|---|---|---|
| `partition_namespace`, `partition_name` | string, primary key | The partition's identity. The primary key arbitrates creation. |
| `incarnation` | UUID, unique | The partition's current life. Its records carry it. |
| `live` | boolean | Whether the partition's storage is prepared. A pending partition holds its name; only a live one is opened. |
| `registered_at` | timestamp | When the partition was registered, on the database clock. An operator finds a partition stuck pending by it, and opening a pending partition reports it. |

`partition_registry_gc`, the purge queue, one row per deleted incarnation
(its *tombstone*; see [purge](vector_store_purge.md)):

| Column | Type | Role |
|---|---|---|
| `incarnation` | UUID, primary key | The deleted life whose records remain. |
| `partition_namespace` | string | Whose purge claims the tombstone. |
| `enqueued_at` | timestamp | When the deletion committed, on the database clock. |
| `attempts_without_progress` | integer | Purge rounds claimed since a round last found records and deleted them, the open one included; a cancelled round's attempt is taken back. |
| `claimable_at` | timestamp, nullable | When a claim may next take the tombstone, on the database clock: once the open claim's lease and the backoff pass, or the backoff after a raise; null when no claim or raise since the last progress set one. |
| `claim_generation` | integer | Incremented by each claim; a round's writes that end its claim are conditioned on it. |

Index `partition_registry_gc__ns_ea` on (`partition_namespace`, `enqueued_at`)
bounds the purge claim, equality before order. The index follows the
repository's naming scheme (table, two letters per column).

The incarnation alone locates a dead life's records, since every store holds
them under it. A tombstone holds only what the purge reads.

### Operations

**`reserve(partition_name) -> Reservation`.** Mints a random
UUID (version 4) and inserts the partition's row, pending. In the same
transaction, *after* the insert, a locking read checks that the incarnation is
not waiting in the purge queue. After the insert, a concurrent deletion that
moved a colliding row to the queue has already committed, because the insert
waited on it. After the check no queue row for this incarnation can appear
before commit, because the only row carrying it is this uncommitted one. A
primary-key violation with a row under the name raises
`PartitionAlreadyExistsError` when the row is live, or pending for less than
`pending_partition_timeout_seconds` (a registry parameter; 300 by default)
on the database clock. A row pending longer is a creation that stopped: one
transaction unregisters it, conditional on its incarnation and on its still
being pending, queues its tombstone, and inserts the new row with the same
queue check, so a confirmation or another replacement that commits first
leaves the name taken. The stopped creator's `confirm` then matches nothing.
A rejected incarnation (unique
violation, or queued) is re-minted, up to 10 attempts, then
`AttemptsExhaustedError`. The loop and its bound are the segment
store's.

**`Reservation.confirm()`** marks the row live with an `UPDATE` conditional on
the reservation's incarnation, which is unique, and on the row being pending,
and answers the `Registration`; it raises `PartitionDeletedError`
when it matches nothing. A creation whose partition was deleted while its
storage was prepared matches nothing, so it cannot confirm a partition
reserved under the name since.

**`resolve(partition_name)`** reads the partition's row once and answers by its
state: a `Registration` when it is live, `PartitionPendingError`, with when
the partition was reserved, when it is pending, and `None` when there is
none. It is how a handle is opened: callers address partitions by name, and
the name is resolved to a live incarnation once, at open.

**`Registration.require_current()`** reads the row under the
registration's name and raises `PartitionHandleStaleError` unless it
carries the registration's incarnation. It is how a handle is fenced (see
[Fencing](#fencing)).

**`is_live(incarnation)`** reads whether a partition name holds that life
live, by the incarnation's unique index: one read, with no handle answered.

**`unregister(partition_name)`** is one transaction: `DELETE ... RETURNING` the
partition's row, pending or live, then insert its tombstone with
`enqueued_at = now()`. The partition is unreachable when it commits.
**`Reservation.cancel()`** does the same for the row carrying the
reservation's incarnation while it is pending: a creation whose storage
preparation raised takes back its own reservation that way, never one made
under the name since, and never its partition once confirmed, which only a
deletion by name ends, even when the confirmation committed but its answer
was lost. Racing deleters serialize on the row's write lock and the loser
deletes nothing, on PostgreSQL and SQLite alike, so a deletion is idempotent
and queues one tombstone.

**`run_purge_round()`** is described in
[purge](vector_store_purge.md) and [Deletion and
purge](#deletion-and-purge).

### Creation

Creation is *reserved, prepared, then confirmed*. The owner reserves the
name, which mints the incarnation and leaves the life pending, calls every
store's `create_partition` at once, and confirms the reservation, which makes
the life live. Every call finishes before the outcome is decided: if any
raised, the owner logs each error, cancels the reservation, and raises the
first, so no store is still preparing storage when the reservation is
cancelled, and the caller sees a store's own exception. No order between
stores matters, since nothing opens the life before it is confirmed.

- The registry's primary key is the one arbiter: a racing creator on any
  process loses at the reservation, never in a backend.
- A pending partition holds its name but is not opened: `open_partition`
  raises `PartitionPendingError`, which says since when it has been pending,
  a `create_partition` of the name raises `PartitionAlreadyExistsError`, and
  open-or-create waits for it. `None` from `open_partition` means only that
  no partition holds the name.
- A preparation or confirmation that raises, or is cancelled, cancels the
  reservation when the registry can, which frees the name and queues the
  incarnation's tombstone; a confirmation that committed before its failure
  was observed stands, since the cancel acts only on a pending partition. The
  reservation's cancellation is shielded, so a cancellation of the creation
  does not cut it short. Otherwise, and after a crash, the partition stays
  pending until it is deleted like any other, or until the next creation of
  its name replaces it once the pending partition timeout has passed.
- A life is confirmed only once every store has prepared its storage, so
  storage that cannot be prepared never goes live. Whatever a failed or
  interrupted preparation leaves is recoverable: the life's own storage is
  reclaimed by its incarnation's purge rounds once it is cancelled or
  deleted. Every creation prepares storage under a new incarnation, so
  nothing deletes and creates storage in place. The storage a store's
  partitions share is prepared at startup, each step idempotent, so a
  startup that failed part way is completed by the next (the
  [Qdrant](qdrant_vector_store.md) and [Milvus](milvus_vector_store.md)
  documents say how).
- A partition deleted while its storage is prepared is not marked live, since
  the mark is conditional on its incarnation: the creation raises
  `PartitionDeletedError`.

Only creation creates. Opening a name reads the registry: a live life is
opened, and a pending one is waited for, a bounded number of times, and
then reported as pending. A creation that stopped part way, with its
process, leaves its life pending; the next creation of the name replaces
it, once it is older than any creation takes. `reserve` does that in one
transaction, conditional on the old life still being pending under its
incarnation (see [Operations](#operations)). The old creator's `confirm`,
conditional on its own incarnation, then fails, so a slow creator never
confirms a life that was replaced, and nothing that went live is ever
replaced. The age decides only when to stop waiting; which life holds the
name is decided by the incarnation.

**`open_or_create_partition`** is read-then-create, retried a second apart: a
live row is opened; a pending row is another creator's, so the loop waits for
it rather than registering; no row means create; losing the create means a
racing creator took the name; losing the mark means a racing deleter removed
the partition while its storage was prepared, so the loop creates again.
After 10 attempts it raises `PartitionPendingError` if the last
lookup found the partition pending, and `AttemptsExhaustedError`
otherwise. The event backend's service locator opens a session's partition
with it.

The contract tests (`partition_lifecycle_contract.py`) pin the outcome of a
lost race on Qdrant and Milvus: the loser opens the winner's partition, and
exactly one reservation is lost.

### Fencing

A handle is bound to the incarnation it was opened under. After that life is
deleted, every operation on the handle raises `PartitionHandleStaleError`,
and a partition created again under the same name is a new life the old
handle cannot reach.

The owner fences each of its operations with the life's registration, as
the registry-backed vector store fences each call: an operation that writes
checks `require_current` before and after it, one that reads checks before
it, and one that deletes checks after it. For event memory, an ingest is the
first, a query the second, and a forget the third.

- The check before an operation runs once its inputs are checked; one with
  nothing to send (no records, no query vectors) checks too.
- The check after a write makes a write that completed under a life that
  died meanwhile raise instead of reporting success.
- A delete checks once, after it runs: it adds nothing a purge must reclaim,
  and a stale handle's delete reaches only its own incarnation.
- A handle is given its registration alone.
- A read is not checked afterwards. A deleted life's records stay until a
  purge round claims its tombstone, so a read in flight when the deletion
  commits returns a snapshot from before it, never a state halfway through a
  deletion: the answer it would have given had it run a moment earlier.
- Stores, SQL ones included, check nothing themselves. No lock spans a
  store's call or transaction, on the registry or in a process.

A write can still land under a dead incarnation: between its two checks, or
after a check that never ran because the process died. No store refuses it
and no handle reads it, so the purge reclaims it, which is why a tombstone
waits out a retention before its purge starts. Every write finishes within a
bound well under the retention, and the bound is the client's, for SQL
stores as for remote ones: a remote store's request timeout, and the SQL
driver's statement timeout. A process paused, or an event loop stalled,
between its check and its send or commit is outside this design, as a
paused remote writer already is.

Background work starts from durable rows that carry an incarnation, and a
deleted life's rows stay until its purge. Event memory's recovery of
pending derivatives is the work that writes: stores check nothing, so a
write it sends for a dead life lands, and one sent while the purge runs can
land after its last round and outlive it. A recovery therefore asks
`is_live` right before it writes, after its budget starts, so its write
lands within the budget and the in-flight time of a moment the life was
alive, as a request's does. A life that is not live is deleted; the
recovery then ends its claim by dropping the row, and the life's purge
reclaims the rest. Work that only deletes, such as a tombstone's deletes,
needs no check: deleting in a dead life is harmless.

### Deletion and purge

`unregister(partition_name)` is one transaction: the name is free and the
life unreachable when it commits, and its tombstone is queued. A creation
under the name mints a new incarnation at once, so a re-created tenant starts
empty in every store.

A tombstone comes due once the namespace's retention has passed since the
deletion. A purge round claims it under a lease, with backoff and
dead-lettering, and calls the owner's callback with the incarnation. The
callback runs every store's `purge_partition` to completion at once, then
raises if any raised, and otherwise answers whether any found records; the
tombstone goes when a round finds nothing in any store. Letting every store
finish its batch keeps the stores that answer purging while another is down;
the round still counts as failed, so the registry's backoff applies. A store
keeps whatever progress its purge needs between rounds, such as the segment
store's per-table cursors, under the incarnation, and drops it when its purge
finds nothing. A namespace's rounds run through its owner's stores: a round
run through stores that cannot reach the records finds none, removes the
tombstone, and leaves the records behind.

A tombstone's lease and backoff are stored as one `claimable_at`, the
earliest time a claim may take it again: a claim sets it to its lease end
plus the backoff for its attempt, a failure to the backoff's end, and
progress clears it. The retention stays its own condition, computed from
`enqueued_at` when the claim runs, so a change to it reaches every queued
tombstone and no retry schedule can shorten it. The claim keeps its one
index range, oldest deletion first, and skips tombstones whose
`claimable_at` has not passed. Every write after a tombstone's insert is a
claim's, matched on `claim_generation`; a transition made by anything that
holds no claim would have to increment the generation, so that an
outstanding claim's late failure matches nothing.

The retention is the namespace's. An owner's configuration names its
registry's database and its retention, as a Qdrant or Milvus store's
configuration names them today, and the retention's floor is checked
against the request timeout of each of the owner's remote stores.

### Event memory on the registry

- Event memory's registry holds its partition namespace. Its stores are the
  segment store and the vector store, each holding a tenant's records under
  the incarnation. The segment store keeps one row per incarnation, with its
  payload codec configuration and, during its purge, its cursors, and no
  name.
- `delete_partition` is `unregister`. The purge callback purges the segment
  store's and the vector store's records of the incarnation.
- Event memory's deployment configuration (the episodic memory's long-term
  memory) names the registry's database and the retention, which the Qdrant
  and Milvus configurations no longer carry.
- The record-level protocol ([ingest and
  deletion](event_memory_ingest_and_deletion.md)) is unchanged: pending
  derivatives and tombstones are keyed by incarnation, and a recovery asks
  `is_live` where its renewal checks the segment store's partition.
- A further derived index is a further store: one more `create_partition`
  call and one more `purge_partition` call, with no schema change.

Semantic memory's vector-store backend keeps the one partition it never
deletes: it opens the name `semantic_memory` in the namespace
`semantic_memory` through the registry, as event memory opens a name, and
opens its vector store's partition by the incarnation. The registry is in
the SQL database semantic memory's vector backend already uses, and the
partition is never deleted, so semantic memory takes no new setting.
Nothing else about semantic memory changes.

### Other owners

The registry knows nothing of what owns a namespace, so any component that
keeps tenants across several stores can own one the same way. A server's
session layer, for one, could mint each session life's identity in its own
namespace and pass it to every memory as the partition name, so a session's
names are never reused and deleting a name always deletes the life it
meant; that namespace's purge round would then be the session's durable
deletion job. This design builds no owner above event memory.

### Differences from the segment store

The registry reuses the segment store's incarnation logic wherever it can: the
bounded mint loop, the in-transaction locking re-check of the queue, the
idempotent deletion that queues a tombstone, the claim under `FOR UPDATE SKIP
LOCKED`, and the retried read-then-create of open-or-create. It differs where
the backend is remote:

- deletion is `DELETE ... RETURNING` then the queue insert, since there is no
  write fence to pin the row with;
- a write is fenced by a registry lookup before and after it (a read by one
  before, a delete by one after), not by a row lock held across the write,
  since the write is not in the registry's database;
- the purge is one tombstone per call, each backend deleting the way it
  measured best;
- the purge claim is a lease, committed before the round, rather than the
  round's own transaction, since the round's work is remote;
- a tombstone waits out a retention, since a remote write can land after the
  deletion.

## Decisions

- **One identity per tenant life, minted by the registry.** Every store
  holds a life's records under it, so no store's partition can outlive or
  predate another's, and no order between stores matters.
- **The registry knows no stores.** The owner composes them, and one
  callback per purge round covers any number. A round that finds nothing in
  any store proves the life empty, so the registry tracks no per-store
  progress.
- **The owner fences each operation by its registration.** One liveness and
  late-write mechanism serves every store, the registry can live in any
  database, and an operation pays one check before and one after its
  writes, whatever number of stores it calls.
- **Deletion is by name.** Callers delete partitions by name, and the name is
  resolved to its incarnation inside the deleting transaction. Keying it by
  incarnation alone would make every caller resolve the name first, in a
  separate transaction, and would still need the name-addressed path for a
  caller that holds no handle; a pending reservation's `cancel` serves the
  creation taking back its own. A memory system's lifecycle is tied to no
  caller's, which keeps it composable; a caller that needs to delete one
  particular life mints a fresh name for each life.
- **Only creation creates.** Opening waits for a pending life and does no
  work on it; a creation that stopped part way is replaced by the next
  creation of the name, in one conditional transaction, after an age no
  creation reaches.
- **Background work asks whether an incarnation is live.** Callers address
  the registry by name, but a recovery starts from a row that holds only
  its incarnation, and liveness is all it needs: `is_live` answers that one
  question, with no handle, so the registry keeps one way to answer
  handles.
- **A store's schema is fixed at construction.** The composition root builds
  the stores its services need, each with its dimensions, metric, and
  declared properties fixed at construction, and partitions each by tenant.
  A partition name is all a caller names; the store, not each call, carries
  what the records share.
- **`startup` creates the registry's tables when missing**, as every store's
  startup creates its own durable resources idempotently.

## Alternatives considered

- **Keep the catalog in the backend.** Rejected: neither Qdrant nor Milvus
  can arbitrate a create, a delete, or a claim.
- **Process-local locks.** They serialize one process only; the goal is any
  process serving any partition.
- **A lock or coordination service** (etcd, ZooKeeper). It would arbitrate,
  but it would add a stateful dependency for a problem the deployment's
  relational database, which every SQL-backed component already requires,
  solves with a primary key.
- **Each store keeps its own catalog.** Each store is safe on its own, and a
  memory spanning several composes them by name, with the problems above.
- **The segment store mints for the other stores.** It is the same design
  one level down: the segment store would take the registry's operations,
  with a purge round that calls back into its owner for the other stores.
  The registry then lives inside one data store, which drives its peers'
  deletion; the remote-safe purge (leases, retention, backoff,
  dead-lettering), which the segment store's single-transaction purge lacks,
  would be built again there; and a memory system without a segment store
  could not use it.
- **The registry runs each store's creation and purge.** Stores would
  register handlers with the registry, which would record each store's
  progress per life. The owner already holds its stores, and a round that
  finds nothing anywhere ends the life, so per-store progress adds state
  nothing reads.
- **Keep the SQL stores' in-transaction fence.** The segment store checks
  liveness in the same statement as each read and holds its catalog row
  under a shared lock while it writes, so a deletion waits out its writes
  and no SQL write lands late. With the registry's tables in the same
  database, a store could keep that fence through a predicate the registry
  supplies. It saves the check after a write and the late SQL write, at
  the cost of a second liveness mechanism and of placing the registry in
  that database.
- **Storage first, registry last**, with no pending state. Storage named by a
  life's incarnation cannot be made before the registry mints the
  incarnation, so a life's own storage is prepared after its reservation,
  and the storage a store's partitions share is prepared at startup. The cost
  is a pending row a crash leaves, deleted like any other.
- **One registry class for both levels**, with `mark_live(incarnation)`,
  `unregister_incarnation(incarnation)` and a lookup answering the state as
  fields. Callers carry incarnations back into the registry, any incarnation
  fits any call, and the state fields go stale; see
  [Reservations and registrations](#reservations-and-registrations).
- **One handle type for every state**, answered by `reserve` and the lookup
  alike. It would offer `confirm` and `cancel` on a live partition, and need a
  state field or an extra read to tell the two apart.
- **Handle types named by state** (`PendingRegistration`, `LiveRegistration`).
  A state name stops being true once someone deletes the partition; a role
  name does not.
- **Opening by a lookup and a separate liveness read.** Opening would take
  two reads instead of one.
- **The incarnation as the only identifier**, with names resolved elsewhere.
  Callers address partitions by name, so the name lookup has to be arbitrated
  somewhere; the primary key on `(partition_namespace, partition_name)` is that
  place.

## Consequences

- An operation reads the registry once, or twice when it writes, however
  many stores it calls; opening a tenant reads it once. Each read is one
  primary-key lookup on the registry's database.
- A deleted tenant's records stay physically until its purge, after the
  retention; they are unreachable from the deletion.
- A store is used through a registry.
- Stores address partitions by incarnation, not by name.
