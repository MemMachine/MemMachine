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
each store keeps its own catalog of partitions and mints its own identifier
for each partition created under a name: the segment store, the registry
behind the Qdrant and Milvus stores, and both SQLite vector stores, each with
its own mint loop. Event memory keeps events, segments, and derivative links
in the segment store and the derivatives' vectors in a vector store, and
further derived indexes (text, a graph) would add more stores. A memory
system then composes its stores by name:

- **Deleting is not one step.** The memory deletes each store's partition in
  turn, so a process that stops between the deletes leaves the tenant half
  deleted, and nothing durable records the rest of the work.
- **Creating again can inherit the old partition.** A name created again
  while one store's old partition survives opens that partition, records and
  all, beside the other stores' new ones.
- **The order of the stores' steps carries the correctness.** Creation
  makes the vector store's partition before the segment store's, so every
  segment store partition has one, and a deletion that stops between its
  steps leaves the stores out of step.
- **Every derived index adds a catalog, a mint loop, and an ordering rule.**

## Design

A tenant's catalog entry lives where a primary key and a transaction can
arbitrate it: a relational database, reached through SQLAlchemy. One
registry per memory system holds the catalog of the memory's tenants. A
partition has a *name*, which the memory chooses and may reuse, and a *UUID*,
which the registry mints for that partition alone; deleting a name and
creating it again makes a new partition with a new UUID. Every store of the
memory holds a partition's records under its UUID. Stores keep no catalog of
their own and mint nothing, and a backend keeps only records, each carrying
the UUID of the partition that wrote it. Partitions are isolated from one
another, including partitions created under the same name: records written
to one are never read from another, so a name created again starts empty in
every store.

The registry also drives each partition's lifecycle in the namespace's
store: it creates the partition in the store before confirming it, answers
the store's handle on a live partition, and once the partition is deleted,
its purge rounds purge it from the store and then drop it. Every store
implements one interface for those steps, so the order of the steps is
written once, in the registry, for every memory system. The registry
addresses partitions by name as the store addresses them by UUID. A memory
system whose records span several stores composes them into one store, which
runs each step on the stores it holds.

### Roles

- **The registry** belongs to one *partition namespace*: the partitions of one
  memory system, such as `event_memory`, shared by every process serving it,
  and is built over the store that holds the namespace's partitions. It
  arbitrates creation, creates each partition in the store, answers the
  store's handle on a live partition with its registration, makes deletion one
  transaction, queues each deleted partition as a tombstone, and runs purge
  rounds on due tombstones, which purge and then drop the partition in the
  store. It knows partition names, UUIDs, and tombstones, and of the store
  only the `PartitionedStore` interface. `PartitionRegistry`
  (`common/partition_registry/`) is the ABC; its creation, lookup, and purge
  rounds are written in it, over the catalog steps an implementation supplies,
  and `SQLAlchemyPartitionRegistry` is its one implementation. It supports
  PostgreSQL and SQLite 3.35 or newer (for `RETURNING`), and its params refuse
  any other dialect or an older SQLite.
- **The owner** is the memory system, the concept the registry binds to. It
  holds its store and its registry, creates, gets, and deletes partitions by
  name through the registry, builds its handle on a live partition from the
  partition's registration, which carries the store's handle, and fences its
  operations.
- **A store** is a `PartitionedStore`: it holds records under partition UUIDs,
  creates the storage a partition keeps for itself, gets a handle on a
  partition, purges a deleted partition's records from storage it shares
  with other partitions, and drops the partition's own storage, each when the
  registry calls it. Nothing in the registry assumes how a backend lays
  records out. Qdrant and Milvus keep every
  partition of a store in its one native collection, so a partition has no
  storage of its own to create or drop. A backend with a unit per tenant,
  such as a Pinecone or turbopuffer namespace, a Chroma collection, or a
  Weaviate tenant, makes the partition's unit when it creates the
  partition, which runs after the reservation that mints the UUID naming it,
  and removes it when it drops the partition.
- **A composite store** is a `PartitionedStore` over several stores, for a
  memory system whose records span them, such as `EventMemoryStore` over
  event memory's segment store and vector store. It creates, purges, and
  drops a partition in every store it holds, and its handle on a partition
  holds every store's handle.
- **`Reservation` and `Registration`** are handles on one partition (see
  [Reservations and registrations](#reservations-and-registrations)).
- **Whoever builds the registry owns its lifecycle.** It builds the registry
  over the owner's store, starts it, runs its purge rounds, and hands it over
  started to whoever builds the owner. The resource manager does this for
  event memory and semantic memory: it answers every request with a registry
  over the store the request names, and starts a namespace's registry, with
  its purge rounds, on the first request for it. The owner never starts or
  stops it: an owner that started a registry it was given would own a
  lifecycle it does not control.

Memory systems do not share a partition namespace, and nothing ties one
memory system's registry to another's. The registries of every namespace
share two tables (see [Tables](#tables)), so several memory systems keep
their registries in one database, and two registry objects under one
namespace on one database are one registry. A memory system's namespace is
fixed in its code (`event_memory`, `semantic_memory`), never derived from a
configured resource, and identifies one memory system among all those whose
registries share a database, so deployments that configure a memory system
differently keep their registries in different databases. Stores that name
native resources, such as a vector store's collection, keep their own name
for that alone.

### Interface

```python
class PartitionRegistry[PartitionT](ABC):
    def __init__(self, partitioned_store: PartitionedStore[PartitionT]) -> None: ...
    @property
    def partition_namespace(self) -> str: ...
    async def startup(self) -> None: ...
    async def create_partition(
        self, partition_name: str
    ) -> Registration[PartitionT]: ...
    async def get_partition(
        self, partition_name: str
    ) -> Registration[PartitionT] | None: ...
    async def delete_partition(self, partition_name: str) -> None: ...
    async def run_purge_round(self) -> bool: ...

    # Supplied by an implementation.
    async def _reserve(self, partition_name: str) -> Reservation: ...
    async def _resolve(self, partition_name: str) -> UUID | None: ...
    def _registration(
        self, partition_name: str, partition_uuid: UUID, partition: PartitionT
    ) -> Registration[PartitionT]: ...
    async def _unregister(self, partition_name: str) -> None: ...
    async def _run_purge_round(self, purge_round: PurgeRound) -> bool: ...


class Reservation:  # partition_name, partition_uuid
    async def confirm(self) -> None: ...
    async def cancel(self) -> None: ...


class Registration[PartitionT]:  # partition_name, partition_uuid, partition
    async def require_current(self) -> None: ...


class PartitionedStore[PartitionT](ABC):
    async def create_partition(self, partition_uuid: UUID) -> None: ...
    async def get_partition(self, partition_uuid: UUID) -> PartitionT | None: ...
    async def purge_partition(self, partition_uuid: UUID) -> bool: ...
    async def drop_partition(self, partition_uuid: UUID) -> None: ...
```

A store's side of the lifecycle is addressed by a partition's UUID, since
stores hold no names. `create_partition` creates the storage a partition
keeps for itself, raises when it cannot, and is safe to repeat.
`get_partition` answers a handle on the partition, typed for the store,
which reads and writes the partition's records alone and checks nothing
about whether the partition is live; it answers `None` when the store finds
nothing under the UUID, and a store that cannot tell, such as a vector
store, answers a handle.
`purge_partition` deletes some or all of a deleted partition's records from
storage it shares with other partitions, is safe to repeat and to run on two
processes at once, and returns False only when none remains there.
`drop_partition` removes the storage the partition keeps for itself, with
any records still in it, and is idempotent. A store that keeps a
partition's records only in storage of its own finds nothing to purge and
removes them when it drops the partition.

Deletion is the registry's, and a store deletes a partition's storage only in
its purge rounds, after the retention, so a write still in flight cannot land
after the deletion and outlive it, even on a backend that creates storage on
first write.

### Reservations and registrations

The registry is addressed by partition name. What acts on one partition is a
handle bound to its UUID: the reservation the registry's creation holds, or a
registration.

| Object | Operations | Answered by |
|---|---|---|
| `PartitionRegistry` | `create_partition`, `get_partition`, `delete_partition` (by name), `run_purge_round` | |
| `Reservation` | `confirm`, `cancel` | `_reserve`, within `create_partition` |
| `Registration` | `require_current` | `create_partition`, `get_partition` |

- **Why handles.** Every operation on one partition needs its UUID:
  confirming it, cancelling it, and checking that it was not deleted. As
  methods of the registry, each would take the UUID as an argument. The
  registry would then have two addressing schemes, names and UUIDs, and every
  caller would carry the UUID the reservation minted back into later calls,
  where any UUID fits: another partition's, a deleted one's, or one compared
  by hand against a lookup. On a handle, the UUID is bound once, when the
  registry answers the handle, and no call takes it again. A registration
  also carries the store's handle on the partition, so its holder reaches the
  partition's records without passing the UUID anywhere.
- **Why two types.** What may be done with a partition depends on who holds
  it. Only its creation, holding the reservation, confirms or cancels it; a
  registration is checked, and its partition is deleted only by name. With one
  type, a registration would offer `confirm`, which could only fail, and
  `cancel`, which would delete a live partition outside the name-addressed
  path every deletion goes through. With two, each type offers its holder's
  operations and nothing else, and the wrong call is a type error.
- **Why the reservation stays inside the registry.** Creation is reserve,
  create in the store, then confirm, or cancel when a step fails. A caller
  that held the reservation could confirm a partition the store never created,
  or skip the cancel; with the steps in `create_partition`, every creation
  takes them in order.
- **Why role names, not state names.** A partition moves from pending to
  live only through its reservation's own `confirm`, but any caller can
  delete it at any time, through either handle's back. A name that states
  the partition's state ("pending", "live") is only true as of when the
  handle was answered; a name that states the holder's role stays true, and
  a deletion shows as the handle's methods raising. The words follow two
  known patterns: Try-Confirm/Cancel for the creation (reserve, then confirm
  or cancel), and the stale handle for everyone else (`require_current`,
  `PartitionDeletedError`).
- **Why no state fields.** A handle's fields (partition name and UUID, and a
  registration's store handle) belong to its partition and never change.
  Whether the partition is pending, live, or deleted changes under every
  holder: a `live` field would be a snapshot, made stale by the creation's own
  `confirm`. The state is conveyed instead by the outcome of each call.
  `create_partition` answers a registration. `get_partition` answers a
  registration or `None`, or raises when the partition is pending, and
  `require_current` raises once it is deleted. `get_partition` and
  `require_current` each read the registry once, so a get takes one round
  trip to the registry and the fence one.
- **What each part gains.** The registry's public operations on a partition
  are all addressed by name. An owner's handle on a tenant is built from one
  registration, and its fence is one call.

### Names and UUIDs

Callers address partitions by name, and only the registry holds names; a
store addresses its partitions by UUID. The registry's by-name operations
refuse an invalid name with `ValueError` before reading or writing anything,
so no backend sees one. A caller's identifier, such as a session id, is the
partition name as it is.

Deletion is by name: `delete_partition` removes whichever partition holds the
name when it runs. A memory system's lifecycle is its own, tied to no
caller's, so a caller that must delete one particular partition names its
partitions with ids it mints for each, such as a server's session ids, and a
name never comes back.

A setting that belongs to one partition is kept by the store that reads it,
and the registry stores none: the segment store records its payload codec
with each partition it creates.

### Tables

The registries of every namespace share two tables, defined once at module
level as the segment store's are, and keyed by `partition_namespace`. One
table pair with a namespace column, rather than a table pair per namespace,
keeps the schema static.

`partition_registry_pt`, the registered partitions, pending or live:

| Column | Type | Role |
|---|---|---|
| `partition_namespace`, `partition_name` | string, primary key | The name's place. The primary key arbitrates creation. |
| `partition_uuid` | UUID, unique | The partition registered under the name. Its records carry it. |
| `live` | boolean | Whether the store has created the partition. A pending partition holds its name; only a live one is answered. |
| `registered_at` | timestamp | When the partition was registered, on the database clock. An operator finds a partition stuck pending by it, and getting a pending partition reports it. |

`partition_registry_gc`, the purge queue, one row per deleted partition (its
*tombstone*; see [purge](vector_store_purge.md)):

| Column | Type | Role |
|---|---|---|
| `partition_uuid` | UUID, primary key | The deleted partition whose records remain. |
| `partition_namespace` | string | Whose purge claims the tombstone. |
| `enqueued_at` | timestamp | When the deletion committed, on the database clock. |
| `attempts_without_progress` | integer | Purge rounds claimed since a round last found records and deleted them, the open one included; a cancelled round's attempt is taken back. |
| `claimable_at` | timestamp, nullable | When a claim may next take the tombstone, on the database clock: once the open claim's lease and the backoff pass, or the backoff after a raise; null until a claim or raise sets one, and again after progress or a cancelled round. |
| `claim_generation` | integer | Incremented by each claim; a round's writes that end its claim are conditioned on it. |

Index `partition_registry_gc__ns_ea` on (`partition_namespace`, `enqueued_at`)
bounds the purge claim, equality before order.

A deleted partition's UUID alone locates its records, since the store holds
them under it. A tombstone holds only what the purge reads.

### Operations

**`_reserve(partition_name) -> Reservation`.** Mints a random UUID (version 4)
and inserts the partition's row, pending. In the same transaction, *after*
the insert, a locking read checks that the UUID is not waiting in the purge
queue. After the insert, a concurrent deletion that moved a colliding row to
the queue has already committed, because the insert waited on it. After the
check no queue row for this UUID can appear before commit, because the only
row carrying it is this uncommitted one. A primary-key violation with a row
under the name, live or pending, raises `PartitionAlreadyExistsError`. A
rejected UUID (unique violation, or queued) is re-minted a bounded number of
times, then `PartitionCreateAttemptsExhaustedError`.

**`Reservation.confirm()`** marks the row live with an `UPDATE` conditional on
the reservation's UUID, which is unique, and on the row being pending; it
raises `PartitionDeletedError` when it matches nothing. A creation whose
partition was deleted while the store created it matches nothing, so it cannot
confirm a partition reserved under the name since.

**`get_partition(partition_name)`** reads the partition's row once
(`_resolve`) and answers by its state: `PartitionPendingError`, with when the
partition was reserved, when it is pending, and `None` when there is none.
For a live partition it gets the store's handle on the partition's UUID and
answers a `Registration` carrying it, or raises `PartitionDeletedError` when
the store finds nothing under the UUID: the partition was deleted and
dropped after the read. It is how an owner gets a handle: callers address
partitions by name, and the name is resolved to a live partition's UUID
once, when the handle is built.

**`Registration.require_current()`** reads the row under the registration's
name and raises `PartitionDeletedError` unless it carries the registration's
UUID. It is how a handle is fenced (see [Fencing](#fencing)).

**`delete_partition(partition_name)`** is one transaction (`_unregister`):
`DELETE ... RETURNING` the partition's row, pending or live, then insert its
tombstone with `enqueued_at = now()`. The partition is unreachable when it
commits. **`Reservation.cancel()`** does the same for the row carrying the
reservation's UUID while it is pending: a creation whose store step raised
takes back its own reservation that way, never one made under the name since,
and never its partition once confirmed, which only a deletion by name ends,
even when the confirmation committed but its answer was lost. Racing deleters
serialize on the row's write lock and the loser deletes nothing, on PostgreSQL
and SQLite alike, so a deletion is idempotent and queues one tombstone.

**`run_purge_round()`** is described in [Deletion and
purge](#deletion-and-purge) and [purge](vector_store_purge.md).

### Creation

`create_partition(partition_name)` is *reserved, created, then confirmed*. It
reserves the name, which mints the UUID and leaves the partition pending,
calls the store's `create_partition`, confirms the reservation, which makes
the partition live, and answers a registration carrying the store's handle on
the partition. When the store's call raises, the registry cancels the
reservation and raises the store's own exception. Nothing gets the partition
before it is confirmed.

- The registry's primary key is the one arbiter: a racing creator on any
  process loses at the reservation, never in a backend.
- A pending partition holds its name but is not answered: `get_partition`
  raises `PartitionPendingError`, which says since when it has been pending,
  and a creation under the name raises `PartitionAlreadyExistsError`. `None`
  from `get_partition` means only that no partition holds the name.
- A store step or confirmation that raises, or is cancelled, cancels the
  reservation when the registry can, which frees the name and queues the
  partition's tombstone; a confirmation that committed before its failure
  was observed stands, since the cancel acts only on a pending partition. The
  reservation's cancellation is shielded, so a cancellation of the creation
  does not cut it short. Otherwise, and after a crash, the partition stays
  pending until it is deleted like any other.
- A partition is confirmed only once the store has created it, so storage
  that cannot be created never goes live. Whatever a failed or interrupted
  creation leaves is recoverable: the partition's purge rounds reclaim its
  storage once it is cancelled or deleted. Every creation makes a new
  partition under a new UUID, so nothing deletes and creates storage in
  place. The storage a store's partitions share is prepared at startup, each
  step idempotent, so a startup that failed part way is completed by the
  next (the [Qdrant](qdrant_vector_store.md) and
  [Milvus](milvus_vector_store.md) documents say how).
- A partition deleted while the store creates it is not marked live, since
  the mark is conditional on its UUID: the creation raises
  `PartitionDeletedError`.

Only creation creates. Getting a name reads the registry: a live partition is
answered, and a pending one is reported as pending, which a caller that gets
or creates waits out a bounded number of times. A creation that stopped part
way, with its process, leaves its partition pending until the name is deleted.
Whether and when to give up on a creation is its caller's decision: the caller
deletes the name and creates it again, and the stopped creation's `confirm`,
conditional on its own UUID, then fails.

A caller that gets a name and creates the partition when none holds it
composes the two itself, as read-then-create retried a bounded number of
times: a live partition is answered; a pending one is another creator's, so
the caller waits for it rather than creating; no partition means create; a
creation that finds the name taken lost it to a racing creator; one that
finds the partition deleted lost it to a racing deleter while the store
created it, so the caller tries again. Once the attempts run out it raises
`PartitionPendingError` if the last lookup found the partition pending, and
`PartitionCreateAttemptsExhaustedError` otherwise. The server's service
locator gets a session's event memory partition this way, and semantic
memory's resource manager its one partition.

### Fencing

A handle is bound to the partition whose registration it was built from. After
that partition is deleted, every operation on the handle raises
`PartitionDeletedError`, and a partition created again under the same name is
a new partition the old handle cannot reach.

The owner fences each of its operations with the partition's registration:
an operation that writes checks `require_current` before and after it, one
that reads checks before it, and one that deletes checks after it. For event
memory, an ingest is the first, a query the second, and a forget the third.

- The check before an operation runs once its inputs are checked; one with
  nothing to send (no records, no query vectors) checks too.
- The check after a write makes a write that completed to a partition deleted
  meanwhile raise instead of reporting success.
- A delete checks once, after it runs: it adds nothing a purge must reclaim,
  and a stale handle's delete reaches only its own partition.
- A handle reaches the registry through its registration alone.
- A read is not checked afterwards. A deleted partition's records stay until
  a purge round claims its tombstone, so a read in flight when the deletion
  commits returns a snapshot from before it, never a state halfway through a
  deletion: the answer it would have given had it run a moment earlier.
- Stores, SQL ones included, check nothing themselves. No lock spans a
  store's call or transaction, on the registry or in a process.

A write can still land in a deleted partition: between its two checks, or
after a check that never ran because the process died. No store refuses it
and no handle reads it, so the purge reclaims it, which is why a tombstone
waits out a retention before its purge starts. Every write finishes within a
bound well under the retention, and the bound is the client's, for SQL
stores as for remote ones: a remote store's request timeout, and the SQL
driver's statement timeout. A process paused, or an event loop stalled,
between its check and its send or commit is outside this design, as a
paused remote writer is.

### Deletion and purge

`delete_partition(partition_name)` is one transaction: the name is free and
the partition unreachable when it commits, and its tombstone is queued. A
creation under the name mints a new UUID at once, so a re-created tenant
starts empty.

A tombstone comes due once the namespace's retention has passed since the
deletion. A purge round claims it under a lease, with backoff and
dead-lettering, and then works through the store in two steps. It runs the
store's `purge_partition`; while that finds records, the round ends there and
the tombstone stays due. Once it finds none, the round runs the store's
`drop_partition` and answers that it found nothing, and the registry deletes
the tombstone. A step that raises fails the round, so the registry's backoff
applies, and a later round purges and drops again. A store keeps whatever
progress its purge needs between calls, such as the segment store's cursor,
in the partition's own storage, which the drop removes. A namespace's rounds
run through its registry's store: a round run through a store that cannot
reach the records finds none, removes the tombstone, and leaves the records
behind.

A tombstone's lease and backoff are stored as one `claimable_at`, the
earliest time a claim may take it again: a claim sets it to its lease end
plus the backoff for its attempt, a failure to the backoff's end, and
progress or a cancelled round clears it. The retention stays its own
condition, computed from `enqueued_at` when the claim runs, so a change to
it reaches every queued tombstone and no retry schedule can shorten it. The
claim keeps its one index range, oldest deletion first, and skips tombstones
whose `claimable_at` has not passed. Every write after a tombstone's insert
is a claim's: the writes that end a claim are matched on
`claim_generation`, and a round that found nothing deletes the tombstone
under any claim (see [purge](vector_store_purge.md#the-lease)). A
transition made by anything that holds no claim would have to increment the
generation, so that an outstanding claim's late failure matches nothing.

The retention is the namespace's. Event memory's configuration names its
registry's database and its retention, and checks the retention's floor
against its remote vector store's request timeout; semantic memory's
retention is fixed in its code.

### Event memory on the registry

- Event memory's registry holds its partition namespace, over
  `EventMemoryStore`, which composes event memory's two stores: the segment
  store and the vector store, each holding a tenant's records under the
  partition's UUID. The segment store keeps one row per partition, with its
  payload codec configuration and, during its purge, its cursor, and no name.
- `EventMemoryStore` creates, purges, and drops a partition in both stores
  at once. Each such call finishes in every store before the outcome is
  decided: when any raised, it logs each error and raises the first, so no
  store is still creating storage when the registry cancels the reservation,
  the caller sees a store's own exception, and the stores that answer keep
  purging while another is down. Its purge finds records while either
  store's does. Its handle on a partition pairs the segment store's handle
  with the vector store's. It gets the segment store's handle first and
  answers `None` when the segment store finds nothing under the UUID, since
  a vector store answers a handle for any UUID.
- Event memory's `create_partition`, `get_partition`, and `delete_partition`
  address its partitions by name through the registry, and the first two
  answer event memory's handle on the partition. The service locator gets a
  session's partition and creates it when no partition holds the name.
- Event memory's deployment configuration (`episodic_memory.long_term_memory`)
  names the registry's database (`partition_registry`, required for the
  event backend) and the retention (`tombstone_retention_seconds`), one of
  each for every session's partition.
- A further derived index is a further store in `EventMemoryStore`, with no
  schema change and no change to the registry.

Semantic memory's vector-store backend keeps the one partition it never
deletes: it gets the partition named `semantic_memory` in the namespace
`semantic_memory` through a registry over its vector store, creating it when
no partition holds the name, as the service locator gets a session's, and
builds its storage on the vector store's handle that the registration
carries. The registry is in the SQL database semantic memory's vector
backend uses, and the partition is never deleted, so semantic memory takes
no setting for it: its retention is fixed in its code.

### Other owners

The registry knows nothing of what owns a namespace, so any component that
keeps tenants in one store, or in several composed into one, owns a
namespace the same way.

### Differences from a catalog beside the data

A catalog kept in the same SQL database as its data can fence a write in the
write's own transaction. The registry has that design's bounded mint loop,
its in-transaction locking re-check of the queue, its idempotent deletion
that queues a tombstone, and its claim under `FOR UPDATE SKIP LOCKED`. It
differs where a store is remote:

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

- **One identity per partition, minted by the registry.** Every store holds
  a partition's records under its UUID, so no store's partition can outlive
  or predate another's, and no order between stores matters.
- **The registry drives its store's steps.** Creation (reserve, create in
  the store, confirm or cancel) and the purge round (purge, then drop) are
  written once, in the registry, over the `PartitionedStore` interface, so no
  owner repeats the sequence or its cancellation handling, and no owner can
  miss or reorder a step. The registry depends on that interface alone: any
  store that implements it works. A round that finds nothing in the store
  proves the partition gone, so the registry tracks no per-store progress.
- **One store per registry; a memory system with several composes them.**
  The registry's operations are the store's, addressed by name instead of
  UUID, and each registration carries the store's own handle, typed for that
  store, so an owner never pairs a UUID with a store by hand. How several
  stores run one step is the composite store's, which knows its stores; a
  new store is one more store in the composite, and the registry is
  unchanged.
- **Purge and drop are separate steps.** A purge deletes records from
  storage the partition shares with other partitions, over one call or many,
  until none remains; a drop removes the storage the partition keeps for
  itself.
  The registry drops only after the store's purge finds nothing, and a
  composite store's purge finds nothing only when none of its stores' does,
  so a round that drops proves the partition gone from every store.
- **The owner fences each operation by its registration.** One liveness and
  late-write mechanism serves every store, the registry can live in any
  database, and an operation pays one check before and one after its
  writes, whatever number of stores it calls.
- **Deletion is by name.** Callers delete partitions by name, and the name is
  resolved to its partition's UUID inside the deleting transaction. Keying it
  by UUID alone would make every caller resolve the name first, in a
  separate transaction, and would still need the name-addressed path for a
  caller that holds no handle; a pending reservation's `cancel` serves the
  creation taking back its own. A memory system's lifecycle is tied to no
  caller's, which keeps it composable; a caller that needs to delete one
  particular partition mints a fresh name for each.
- **Only creation creates.** Getting a pending partition does no work on it: a
  caller that gets or creates waits for it. A creation that stopped part way
  stays pending until its caller deletes the name: the registry offers the
  deletion and holds no policy on when to give up.
- **`startup` creates the registry's tables when missing**, as every store's
  startup creates its own durable resources idempotently.

## Alternatives considered

- **Keep the catalog in the backend.** Neither Qdrant nor Milvus can
  arbitrate a create, a delete, or a claim.
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
- **Each owner sequences its stores' steps.** The registry would know no
  stores: each owner would reserve, create in its stores, and confirm or
  cancel, and supply a purge callback that runs its stores. Every owner then
  repeats the sequence and its shielded cancellation, and one that misses or
  reorders a step, such as confirming before every store has created the
  partition, leaves the stores out of step.
- **The registry over a list of stores**, running each step on every store
  itself. Each store's handle has its own type, so the registry could answer
  none of them: the owner would get each store's handle by the
  registration's UUID, and how the stores run one step would be the
  registry's, for stores it knows nothing of.
- **Stores register handlers that the registry tracks per store.** The
  registry would record each store's progress for each partition. A round
  that finds nothing in any store already ends the partition, so per-store
  progress adds state nothing reads.
- **One purge step that also removes the partition's own storage.** Each
  store's purge would delete its records and then, once none remain, its
  own storage. One call then does two kinds of work, and one store's own
  storage is removed while another store's purge may still find records.
- **An in-transaction fence in the SQL stores.** A SQL store would check
  liveness in the same statement as each read and hold the partition's
  catalog row under a shared lock while it writes, so a deletion would wait
  out its writes and no SQL write would land late. With the registry's
  tables in the same database, the store would do so through a predicate
  the registry supplies. It saves the check after a write and the late SQL
  write, at the cost of a second liveness mechanism and of placing the
  registry in that database.
- **Storage first, registry last**, with no pending state. Storage named by a
  partition's UUID cannot be made before the registry mints the UUID, so a
  partition's own storage is created after its reservation, and the storage
  a store's partitions share is prepared at startup. The cost is a pending
  row a crash leaves, deleted like any other.
- **One registry class for both levels**, with `mark_live(partition_uuid)`,
  `unregister_partition(partition_uuid)` and a lookup answering the state as
  fields. Callers carry UUIDs back into the registry, any UUID fits any call,
  and the state fields go stale; see [Reservations and
  registrations](#reservations-and-registrations).
- **One handle type for every state**, answered by the reservation and the
  lookup alike. It would offer `confirm` and `cancel` on a live partition, and
  need a state field or an extra read to tell the two apart.
- **Handle types named by state** (`PendingRegistration`, `LiveRegistration`).
  A state name stops being true once someone deletes the partition; a role
  name does not.
- **Getting by a lookup and a separate liveness read.** Getting would take
  two reads of the registry instead of one.
- **The UUID as the only identifier**, with names resolved elsewhere. Callers
  address partitions by name, so the name lookup has to be arbitrated
  somewhere; the primary key on `(partition_namespace, partition_name)` is that
  place.

## Consequences

- An operation reads the registry once, or twice when it writes, however many
  stores it calls; getting a tenant's partition reads it once. Each read is
  one primary-key lookup on the registry's database.
- A deleted tenant's records stay physically until its purge, after the
  retention; they are unreachable from the deletion.
- A store is used through its owner, which holds the registry, and the
  registry alone creates, gets, purges, and drops its partitions.
- Stores address partitions by UUID, not by name.
