# Partition registry: one identity and lifecycle for a memory system's tenants

## Problem

A memory system keeps each tenant's data in several stores. Event memory
keeps events, segments, and derivative links in the segment store and the
derivatives' vectors in a vector store, and further derived indexes (text, a
graph) would add more stores. Each store keeps its own catalog of
partitions and mints its own incarnation for each life of a key: the
segment store, the [partition registry](vector_store_partition_registry.md)
behind the Qdrant and Milvus stores, and both SQLite vector stores, each
with its own mint loop. A memory system then composes its stores by key:

- **Deleting is not one step.** The memory deletes each store's partition in
  turn, so a process that stops between the deletes leaves the tenant half
  deleted, and nothing durable records the rest of the work.
- **Creating again can inherit the old life.** A key created while one
  store's partition of the old life survives opens that partition, records
  and all, beside the other stores' new ones.
- **The stores' orders and checks carry the correctness.** Creation goes
  one way and deletion the other, and background work that opens a store's
  partition by key checks afterwards that the partition it started from
  still exists.
- **Every derived index adds a catalog, a mint loop, and an ordering rule.**

## Design

One registry per memory system holds the catalog of the memory's tenants. It
mints one incarnation for each life of a partition name, and every store of
the memory holds that life's records under that incarnation. Stores keep no
catalog of their own and mint nothing.

### Roles

- **The registry** belongs to one *partition namespace*: the partitions of one
  memory system, such as `event_memory`, shared by every process serving it.
  It arbitrates creation, makes deletion one transaction, queues each deleted
  life as a tombstone, and runs purge rounds on due tombstones through a
  callback. It knows partition names, incarnations, and tombstones, and
  nothing about stores. Its interface and its SQL implementation are the
  existing partition registry's, moved out of the vector store package and
  addressed by partition namespace where it is addressed by vector store name.
- **The owner** is the memory system, the concept the registry binds to. It
  holds its stores, prepares each store's partition for a new life, opens
  each store's partition for a live one, fences its operations, and
  supplies the purge callback. It takes the place the registry-backed
  vector store holds today: that base class reserves, prepares, confirms or
  cancels, opens, fences, deletes, and purges, and its subclasses supply the
  backend steps; here the memory system does the first, and its stores
  supply the steps.
- **A store** holds records under incarnations. It prepares a life's
  storage, opens a handle on a live life, and purges a dead life's records,
  each when its owner calls it.

Memory systems do not share a partition namespace, and nothing ties one
memory system's registry to another's.

The registry's rows are keyed by `partition_namespace` and `partition_name`,
in one pair of tables for every namespace, so the schema stays static. A
memory system's namespace is fixed in its code (`event_memory`,
`semantic_memory`), never derived from a configured resource, and
identifies one memory system among all those whose registries share a
database. Stores that still name native resources, such as a vector store's
collection, keep their own name for that alone.

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

### Names and incarnations

Callers address partitions by name, and only the registry holds names. A
store addresses its partitions by incarnation, so the rules stores put on
partition keys today (the segment store's 32-byte identifier rule, each
vector store's validation) give way to the registry's own limit, and a
caller's identifier, such as a session id, is the partition name as it is.

Deletion is by name: `unregister` removes whatever life holds the name when
it runs. A memory system's lifecycle is its own, tied to no caller's, so a
caller that must delete one particular life names its partitions with ids it
mints for each life, such as a server's session ids, and a name never comes
back.

A setting that belongs to one life is kept by what reads it. No reader of a
per-tenant setting exists beyond the segment store's payload codec, which
the segment store passes and keeps itself, so the registry stores none.

### Creation

The owner reserves the name, which mints the incarnation and leaves the life
pending, calls every store's `create_partition` at once, and confirms the
reservation, which makes the life live. Every call finishes before the
outcome is decided: if any raised, the owner logs each error, cancels the
reservation, and raises the first, so no store is still preparing storage
when the reservation is cancelled, and the caller sees a store's own
exception. No order between stores matters, since nothing opens the life
before it is confirmed. The registry's primary key is the one arbiter: a
racing creator loses at the reservation.

A life is confirmed only once every store has prepared its storage, so
storage that cannot be prepared never goes live: the creation cancels the
life and fails, and the cancelled life's storage is purged like any deleted
life's. Every creation prepares storage under a new incarnation, so nothing
deletes and creates storage in place.

Only creation creates. Opening a name reads the registry: a live life is
opened, and a pending one is waited for, a bounded number of times, and
then reported as pending. A creation that stopped part way, with its
process, leaves its life pending; the next creation of the name replaces
it, once it is older than any creation takes. `reserve` does that in one
transaction, conditional on the old life still being pending under its
incarnation: it removes the old life, queues its tombstone, and reserves the
name under a new incarnation. The old creator's `confirm`, conditional on
its own incarnation, then fails, so a slow creator never confirms a life
that was replaced, and nothing that went live is ever replaced. The age
decides only when to stop waiting; which life holds the name is decided by
the incarnation.

### Liveness

The owner fences each of its operations with the life's registration where the
registry-backed vector store fences each call today: an operation that writes
checks `require_current` before and after it, one that reads checks before it,
and one that deletes checks after it. For event memory, an ingest is the
first, a query the second, and a forget the third. A write that passes the
first check and lands after the deletion is under a dead incarnation, which no
handle reads, and the purge reclaims it. Stores, SQL ones included, check
nothing themselves, and no lock is held on the registry across a store's
transaction.

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

Every write finishes within a bound well under the retention, and the
bound is the client's, for SQL stores as for remote ones: a remote store's
request timeout, and the SQL driver's statement timeout. A process paused,
or an event loop stalled, between its check and its send or commit is
outside this design, as a paused remote writer already is.

### Deletion and purge

`unregister(partition_name)` is one transaction: the name is free and the
life unreachable when it commits, and its tombstone is queued. A creation
under the name mints a new incarnation at once, so a re-created tenant starts
empty in every store.

A tombstone comes due once the namespace's retention has passed since the
deletion. A purge round claims it under a lease, with the backoff and
dead-lettering the registry has now, and calls the owner's callback with the
incarnation. The callback runs every store's `purge_partition` to
completion at once, then raises if any raised, and otherwise answers
whether any found records; the tombstone goes when a round finds nothing in
any store. Letting every store finish its batch keeps the stores that answer
purging while another is down; the round still counts as failed, so the
registry's backoff applies. A store keeps whatever progress its purge needs
between rounds, such as the segment store's per-table cursors, under the
incarnation, and drops it when its purge finds nothing.

The purge queue keeps the registry's columns, except that a tombstone's
lease and backoff are stored as one `claimable_at`, the earliest time a
claim may take it again: a claim sets it to its lease end plus the backoff
for its attempt, a failure to the backoff's end, and progress clears it.
The retention stays its own condition, computed from `enqueued_at` when the
claim runs, so a change to it reaches every queued tombstone and no retry
schedule can shorten it. The claim keeps its one index range, oldest
deletion first, and skips tombstones whose `claimable_at` has not passed.
Every write after a tombstone's insert is a
claim's, matched on `claim_generation`; a transition made by anything that
holds no claim would have to increment the generation, so that an
outstanding claim's late failure matches nothing.

The retention is the namespace's. An owner's configuration names its
registry's database and its retention, as a Qdrant or Milvus store's
configuration names them today, and the retention's floor is checked
against the request timeout of each of the owner's remote stores, as it is
today.

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
- **Deletion is by name.** A memory system's lifecycle is tied to no
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

## Alternatives considered

- **Each store keeps its own catalog.** What the stores do now: each store
  is safe on its own, and a memory spanning several composes them by key,
  with the problems above.
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

## Consequences

- An operation reads the registry once, or twice when it writes, however
  many stores it calls; opening a tenant reads it once.
- A deleted tenant's SQL rows stay physically until its purge, after the
  retention; they are unreachable from the deletion.
- A store is used through a registry.
- Stores address partitions by incarnation, not by name.
