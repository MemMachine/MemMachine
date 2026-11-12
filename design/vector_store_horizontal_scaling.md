# Vector store horizontal scaling

## Problem

The `VectorStore` contract said "a collection must be managed by at most one
process; the consumer shards names across processes." No consumer sharded
anything, and the Qdrant and Milvus stores relied on that sentence for
correctness:

- their catalogs lived inside the backend, which cannot arbitrate a create, a
  delete, or a purge between processes;
- they serialized their own operations with process-local locks;
- a collection's name was the tenant discriminator on its records, so deleting
  and re-creating a name did not end the old partition (#1563).

The goal is that any number of MemMachine server processes, on any hosts, can
serve any partition of a Qdrant or Milvus store they share, with:

- creation of a partition name arbitrated once;
- deletion final, not undone by a write in flight;
- a re-created name starting empty;
- reclamation of deleted data bounded in cost and safe to run from every
  process;
- reads whose relation to earlier writes the contract states;
- partitions isolated from one another whatever record UUIDs they carry.

The SQLite stores keep their single-process (engine-backed store) and
single-node (sqlite-vec) bounds.

## The store and its partitions

A vector store is constructed with a name and a schema: a dimensionality, a
similarity metric, and declared properties. Every record it holds has a vector
of that dimensionality, and any declared property it carries has its declared
type. The composition root builds the stores its services
need, and the name keeps two stores over one backend apart. Within a store, a
partition holds one tenant's records, addressed by the UUID its owner's
[partition registry](partition_registry.md) mints for it, and every search
runs within one partition. Every partition of a store shares its
dimensions, metric, and schema.

## Documents

Shared: contracts and choices made with every backend in mind.

| Document | What it covers |
|---|---|
| [partition registry](partition_registry.md) | The SQL catalog that arbitrates which partitions exist and their UUIDs, reservations and registrations, handles and their fencing, creation races, and the registry's creation and purge rounds over its stores. |
| [purge](vector_store_purge.md) | Tombstones, the retention, the claim, backoff, dead-lettering, the sweeper. |
| [consistency](vector_store_consistency.md) | What a query sees of earlier writes, as the contract states it; why the contract has no `get`; asynchronous clients; how tests observe state. |
| [isolation](vector_store_isolation.md) | Isolation between partitions, record UUIDs and their reuse, and whether other vector databases can meet the guarantee. |

Per backend: how each implementation meets the contracts, and the measurements
behind its choices.

| Document | What it covers |
|---|---|
| [Qdrant](qdrant_vector_store.md) | One native collection per store, per-tenant graphs, derived point ids, filtered-search correctness, purge by filter, consistency on one node and replicated. |
| [Milvus](milvus_vector_store.md) | One native collection per store, partition-key tenancy, the composite key, the index, purge in batches, consistency levels, the async client. |

## Lifecycle of a partition

1. **Create.** The owner, such as event memory, creates the partition by
   name through its registry, which reserves the name, inserting a pending
   row under a freshly minted UUID, has every store create the storage the
   partition keeps for itself (on Qdrant and Milvus, none: its records go
   into the store's one native collection, which startup prepared), and
   confirms the reservation, which marks the row live. A racing creator loses at the registry's primary
   key, and a pending partition is not opened.
2. **Open.** The registry resolves the partition name to the live
   partition; the owner opens the store's handle on its UUID.
3. **Use.** Every record a handle writes carries the partition's UUID, and
   every read and delete is scoped to it. Before each of its writes and
   reads, and after each of its writes and deletes, the owner checks that
   the partition is still live.
4. **Delete.** One registry transaction removes the partition's row and
   queues a tombstone. Every operation of the owner on that partition raises
   from then on, in every process.
5. **Purge.** Once the retention (a day by default) has passed, sweepers in
   every process claim tombstones oldest first and purge their records from
   every store of the registry in bounded rounds; once a round finds none,
   it drops the partition from every store and the tombstone goes.

## Guarantees by store

| Store | Processes that may serve a partition | Queries reflect a write | Deletion reclaims |
|---|---|---|---|
| SQLite (engine-backed) | one | as soon as it returns | by purge, after the retention |
| sqlite-vec | any on one node | as soon as it returns | by purge, after the retention |
| Qdrant | any, sharing the owner's registry | not stated; on one node as soon as it returns, since the store waits for each write; replicated, after an unbounded delay | by purge, after the retention |
| Milvus | any, sharing the owner's registry | within the server's `common.gracefulTime` (5 s by default) at Bounded | by purge, after the retention |

## Configuration

Event memory's long-term memory configuration names the relational database
that holds its partition registry (`partition_registry`) and the retention
before a deleted partition's purge starts (`tombstone_retention_seconds`, a
day by default). `QdrantConf` and `MilvusConf` bound every request to the
backend (`request_timeout_seconds`, 30 s by default). On a Qdrant or Milvus
vector store, event memory's retention must be at least 10 times the
request timeout plus 300 seconds (see [purge](vector_store_purge.md)).

## Related work

- The segment store's shared tables, keyed by partition UUID ([its
  design](segment_store_shared_tables.md)), are the model for the registry's
  minting of UUIDs.

## Deployment consequences

- Existing Qdrant and Milvus data is not carried over. The stores name their
  native collections by vector store name, which no earlier release did, so
  an existing collection stays as it is, never read or purged, until it is
  dropped, before or after upgrading. No migration; pre-GA.
- Milvus Lite is not supported; the Milvus store needs a Milvus server.
- Event memory needs a relational database for its partition registry.
