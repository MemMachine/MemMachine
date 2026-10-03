# Vector store horizontal scaling

## Problem

The `VectorStore` contract said "a collection must be managed by at most one
process; the consumer shards names across processes." No consumer sharded
anything, and the Qdrant and Milvus stores relied on that sentence for
correctness:

- their catalogs lived inside the backend, which cannot arbitrate a create, a
  delete or a purge between processes;
- they serialized their own operations with process-local locks;
- a collection's name was the tenant discriminator on its records, so deleting
  and re-creating a name did not end the old life (#1563).

The goal is that any number of MemMachine server processes, on any hosts, can
serve any partition of a Qdrant or Milvus store they share, with:

- creation of a partition key arbitrated once;
- deletion final, not undone by a write in flight;
- a re-created key starting empty;
- reclamation of deleted data bounded in cost and safe to run from every
  process;
- reads whose relation to earlier writes the contract states;
- partitions isolated from one another whatever record UUIDs they carry.

The SQLite stores keep their single-process (engine-backed store) and
single-node (sqlite-vec) bounds.

## The store and its partitions

A vector store is one collection: a body of records searched together, with
one dimensionality and one declared schema, named at construction by its
vector store name, and scored by cosine similarity. The composition root
builds one store per collection it needs, and the name keeps two stores over
one backend apart. Within a store, a partition holds one tenant's records,
addressed by a string key. Every partition of a store shares its dimensions
and schema; the registry records them beside each partition, so a store built
with others refuses the partition instead of reading what is not there.

## Documents

Shared: contracts and choices made with every backend in mind.

| Document | What it covers |
|---|---|
| [partition registry](vector_store_partition_registry.md) | The SQL catalog that arbitrates which partitions exist, incarnations, reservations and registrations, handles and their fencing, creation races. |
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

1. **Create.** The store reserves the key, inserting a pending registry row
   under a freshly minted incarnation, prepares the storage the partition
   needs of its own (on Qdrant and Milvus, none: its records go into the
   store's one native collection, which startup prepared), and confirms the
   reservation, which marks the row live. A racing creator loses at the
   registry's primary key, and a pending partition is not opened.
2. **Open.** The registry resolves the partition key to the live incarnation;
   the handle is bound to that incarnation.
3. **Use.** Every record a handle writes carries its incarnation, and every
   read and delete is scoped to it. Before each upsert and query, and after
   each upsert and delete, the handle checks that its incarnation is still
   live.
4. **Delete.** One registry transaction removes the partition's row and
   queues a tombstone. Every handle of that life is stale from then on, in
   every process.
5. **Purge.** Once the retention (a day by default) has passed, sweepers in
   every process claim tombstones oldest first and reclaim their records in
   bounded rounds, until a round finds none and the tombstone goes.

## Guarantees by store

| Store | Processes that may serve a partition | Queries reflect a write | Deletion reclaims |
|---|---|---|---|
| SQLite (engine-backed) | one | as soon as it returns | at once |
| sqlite-vec | any on one node | as soon as it returns | at once |
| Qdrant | any, sharing the store's registry | not stated; on one node as soon as it returns, since the store waits for each write; replicated, after an unbounded delay | by purge, after the retention |
| Milvus | any, sharing the store's registry | within the server's `common.gracefulTime` (5 s by default) at Bounded | by purge, after the retention |

## Configuration

`QdrantConf` and `MilvusConf` name the relational database that holds their
stores' registries (`partition_registry`), the retention before a deleted
partition's purge starts (`tombstone_retention_seconds`, a day by default),
and the bound on every request to the backend (`request_timeout_seconds`,
30 s by default). The retention must be at least 10 times the request timeout
plus 300 seconds (see [purge](vector_store_purge.md)).

## Related work

- The segment store's shared tables with incarnation-scoped keys ([its
  design](segment_store_shared_tables.md)) are the model for the registry's
  incarnation logic.

## Deployment consequences

- Existing Qdrant and Milvus data is not carried over. The stores name their
  native collections by vector store name, which no earlier release did, so
  an existing collection stays as it is, never read or purged, until it is
  dropped, before or after upgrading. No migration; pre-GA.
- Milvus Lite is not supported; the Milvus store needs a Milvus server.
- Every Qdrant or Milvus store needs a relational database for its registry.
