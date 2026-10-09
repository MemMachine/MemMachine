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
  and re-creating a name did not end the old life (#1563).

The goal is that any number of MemMachine server processes, on any hosts, can
serve any collection of a Qdrant or Milvus they share, with:

- creation of a name arbitrated once;
- deletion final, not undone by a write in flight;
- a re-created name starting empty;
- reclamation of deleted data bounded in cost and safe to run from every
  process;
- reads whose relation to earlier writes the contract states;
- collections isolated from one another whatever record UUIDs they carry.

The SQLite stores keep their single-process (engine-backed store) and
single-node (sqlite-vec) bounds.

## Documents

Shared: contracts and choices made with every backend in mind.

| Document | What it covers |
|---|---|
| [collection registry](vector_store_collection_registry.md) | The SQL catalog that arbitrates which collections exist, incarnations, handles and their fencing, creation races. |
| [purge](vector_store_purge.md) | Tombstones, the retention, the claim, backoff, dead-lettering, the sweeper. |
| [consistency](vector_store_consistency.md) | What a query sees of earlier writes, as the contract states it; why the contract has no `get`; asynchronous clients; how tests observe state. |
| [isolation](vector_store_isolation.md) | Isolation between collections, record UUIDs and their reuse, and whether other vector databases can meet the guarantee. |

Per backend: how each implementation meets the contracts, and the measurements
behind its choices.

| Document | What it covers |
|---|---|
| [Qdrant](qdrant_vector_store.md) | Shared native collections, per-tenant graphs, derived point ids, filtered-search correctness, purge by filter, consistency on one node and replicated. |
| [Milvus](milvus_vector_store.md) | Shared native collections, partition-key tenancy, the composite key, the index, purge in batches, consistency levels, the async client. |

## Lifecycle of a collection

1. **Create.** The store reserves the name, inserting a pending registry row
   under a freshly minted incarnation, prepares the collection's storage (on
   Qdrant and Milvus, the native collection its namespace and configuration
   share), and confirms the reservation, which marks the row live. A racing creator loses at the registry's primary key,
   and a pending collection is not opened.
2. **Open.** The registry resolves `(namespace, name)` to the live incarnation
   and its configuration; the handle is bound to that incarnation.
3. **Use.** Every record a handle writes carries its incarnation, and every
   read and delete is scoped to it. Before each upsert and query, and after
   each upsert and delete, the handle checks that its incarnation is still
   live.
4. **Delete.** One registry transaction removes the collection's row and
   queues a tombstone. Every handle of that life is stale from then on, in
   every process.
5. **Purge.** Once the retention (a day by default) has passed, sweepers in
   every process claim tombstones oldest first and reclaim their records in
   bounded rounds, until a round finds none and the tombstone goes.

## Guarantees by store

| Store | Processes that may serve a collection | Queries reflect a write | Deletion reclaims |
|---|---|---|---|
| SQLite (engine-backed) | one | as soon as it returns | at once |
| sqlite-vec | any on one node | as soon as it returns | at once |
| Qdrant | any, sharing the store's registry | not stated; on one node as soon as it returns, since the store waits for each write; replicated, after an unbounded delay | by purge, after the retention |
| Milvus | any, sharing the store's registry | within the server's `common.gracefulTime` (5 s by default) at Bounded | by purge, after the retention |

## Configuration

`QdrantConf` and `MilvusConf` name the relational database that holds the
store's registry (`collection_registry`), the retention before a deleted
collection's purge starts (`tombstone_retention_seconds`, a day by default),
and the bound on every request to the backend (`request_timeout_seconds`,
30 s by default). The retention must be at least 10 times the request timeout
plus 300 seconds (see [purge](vector_store_purge.md)).

## Related work

- The segment store's shared tables with incarnation-scoped keys ([its
  design](segment_store_shared_tables.md)) are the model for the registry's
  incarnation logic.
- #1570 tracks moving provisioning (table and collection creation) out of
  runtime startup for every store.

## Deployment consequences

- Existing Qdrant and Milvus data is not carried over: drop the native
  collections before upgrading. Their names are unchanged, so a Qdrant
  collection kept through the upgrade holds its old points, invisible to every
  search and never purged, beside the new data, and dropping it afterwards
  drops both; an existing Milvus collection has the earlier schema, which the
  store cannot prepare. No migration; pre-GA.
- Milvus Lite is not supported; the Milvus store needs a Milvus server.
- Every Qdrant or Milvus store needs a relational database for its registry.
