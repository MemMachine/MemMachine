# Vector store: isolation between partitions, and record UUIDs

Part of [vector store horizontal scaling](vector_store_horizontal_scaling.md).

## Problem

A record is addressed by its UUID within its partition, and an upsert
replaces a whole record. On a backend where the partitions of a store share
one id space, one partition's upsert of a UUID another partition already
holds would replace that other partition's record. Qdrant is such a backend
as MemMachine lays it out: a store's partitions share its native collection,
and a point's id was its record's UUID. The contract guarded against it with a rule
on callers, that a record's UUID is minted by the service and never a value a
caller supplied, which held only as long as every ingestion path honored it.
This document answers what keeps partitions apart, what happens when a UUID
is reused (across partitions, across lives of one partition, or by a caller
that chose it), and what the contract can promise, given that every backend
has to be able to keep the promise.

## Partition lives

- An incarnation is a random UUID (version 4) minted by the registry at
  registration, one per partition life (see [partition
  registry](vector_store_partition_registry.md)). Every read, write and
  delete of a handle is scoped to it.
- An incarnation is never re-minted while it is registered, pending or live,
  or its tombstone is queued, so a new life starts empty: no dead life's
  records are adopted by it or reclaimed out from under it, whatever record
  UUIDs either life used.

## The guarantee

`VectorStorePartition` states:

> All data operations are scoped to the partition: a record's UUID names it
> in this partition only, and the same UUID in another partition names
> another record.

A record UUID may come from anywhere, a caller included: reusing one in
another partition stores another record, and no operation on one partition
reads, replaces or deletes another's.

| Store | How ids are scoped to a partition |
|---|---|
| SQLite, sqlite-vec | a records table per partition |
| Milvus | the incarnation in the primary key, `"{incarnation}:{record_uuid}"` (see [Milvus](milvus_vector_store.md)) |
| Qdrant | the point id is a UUIDv5 of the record UUID under the incarnation, with the record UUID kept in the payload (see [Qdrant](qdrant_vector_store.md)) |

A UUIDv5 carries 122 bits, and deriving one from a random incarnation and any
record UUID gives a caller no way to aim at another partition's point: a
SHA-1 collision needs control of both inputs.

### Other backends

Surveyed from vendor documentation and source (2026-09-29; none of these
backends was run). Every one can, almost always by scoping the id.

| Backend | Per-tenant unit that scopes ids | Tenant in the native id | Conditional write (outcome on collision) |
|---|---|---|---|
| Qdrant | (custom shard keys; ids "only enforced unique within a shard key", which Qdrant calls an anti-pattern) | id must be a u64 or UUID: a derived UUID | `update_filter` (skipped silently) |
| Milvus | collection (deployment caps) | composite VARCHAR key | none found |
| Pinecone | namespace (100 to 1M per index by plan) | string id, 512 characters | none |
| Chroma | collection (Chroma Cloud: 1M) | string id (Chroma Cloud: 128 bytes) | `add` of an existing id is skipped silently; no conditional replace |
| Weaviate | multi-tenancy tenant (recommended; 1M+ tenants) | id must be a UUID: UUIDv5 of the pair | a single insert of an existing id fails (422), not atomically; batch overwrites |
| turbopuffer | namespace (recommended; unlimited) | string id, 64 bytes (a 73-character pair does not fit) | `upsert_condition` (skipped silently) |
| PostgreSQL + pgvector | table, or list partition (a few thousand partitions) | composite primary key | `ON CONFLICT ... DO UPDATE ... WHERE` (row count 0) |
| Elasticsearch, OpenSearch | index (shard limits cap it) | composite `_id` (512 bytes); custom routing does not scope `_id` | `op_type=create` (409); scripted upsert (`noop`) |
| Redis Query Engine | none: one keyspace | key prefix | `JSON.SET NX` (nil); Lua |
| LanceDB | table | `merge_insert` on (tenant, id); ids are never enforced unique | `merge_insert ... where` (skipped silently) |
| MongoDB Atlas | collection (not recommended) | composite `_id` or unique compound index | a filtered upsert on a raw id raises DuplicateKey |
| Vespa | streaming-mode group (`g=` in the document id) | document id | test-and-set (412), except when every replica lacks the document |
| Azure AI Search | index (at most 3,000 per service) | composite key `{tenant}_{record}` (1,024 characters; no colon) | none |

## Alternatives considered

- **Keep the rule that UUIDs are service-minted.** It holds only while every
  ingestion path honors it, and a slip replaces another tenant's record, a
  security failure far from its cause.
- **A conditional upsert on Qdrant** (`update_filter` on the writer's
  incarnation, first write wins). It keeps partitions from replacing each
  other's points, but a reused UUID's record is dropped without an error, the
  contract has to keep a uniqueness rule for callers, and single-point writes
  ran 46% slower. It is Qdrant-specific, where scoping the id is what nearly
  every backend does.
- **Check before writing** (read the id, then write if it is absent or the
  writer's). Not atomic: another partition's write can land between the read
  and the write.
- **Bare or reversible point ids on Qdrant.** A bare record UUID as the point
  id lets one partition's upsert replace another's point; a reversible
  derivation lets anyone who learns two incarnations compute a colliding UUID.
  The [Qdrant](qdrant_vector_store.md) document has the analysis and the cost
  of reading the record UUID from the payload.
