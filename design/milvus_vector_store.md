# Milvus vector store

How the Milvus store meets the shared contracts: [collection
registry](vector_store_collection_registry.md),
[purge](vector_store_purge.md), [consistency](vector_store_consistency.md),
[isolation](vector_store_isolation.md).

## Layout

- **One native collection per namespace and configuration**, named
  `memmachine_{namespace}__{sha256(config)}`, the digest over the
  configuration's JSON. It holds every logical collection of that namespace
  and configuration, one incarnation each. Admitting a collection creates
  nothing in Milvus unless its configuration is new.
- **Fields:** `id` (VARCHAR primary key, `"{incarnation}:{record_uuid}"`),
  `record_uuid` (VARCHAR), `partition_key` (VARCHAR, the incarnation,
  `is_partition_key`), `vector` (FLOAT_VECTOR), `properties` (JSON), and one
  nullable typed field per declared property, `_p_<name>`. Dynamic fields
  are off, so each property is stored once.
- **Tenancy:** partition-key multi-tenancy with `partitionkey.isolation`: each
  segment builds its vector index per group of tenants, so a search filtered
  on one incarnation searches only its group. Milvus documents isolation for
  HNSW indexes; the store's HNSW_SQ is one.
- **Vector index:** HNSW_SQ, 4-bit codes with FP16 refinement, M=18,
  efConstruction=240: what AUTOINDEX builds on CPU from Milvus 2.6.10
  (`autoIndex.params.build`, `autoindex_param_nocuda.go`). Naming it builds
  the same index on every server, whatever its version (2.6.8 and 2.6.9 build
  float32 HNSW) or AUTOINDEX configuration. A search sets only
  `refine_k = 8` and leaves `ef` at knowhere's default, `max(limit, 16)`:
  knowhere walks the graph on the 4-bit codes keeping `max(ef, limit x
  refine_k)` candidates, and rescores `limit x refine_k` of them against the
  FP16 vectors. Neither the index nor the search is configurable.
- **Declared properties:** each has a scalar AUTOINDEX, which Milvus
  resolves by type (BITMAP for BOOL, STL_SORT for TIMESTAMPTZ, HYBRID
  otherwise: BITMAP under 100 distinct values, STL_SORT above). A datetime is
  a TIMESTAMPTZ field holding its instant, written in UTC: a filter compares
  only instants, a search answers with record UUIDs and scores, and Milvus
  refuses an offset with a seconds component. Milvus caps a collection at
  `proxy.maxFieldNum` fields (64 on 2.6, 256 on 3.0), so a configuration
  declares at most 59 properties on 2.6.
  Undeclared properties go in the JSON field, still filterable by path.
  Negation is the complement, as on Qdrant: a negated condition holds where
  the property has no value, which Milvus's SQL-style null evaluation does not
  give on its own.
- **Scores** are the server's (cosine similarity, inner product, and the
  square root of Milvus's squared Euclidean distance).
- **Server-configured limits stay the server's.** A search `limit` reaches the
  server, which refuses one above `quotaAndLimits.limits.topK`. A declared
  string's VARCHAR length is `max_varchar_length` (65,535 unless configured,
  within `proxy.maxVarCharLength`), and a purge batch `purge_batch_size`
  (10,000 unless configured, within
  `quotaAndLimits.limits.maxQueryResultWindow`); both are settings, not
  constants. A native collection keeps the VARCHAR length it was created
  with, so a changed `max_varchar_length` applies to native collections
  created afterward. A record's undeclared properties share one JSON field,
  which the server refuses above `common.JSONMaxLength` bytes (65,536 unless
  configured). A declared property's field name is its key, at most 32
  bytes, under `_p_`, within `proxy.maxNameLength` (255 unless configured).
- **Creation converges:** the collection and its indexes (named by their
  fields) are created only when missing, and the collection is loaded, a
  no-op when it is loaded already.
- **A delete raises unless Milvus accepted every key sent.** Milvus counts the
  primary keys a delete accepts, present or not (milvus-io/milvus#51566), and
  pymilvus's async client returns that count for a delete Milvus rejected, so
  the store compares it with the keys it sent.
- **Oversized upserts are halved.** Milvus refuses a request over
  `proxy.grpc.serverMaxRecvSize` (64 MiB unless configured) with gRPC's
  RESOURCE_EXHAUSTED status before writing any of it, and pymilvus raises that
  status as it is (measured on 2.6.24). An upsert refused so is halved until
  its halves fit or a single entity is refused, as on Qdrant. Any other error
  raises at once: a timed-out request may still be applied, and sending it
  again adds load to a server already too slow.
- **Milvus Lite is not supported.** It is a separate embedded engine that
  scores, indexes, and enforces collection properties differently; a URI
  ending in `.db`, which pymilvus serves with Lite, is refused. Every call the
  store makes exists in Milvus 2.6.8 and later; CI tests against 2.6.24, and
  the store's tests pass against 2.6.8 and 3.0.2.

### Why a shared collection

Measured (Milvus 2.6.24, 4 CPUs / 5 GB; 200 tenants: 1 of 50,000, 10 of 5,000,
50 of 1,000, 139 of 100; 163,900 vectors of 128 dimensions; one run):

| | Shared (partition key + isolation) | One collection per tenant |
|---|---|---|
| Setup (create, insert, index, load) | 15 s wall, 28 CPU-s | 841 s wall, 155 CPU-s |
| Memory after load | 0.78 GB | 0.92 GB (about 0.7 MB more per collection) |
| Admitting a tenant | a registry row; nothing in Milvus | 2.0 s: create, index, load an empty collection |
| Recall@10 (50k / 5k / 1k / 100-row tenants) | 1.00 / 0.95 / 0.96 / 1.00 | 1.00 / 0.94 / 1.00 / 1.00 |
| Search p50 (same classes) | 2.1 / 1.2 / 1.1 / 1.2 ms | 3.4 / 1.3 / 1.5 / 1.4 ms |
| 4-client throughput | 2,794/s | 3,041/s |

The per-tenant collections' 1k-row recall of 1.00 is an artifact: Milvus does
not index a segment under 1,024 rows, so those tenants were searched exactly.
Milvus also caps a deployment's collection count.

### Why this index

Milvus's defaults are cheap rather than accurate. AUTOINDEX's HNSW_SQ was
chosen for memory, and with no search parameters knowhere rescores only the
results the 4-bit codes picked (`refine_k = 1`); the change that made it the
default says to raise `ef` and `refine_k` if recall is insufficient
(milvus-io/milvus#47386). The store keeps every default but `refine_k`.

Measured on Milvus 2.6.24 (180k vectors of 384 dimensions from a mixture of
64 clusters; tenants of 100k, 10k, 1k, and 100 rows; 100 queries per tenant
size; recall@k against exact search within the tenant), on the 100k-row
tenant, over two runs unless marked:

| Index and search | Recall@10 | Recall@100 | Search p50, k=10 / k=100 |
|---|---|---|---|
| AUTOINDEX, no search parameters (one run) | 0.73 | 0.88 | |
| HNSW_SQ, `refine_k = 4` | 0.945-0.955 | 0.998-0.999 | 2.2 / 2.6-3.0 ms |
| HNSW_SQ, `refine_k = 8` (the store) | 0.988-0.990 | 0.999 | 2.2-2.3 / 2.7-3.0 ms |
| HNSW_SQ, `refine_k = 16` | 0.993-0.994 | 0.999 | 2.4 / 3.7-4.0 ms |
| HNSW_SQ, `ef = max(k, 128)`, `refine_k = 4` | 0.989-0.990 | 0.998-0.999 | 2.3-2.4 / 2.5-2.6 ms |
| Float32 HNSW, M=18, efConstruction=240 | 0.855-0.860 | 0.984-0.986 | 2.3-2.6 / 2.5-2.6 ms |
| Float32 HNSW, M=18, efConstruction=240, `ef = max(k, 128)` | 0.998-1.000 | 0.992-0.993 | 2.3-2.5 / 2.5-2.6 ms |
| Qdrant store as configured (one run) | 0.977 | 0.940 | 5.8 / 5.9 ms |

Smaller tenants recall 0.99 or more in every HNSW_SQ row with `refine_k` of
4 or more. The build is not deterministic: one float32 index scored 0.987 and
0.998 at recall@10 in two runs, so a difference of 0.01 is noise.

The loss is the 4-bit codes', not the graph's. Across 96 comparisons in two
runs, MemMachine's own HNSW engines' parameters (hnswlib and usearch: M=16,
efConstruction=128) recalled up to 0.03 less than AUTOINDEX's and at most
0.002 more, and building took 1.3 to 1.7 times as long with AUTOINDEX's.
Raising `ef` changed little once enough candidates were rescored; `refine_k`
decided it (with `ef = max(k, 128)`: 0.80, 0.94, and 0.99 at recall@10 for
`refine_k` of 1, 2, and 4, one run). With the default `ef`, `refine_k = 8` is
the least measured that comes within 0.01 of float32 HNSW searched with
`ef = max(k, 128)` at k=10 and beats it at k=100, at a latency within noise
of `refine_k = 4`; 16 recalls within noise of 8 and adds 1 ms at k=100.

Memory, measured on Milvus 3.0.2 at 600k vectors of 768 dimensions (tenants
of 200k, 50k, 5k, and 2,000 of 100; 4 CPUs; one run): HNSW_SQ took 0.8 GB less
than float32 HNSW and 2.9 against 3.7 ms of CPU per search on the 200k
tenant, at similar recall. Partition-key isolation halved the index-build CPU
there (855 against 448 CPU-seconds) for about 0.2 GB more memory, with search
throughput unchanged.

## The composite key

The incarnation is in the primary key because Milvus's upsert deletes by
primary key in every partition (`AllPartitionsID`) before inserting: with the
bare record UUID as the key, one collection's upsert would delete another
collection's record of the same UUID. With the composite key, two collections'
records of one UUID are two entities, and a reused UUID's record is simply
stored, which meets the [isolation](vector_store_isolation.md) guarantee. The
record UUID is also kept in its own field, which reads return. Measured
against bare keys (Milvus 2.6.24, 4 CPUs / 5 GB; 1.3M rows: 1,000 tenants of
1,000 and 3 dead tenants of 100,000; one run):

| p50 | Composite key | Bare key | Bare, filtered on incarnation |
|---|---|---|---|
| Get 10 ids | 1.29 ms | 0.87 ms | 1.22 ms |
| Delete 1 id | 1.10 ms | 1.10 ms | 200 ms (a delete by filter queries first) |
| Upsert 5 | 1.61 ms | 1.50 ms | |
| Search | 1.56 ms | 1.73 ms | |
| Memory after load | 1.51 GB | 1.44 GB | |

A later sweep (upsert by batch size, get, delete, 8 concurrent writers,
searches with and without writers) put every difference within run-to-run
variation.

## Purge

Bounded batches. A round lists up to `purge_batch_size` of the incarnation's
primary keys, by a query on the incarnation field at the store's read level,
and deletes them by key. It first creates the native collection's missing
indexes and loads it, as a creation does: a creation that failed partway can
leave it unindexed and unloaded, and only a loaded collection answers the
listing. Measured (Milvus 3.0.2; 2.11M points in this layout,
dead incarnations of 10k to 1M among live tenants under search, upsert, and
scroll traffic; 2 CPUs / 4 GB): rounds stayed flat at about 100 ms to the end
of a 1M purge, with at most a 0.3 s stall for other tenants; one filter-delete
of 1M points instead stalled every tenant's reads and writes for 2.7-9 s at
Session consistency. Listing by the incarnation field took 57-63 ms per round
of 10,000 against 70-75 ms by primary-key range (1.4M rows, Milvus 2.6.24).

The listing reads at Bounded, as every other read does, and is correct there:
a Bounded read is at most `common.gracefulTime` (5 s by default) behind and
waits rather than read staler, far inside the tombstone retention, so a round
that lists nothing has found the incarnation empty. When the query node keeps
up, a Bounded read waits for nothing, so a round can list entities the
previous round deleted but the node has not applied yet, and delete them
again. At the resource manager's one-second pause after a busy round this does
not happen. Measured (Milvus
2.6.24, 4 CPUs / 5 GB; one dead incarnation of 200,000 entities among 50 live
tenants of 2,000, 128 dimensions; rounds of 10,000 on the async client while 4
tasks search live tenants; two runs each):

| Listing level | Pause | Rounds | Entities listed again | Wall | Search p99 |
|---|---|---|---|---|---|
| Bounded | 1 s | 20 | 0 | 21.3-21.6 s | 4.2-4.5 ms |
| Session | 1 s | 20 | 0 | 21.6-22.1 s | 5.1-5.7 ms |
| Strong | 1 s | 20 | 0 | 24.4-25.2 s | 9.7-10.0 ms |
| Bounded | none | 83-87 | 630,000-670,000 | 5.7-6.5 s | 23-27 ms |
| Session | none | 20 | 0 | 4.0-4.5 s | 6.4-9.1 ms |
| Strong | none | 20 | 0 | 2.2-2.4 s | 7.6-12.9 ms |

At 1M the pattern held (Milvus 3.0.2, the conditions of the first
measurement above, one dead incarnation of 1M, rounds of 10,000): at the
one-second pause Bounded took 114.4 and 114.9 s over 101-102 rounds against
Session's 120.8 and 121.2 s over 101, every entity purged each time; with no
pause Bounded took 118.9 s over 379 rounds against Session's 25.0 s over 101.

## Consistency

**How Milvus orders and reads.** Each write is appended to its shard's
write-ahead log, which stamps it with a timestamp at append
(`internal/streamingnode/server/wal/interceptors/timetick/timetick_interceptor.go`,
v2.6.24), and every replica applies the log in that order, so a record's
writes take effect in one order everywhere. A query node searches at its
*tsafe*, the point up to which it has applied the log, after waiting until
that reaches the request's guarantee timestamp (`waitTSafe`,
`internal/querynodev2/delegator/delegator.go`):

- `Strong`: the request's own start, so every write that returned before it;
- `Bounded`: the start less `common.gracefulTime` (5,000 ms by default);
- `Session`: the searching process's last write, which pymilvus caches per
  process and collection (`GlobalCache.collection_ts`); a process that wrote
  nothing waits for nothing;
- `Eventually`: no wait.

A request that names a level overrides the collection's
(`internal/proxy/task_query.go`). The wait is per shard, not per tenant: every
tenant of a native collection shares its shards' logs, so a Strong or Session
read waits for every tenant's writes. When the node keeps up, a Bounded read
waits for nothing and reads the node's latest state; when it falls further
behind, the read waits rather than reads staler, and fails over to another
replica if the node's tsafe stalls for 3 s (`queryNode.waitTsafeStallTimeout`,
from 2.6.15).

**The store reads at Bounded**: it creates each native collection at Bounded
by name, rather than at whatever level pymilvus defaults to, and names no
level on a read, so every read runs at the collection's level. The level is not configurable:
the stated delay of at most `common.gracefulTime` and the tombstone retention
depend on it.

Measured on Milvus 2.6.24 (4 CPUs / 5 GB; one native collection in this
layout, 300 tenants x 1,000 rows of 128 dimensions; 8 tasks searching tenants
0-149, top 10, on the async client; 4 writers upserting 5 new points at a time
into tenants 150-299, which the searches never read; collection created at
Session, "default" passing no level; 15 s per phase, two rounds, the second in
reverse order):

| Writers | Search level | Searches/s | p50 | p99 |
|---|---|---|---|---|
| none | default (Session) | 1,272-4,008 | 1.8-4.3 ms | 4.1-36 ms |
| none | Bounded | 1,256-4,106 | 1.8-4.1 ms | 3.9-33 ms |
| same process | default (Session) | 4.3-5.5 | 649-850 ms | 5.3-5.8 s |
| same process | Strong | 7.6-8.4 | 154-188 ms | 6.7-6.9 s |
| same process | Bounded | 718-1,403 | 4.2-7.1 ms | 39-58 ms |
| same process | Eventually | 654-1,263 | 4.6-7.9 ms | 37-58 ms |
| other process | default (Session) | 715-874 | 6.3-7.2 ms | 45-55 ms |
| other process | Bounded | 769-891 | 6.1-6.7 ms | 45-57 ms |
| other process | Strong | 40-41 | 5.8-6.3 ms | 6.5 s |

Writes grow the collection, so later phases run slower; the ranges cover both
rounds. Session stalls whenever the searching process also writes, which is
MemMachine's normal case, and Strong stalls whoever writes; Bounded serves as
many searches as Eventually.

**Tests** run the store at its default and settle before its reads: one Strong
read of the native collection, after which, with one replica, the store's
reads reflect every earlier write (see
[consistency](vector_store_consistency.md)).

## Client

The store calls pymilvus's `AsyncMilvusClient`, whose docstring calls it
experimental and partial; it has every call the store makes. The alternative,
the synchronous `MilvusClient` under `asyncio.to_thread`, holds a thread of
the event loop's default executor, min(32, CPUs + 4) threads shared by every
`to_thread` call of the process, for as long as Milvus takes to answer, a
read's wait for its level included. Measured on Milvus 2.6.24
(collection Bounded; writers in another process; the measured process running
8 Bounded searches plus G gets by id at level L; 15 executor threads for the
synchronous client; two rounds):

| Beside the 8 searches | Client | Searches/s | Search p99 | Get p99 |
|---|---|---|---|---|
| nothing | sync | 1,128-1,627 | 20-43 ms | |
| 4 Strong gets | sync | 806-1,452 | 31-53 ms | 2.9-3.9 s |
| 32 Bounded gets | sync | 201-293 | 49-96 ms | 49-95 ms |
| 32 Strong gets | sync | 12-21 | 9.3-10.0 s | 13-15 s |
| 4 Strong gets | async | 1,055-1,434 | 32-44 ms | 3.0-3.5 s |
| 32 Strong gets | async | 670-752 | 43-45 ms | 0.4-2.5 s |

On the synchronous client, once more calls waited than the executor had
threads, every other call of the process queued behind them; on the async
client a waiting call holds no thread. The server-side wait is the same on
both.

## Consequences

- An existing native Milvus collection created with the earlier schema is not
  usable by this store and has to be dropped.
- A deployment that raises `common.gracefulTime` lengthens the store's read
  delay, which the tombstone retention must still exceed by far.
