# Qdrant vector store

How the Qdrant store meets the shared contracts: [partition
registry](vector_store_partition_registry.md),
[purge](vector_store_purge.md), [consistency](vector_store_consistency.md),
[isolation](vector_store_isolation.md).

## Layout

- **One native collection per store**, named by the vector store name. It
  holds every partition of the store, one incarnation each. Creating a
  partition creates nothing in Qdrant.
- **Vectors:** one unnamed dense vector, the store's dimensions and metric.
- **Graphs per tenant, not per store:** HNSW `m=0`, `payload_m=16`. No
  collection-wide graph is built; each value of an indexed payload field gets
  its own graph once it has enough points in a segment, so a search filtered
  on one tenant walks that tenant's graph.
- **The incarnation:** payload field `sys-incarnation`, a keyword index with
  `is_tenant=true`, which also has Qdrant keep a tenant's points together.
  A handle's searches filter on it, as do the purge's scrolls and deletes. A
  handle's deletes name its points' ids instead (see [Point ids](#point-ids)).
- **Declared properties:** a payload index per property of the store's
  declared schema, typed by the property's type. A record or a filter naming
  any other key is refused.
- **Strict mode:** the collection is created with filtering on unindexed
  fields off, for reads and updates alike, so the server refuses a filter on
  a field it has not indexed instead of scanning for it. Every key a filter
  may name is indexed: the declared keys and the incarnation. A condition
  whose value is of another type than its key declares matches nothing; the
  store answers it with a filter no point satisfies, since the server would
  refuse it for the field's index.
- **Startup converges:** the collection and each payload index are created at
  startup under separate already-exists guards, so a startup that failed
  between them is completed by the next.
- **Client:** `AsyncQdrantClient`, with `request_timeout_seconds` as its
  timeout.
- **Oversized upserts are halved.** Qdrant's REST API refuses a request over
  `service.max_request_size_mb` (32 unless configured) with a 400 (measured on
  1.19.1; the status is not documented), and a proxy in front of it may
  refuse one with a 413. An upsert refused with either is halved until its
  halves fit or a single point is refused. A malformed batch is halved too,
  since only the message, which the store does not parse, tells the two 400s
  apart. Sizing requests beforehand would mean reproducing qdrant-client's
  undocumented serialization. gRPC has no such limit: Qdrant's server and
  qdrant-client set none, and a 140 MB upsert went through. Any other error
  raises at once: a timed-out request may still be applied, and sending it
  again adds load to a server already too slow.

## Point ids

A point's id is `uuid5(incarnation, record UUID)`, and the record UUID is kept
in the payload, `sys-record_uuid`. A search returns that one payload field to
answer each match's record UUID; a delete names the derived ids directly;
someone inspecting a collection finds a record by filtering on the field.

The id space is the native collection's, shared by the store's partitions,
and Qdrant's upsert replaces a whole point, vectors and payload alike
(`lib/shard/src/update/points/upsert.rs` at v1.19.1). With the record UUID as
the point id, an upsert of a UUID another partition's point held replaced
that point and moved it to the writer's incarnation. Derived ids differ
between partitions whatever record UUIDs they carry, so the store keeps the
[isolation](vector_store_isolation.md) guarantee without a rule on callers,
and a partition created again under a key writes ids its dead predecessor's
points, awaiting purge, cannot collide with.

Measured on Qdrant 1.19.1 (4 CPUs / 4 GB, gRPC; 300 tenants x 1,000 points of
128 dimensions in this layout; searches filtered on a tenant, top 10; one idle
collection for searches, six rounds; a fresh collection per upsert phase,
three rounds):

| | Record UUID as the point id | Derived id, record UUID in the payload |
|---|---|---|
| Searches, 8 clients | 4,748-5,392/s, p50 1.30-1.46 ms | 4,159-4,613/s, p50 1.52-1.67 ms (returning `sys-record_uuid`) |
| Upserts of 1 point, 8 clients | 4,193-4,722/s | 4,064-4,745/s |
| Upserts of 10 points, 8 clients | 1,462-1,733/s | 1,627-1,629/s |

A keyword index on `sys-record_uuid` would turn a lookup by hand from a scan
into an index read, at about 5% of 10-point upsert throughput (1,524-1,570/s);
it is not added.

The search cost is reading the field back, not the ids: both search variants
ran on one collection with the same derived ids and payload, and differed only
in whether each hit returned `sys-record_uuid`. Qdrant keeps payloads out of
RAM by default (`on_disk_payload: true`; only indexed filter fields stay in
memory), so each hit's field is read from payload storage. Upserts, which
compute the UUIDv5 and store the field, ran the same as with bare ids. If
search throughput at saturation ever matters, keeping the payload in RAM
(`on_disk_payload=false`) is the lever that keeps one-way ids, at the memory
of every property's payload; it is not measured.

**Why one-way ids.** The record UUID has to come back from the payload because
a UUIDv5 cannot be inverted, and that is also what makes it safe. An attacker
who writes through the API into one partition, and wants to hide a record of
another partition sharing the native collection, needs a record UUID whose
point id equals the target's. With UUIDv5 that is a second preimage of SHA-1
on 122 bits, about 2^122 work, even for someone who knows both incarnations
and the target's record UUID; known SHA-1 attacks need control of both inputs.
Isolation then holds as long as the attacker cannot write to Qdrant or the
registry directly, where isolation is moot anyway.

**Why not reversible ids.** A point id the store can invert, such as the
record UUID XOR the incarnation, would let a search recover the record UUID
from the point id and return no payload. But whatever lets the store invert
the mapping lets anyone holding the incarnations aim it: with the target's
incarnation, their own, and the target's record UUID, an attacker computes the
colliding UUID directly (`record ^ incarnation_A ^ incarnation_B`), and a
write of it through an ingestion path that passes a caller's UUID through
replaces the target's point, which then disappears from its partition.
Incarnations are not guarded as secrets: the registry logs them in its
warnings and dead-letter errors, and they are in its tables and any backup, as
record UUIDs are in any Qdrant snapshot. So a read-level leak plus an ordinary
tenant account would become targeted deletion of other tenants' records, where
UUIDv5 needs write access to a database. Any reversible keyed mapping,
encryption included, has the same property. A reversible id is also not a
proper UUID: an XOR of two version-4 UUIDs has version 0 and variant 0, and a
UUIDv8 fixes 6 of its 128 bits, leaving 122 free, too few to hold an arbitrary
record UUID's 122 random bits and 4 version bits reversibly.

**Why not a conditional upsert.** Qdrant 1.16 added `update_filter`
(qdrant/qdrant#7006): an upsert filtered on the writer's incarnation leaves a
point under another incarnation as it is and skips the writer's point without
an error. It kept bare ids, but a reused UUID's record was silently dropped,
callers still needed a uniqueness rule, and its cost stayed with the writer
(Qdrant 1.19.1, same setup):

| Request | p50 plain -> scoped | Server CPU per point |
|---|---|---|
| 1 new point | 1.53 -> 2.23 ms (+46%) | x1.85 |
| 1 point overwritten | 1.90 -> 2.00 ms (+6%) | x1.23 |
| 10 points | +12-14% | x1.10-1.20 |
| 100 points | +4% new, +39% overwrite (noisy) | x1.08 / x1.60 |
| 1,000 points | +1% new, +12% overwrite | x1.23-1.28 |
| 8 clients, 1 new point each | 4,384-4,684 -> 3,440-3,554 requests/s (-24%) | x1.10 per request |

## Filtered-search correctness (qdrant#10741)

With per-tenant graphs, a search filtered on a tenant *and* a second condition
can return nothing, or unrelated points, when the tenant's graph entry point
fails the second condition. Measured on Qdrant 1.19.1 (12 tenants of 100k
points, `m=0, payload_m=16`, a boolean set on half the points; top 10 filtered
on the tenant and the boolean, against the same query run exactly): 26 of 60
searches correct with the boolean indexed; 30 returned nothing while the exact
search returned 10; 4 returned 10 points sharing nothing with the exact top
10; none violated the filter. It first appears in 1.15.0 (0 of 60 wrong on
1.14.1, 15 on 1.15.0, 30-45 on later versions) and is reachable for tenants of
a few thousand points and up at default settings. The upstream fix,
qdrant/qdrant#10741, is open. The store is written as if that fix has
shipped: it has no graphless layout or exact-search workaround.

## Purge

One filter-delete of the whole incarnation per round: the round scrolls for
one point under the incarnation and, when it finds one, deletes by filter.
Measured (Qdrant 1.18.3, 1.19.0 and 1.19.1; 2.11M points in this layout, dead
incarnations of 10k to 1M among live tenants under search, upsert, and scroll
traffic; 2 CPUs / 4 GB): the delete stalls writes and scrolls on the shard,
never searches, for its duration, about 1.3 s per 1M points on 1.19.1 at 3
segments per shard (7.3 s at 101), with no errors. Deleting in batches instead
collapsed once a vacuum rebuilt the incarnation's segments mid-purge: every
batched 1M purge (7 of 7 runs, on all three versions) slowed to minutes-long
deletes and 60-120 s stalls for every tenant.

## Consistency

**One node.** Upserts and deletes pass `wait=True` and return once Qdrant has
applied them, so a later query reflects them; writes to a point apply in one
order. The store does not state this: the contract needs only a write durable
on return, which Qdrant gives once the write is in its write-ahead log, and
Milvus at Bounded gives no more, so a caller relies on neither store to see
its own writes at once.

**Why the writes wait.** Without `wait`, Qdrant replies once a write is in its
write-ahead log and applies it from a queue. Measured (Qdrant 1.19.1, 4 CPUs /
4 GB; each phase on a fresh collection of 100 tenants x 1,000 points of 768
dimensions in the store's per-tenant layout; two rounds):

| | `wait=True` | `wait=False` |
|---|---|---|
| One writer, 1 point: p50 | 1.5-1.7 ms | 0.6 ms |
| One writer, 10 points: p50 / p99 | 3.4-3.5 / 16-25 ms | 2.6-2.7 / 7-9 ms |
| One writer, 100 points: p50 / p99 | 23-24 / 88-181 ms | 21-22 / 33-76 ms |
| 800 points/s in batches of 10 beside 4 searchers: write p50 / p99 | 4.9-5.0 / 119-170 ms | 3.6 / 7-10 ms |
| The same: search p50 / p99 | 1.6 / 5 ms | 1.7 / 4 ms |
| 2,400 points/s: write p50 / p99 | 3.7-4.1 / 712-835 ms | 3.5-3.6 / 17 ms, and 3,969 ms in a round with a 4.4 s stall |
| The same: search p50 / p99 | 4.2 / 17-21 ms | 4.2 / 19-24 ms |
| Delay until a retrieve sees a write: p99 / max | none | 5-8 / 71-145 ms at 800 points/s; 137-362 / 226-4,847 ms at 2,400 |

Not waiting saves a writer about a millisecond at the median and much of its
tail at load. It changes neither the searches nor the CPU, nor what Qdrant
sustains: at three times the workload's peak both stalled, a waiting writer
for up to 1.3 s and, in one round, a writer that did not wait for 4.4 s. A
tenant's writes, a few small batches at a time, are well served either way,
so the store waits. Then the writes Qdrant has accepted but not applied are at
most the store's writes in flight, each one's wait visible to its caller as
latency, where without waiting they would build up unseen; and a failure to
apply reaches the caller.

**Replicated.** Qdrant's consistency documentation: a write succeeds once
`write_consistency_factor` replicas (1 by default) apply it; a query reads one
replica by default and can miss a write another replica has; and with the
default `weak` write ordering "write operations can be freely reordered",
while `medium` and `strong` serialize them through a leader. The store sets
none of these, and creates its native collection with the server's default
replication factor. It states no replicated read delay, which is unbounded.

Measured on a three-node Qdrant 1.19.1 cluster in Docker (1.5 CPUs / 1.5 GB
per node; one shard replicated on all three nodes; write consistency factor 1;
each node's reads served by its local replica, so reading from each node
compares the replicas; 10 ms +- 10 ms of network jitter on every node for the
hazards; 8 threads, 400 points per case):

| Case | `weak` | `medium` | `strong` |
|---|---|---|---|
| Upsert through node 1, then, once it returned, delete through node 2: replicas still holding the point | 0 | 0 | 0 |
| Upsert v1 through node 1, then v2 through node 2: replicas holding v1 | 0 | 0 | 0 |
| Two clients upsert different values of one point at once, through nodes 1 and 2: points whose replicas disagree | 115 of 400 | 0 | 0 |

So sequential writes kept their order under the default, but overlapping
writes to one point left replicas disagreeing (compared 5 s after the writes),
against Qdrant's documented `weak` semantics, not against the contract. The
cost of the stricter orderings falls on writes, so this measures writes alone,
no reads running: each phase on a fresh collection, 8 gRPC clients for 10 s,
three rounds with the orderings rotated:

| Upserts | `weak` | `medium` | `strong` |
|---|---|---|---|
| cluster, 1 point: per second (p50) | 2,331-2,612 (2.8-3.1 ms) | 1,560-1,781 (4.1-4.7 ms) | 1,659-1,748 (4.2-4.6 ms) |
| cluster, 10 points: per second (p50) | 1,616-1,649 (3.3-3.9 ms) | 1,047-1,313 (5.6-6.9 ms) | 1,228-1,307 (5.6-5.9 ms) |
| single node, 1 point: per second (p50) | 5,570-5,727 (1.2 ms) | 3,533-4,425 (1.7-1.9 ms) | 3,684-4,321 (1.7-1.9 ms) |
| single node, 10 points: per second (p50) | 1,968-2,103 (1.9 ms) | 1,988-2,096 (2.6-2.8 ms) | 1,918-2,126 (2.5-2.7 ms) |

The store passes no ordering: the contract does not promise convergence, and
nothing writes one point from two places at once (see
[consistency](vector_store_consistency.md)).

## Consequences

- A Qdrant tenant's filtered searches can be wrong on Qdrant 1.15 and later
  until qdrant#10741 ships.
