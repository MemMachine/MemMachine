# Vector store: implementation requirements

A handoff for the agent implementing the accepted vector store changes
of the server redesign, cut so that none of it depends on the tenant
service, jobs, tombstones, handles or the key registry.

## The seam

The vector store changes fall on two sides of one line.

Records and queries: what a record carries, which keys are indexed,
what a filter may say, which nodes each backend evaluates during a
search, how a score is read back, how datetimes are stored, what a
client needs. None of this cares how a collection is addressed or how
it lives and dies. This document is that side.

Collections and keys: UUID keys minted once, one native container per
embedder with the tenant as a value inside it, registry rows in the key
registry or beside the data, strict create, logical delete, batched
purge, the check after each remote operation, stateless handles, the
SQLite stores on shared tables, containers provisioned by the schema
command, and the removal of incarnations, content-addressed names and
Qdrant shard keys. All of it rests on a key that is never reused, which
is what the tenant service's tombstone provides, so it lands with the
lifecycle work (`design/components/vector_store.md`, "Changes required",
and `design/server_redesign.md`, "Vector store"). Doing it now would
mean either incarnations, which the design then removes, or a
never-reused key nothing yet guarantees. Everything on this side of
the line is written against `VectorStoreCollection` as it is today, so
the later change replaces how a handle is obtained and nothing this
document adds.

## Base and order

- Branch off upstream `speedkick` after the #1603 stack, #1598 (cosine
  scores and uuids; replaces #1591 and #1593), #1602 (filtered queries
  routed by selectivity inside the engine-backed SQLite store, with an
  exact engine allowlist) and #1603 (`Record` requires a vector and
  never has `None` properties), and after #1597 (the EventMemory
  handoff, stacked on #1598, which owns the reserved keys,
  `system_filters.py`, and the segment store's `session_id`,
  `source_id` and `block_kind` columns). #1599 (turbovec engine, which
  takes #1602's allowlist natively) is independent. #1588 (atomic
  index publish) is merged.
- #1598's decision that the store scores nothing by id stands, and
  #1602 keeps it: its allowlist is the engine's interface, reached only
  by the store's own regime, never by a caller. The plan below needs
  no such call.
- `Record` is what #1603 made it: `uuid`, a required `vector`, and
  `properties` defaulting to `{}`, matching `vector_store.md`.

## References

- `speedkick` in this worktree, `packages/server/src/memmachine_server/`:
  `common/vector_store/vector_store.py` (the ABCs: `VectorStoreCollection`
  from line 22, `VectorStore` from 147), `common/vector_store/data_types.py`
  (`VectorStoreCollectionConfig` at 18 with `indexed_properties_schema`,
  `Record`, `QueryMatch`, `QueryResult`), the four stores and their
  filter compilers (`qdrant_vector_store.py:88` `_build_qdrant_filter`,
  `milvus_vector_store.py:92`, `sqlite_vec_vector_store.py:254`,
  `sqlite_vector_store.py:414` `_build_key_filter`), the engines under
  `common/vector_store/vector_search_engine/`, `common/filter/` (the
  string parser and `sql_filter_util.py`).
- The `default` branch of `edwinyyyu/MemMachine`, worktree
  `/Users/eyu/edwinyyyu/mmcc/default`:
  - `822ccb6b`, the closed filter union
    (`common/filter/filter_expression.py`) and the per-backend
    compilers recompiled over it.
  - `2d5dc2b5`, score-only queries; its selective and broad regimes
    live inside the engine-backed SQLite store, where #1602 ports them
    to `speedkick` (a LIMIT probe over the records table, one row past
    `selective_filter_limit`; an exact engine allowlist when
    selective, an unrestricted search post-filtered with widening up
    to `limit * max_overfetch_factor` when broad), and its
    `get_cosine_similarity` is not taken (zero callers; YAGNI, decided
    2026-09-09).
  - `b17fd0a1`, the widening cap returns what survived.
  - `a0753d3c`, Qdrant `hnsw_config`, `optimizers_config` and
    `quantization_config` as plain mappings validated against the
    Qdrant models, with the `m = 0` rule.
  - `27b3279b`, `common/property_keys.py` (already taken by #1597).
- Design: `design/components/vector_store.md` (contract, backends
  table, schemas), `design/components/filters_and_properties.md`
  (union, provider support table, normalizations, `split_declared`),
  `design/server_redesign.md` under "Properties and filtering" (why
  declared indexes, routing, limits are maximums).

## Declared, not dynamic, indexes

Today each collection carries its own `indexed_properties_schema` in
`VectorStoreCollectionConfig`, and `EventMemory` tells the collection
what it expects. That is the dynamic index creation the redesign
prohibits: a filter index appears per collection, from a request's
configuration, with no owner and no migration.

- The store declares one schema for every collection it holds:
  `indexed_properties: Mapping[str, PropertyType]` in the store's
  params, from deployment configuration. `PropertyType` is the scalar
  set (`str`, `int`, `float`, `bool`, `datetime`).
- The system keys are always declared, by the consumer that writes
  them. `EventMemory.expected_vector_store_collection_schema` stays as
  that declaration (the four reserved keys #1597 writes:
  `memmachine_event_timestamp: datetime`, `memmachine_event_session:
  str`, `memmachine_event_source: str`, `memmachine_block_kind: str`),
  and the schema a store is built with is the configured user keys
  plus its consumer's system keys, merged where the store is
  constructed for that consumer. There is no central list of reserved
  keys: the prefix is reserved as a whole, each service names its own
  keys under it, and services do not share a vector store, so two
  services' keys never meet in one schema.
- `indexed_properties_schema` leaves `VectorStoreCollectionConfig`;
  `create_collection` and `open_or_create_collection` keep their
  shape otherwise (their removal is the other side of the seam).
  `EventMemory` checks at construction that the collection's store
  declares its four system keys and raises `InvalidCollectionSchemaError`
  if not.
- The server's configuration gains the store-level
  `indexed_properties` setting and stops passing a schema per
  collection; that is the one wiring change, and it is a config field.

## Undeclared keys are rejected

- `upsert` raises `UndeclaredPropertyKeyError` before anything is sent
  when a record carries a key the store has not declared.
- `query` raises the same when a filter names an undeclared key. An
  undeclared key therefore never exists in the vector store: not stored
  write-only, not scanned for. The split between what the store filters
  and what the segment store filters is the caller's job (below), never
  something a store guesses at.
- Every store keeps storing declared properties and filtering on the
  stored copy; nothing returns them (#1598).

## The filter language

- Take the closed union from `822ccb6b`: `Equals`, `NotEquals`,
  `Ordering` (numbers and datetimes only), `In` (homogeneous,
  non-empty), `IsMissing`, n-ary `And` and `Or`, `Not`, in
  `common/filter/filter_expression.py`. `Comparison` and `IsNull` go.
  The string parser may stay as the server's translation into the new
  union for now, since the HTTP API still speaks it; `filter_from_json`
  from the same commit is the JSON form the redesign's API will use.
- Each store compiles the tree with an exhaustive `match`, so a node
  it does not handle is a type error, and declares
  `supported_filter_nodes: frozenset[type]`, the node classes it
  evaluates during the search, per the table in
  `filters_and_properties.md`: Qdrant, Milvus and the SQL stores
  evaluate everything; sqlite-vec evaluates `Equals`, `NotEquals`,
  `Ordering` and `And` only (its KNN takes comparisons joined by
  `AND`); the engine-backed store resolves the tree over its records
  table in SQL (#1602's regime) and reports the SQL set. A `query`
  whose tree uses a node outside the set raises
  `UnsupportedFilterError`. The declared schema gives that records
  table one typed column per declared key, which is what #1602's probe
  and post-filter then run over.
- Two normalizations belong to the compilers that need them: `Not` is
  pushed to the leaves by De Morgan where a backend has no negation
  node, and `NotEquals` compiles to "not equal and exists" where a
  backend's inequality would match records lacking the key. Semantics
  are the design's: a predicate matches only a record holding a value
  of the compared type; `NotEquals` keeps records holding a differing
  value; `Not(Equals)` also keeps records holding none.

## No scoring by id

`get_cosine_similarity` is not added, and no allowlist parameter on
`query` either. Both would exist to score a candidate set assembled
outside the store, which the plan below never does: the vector store
gets the declared part of a filter and applies it during the search,
and the undeclared part is applied afterward by the segment store,
which already holds every segment's properties. The `default` branch
added `get_cosine_similarity` for a consumer that never arrived, and
#1598 removed it with `get`; it stays removed.

## Datetimes

Where a backend has no datetime type (sqlite-vec metadata columns, the
engine-backed store's records table, and S3 Vectors when it arrives),
a datetime property, system or user, is stored as an integer of
microseconds since the epoch, the same precision the SQL stores keep,
so a `since` or `until` bound evaluates identically in every store. A
bound is normalized to UTC before it is compared, as the segment store
already does.

## Clients

Every backend client params model has a required `request_timeout`,
passed to the client at construction (`AsyncQdrantClient(timeout=...)`,
the Milvus client's timeout). The design's tombstone retention rests
on every remote write being bounded by it, so it is required, not
defaulted.

## Qdrant collection options

From `a0753d3c`: `hnsw_config`, `optimizers_config` and
`quantization_config` on the Qdrant store's params as plain mappings,
validated against the `qdrant_client.models` types when the store is
built so the client stays an optional import, applied to data
collections only. In payload-partitioned mode `hnsw_config.m` must be 0
or unset, since a multi-tenant collection disables the global graph in
favor of per-tenant payload indexing, and `payload_m` is the knob; the
store rejects any other `m`.

## What EventMemory does with it

This is the consumer side of the contract and belongs with it; the
author of #1597 is the natural owner.

- `EventMemoryParams` gains `filter: FilterOptions` with one field,
  `max_overfetch_factor: int`, a multiple of `limit`, named as #1602
  names the store's own cap. The two caps are two layers: the store's
  bounds its broad regime over declared keys; this one bounds the
  widening below over undeclared keys, and a query with no undeclared
  part never reaches it.
- `split_declared(expr, declared) -> tuple[FilterExpr | None,
  FilterExpr | None]` (`filters_and_properties.md`): the part of a
  conjunction naming declared keys only, and the rest; a disjunction
  or negation that mixes the two is undeclared as a whole. Applied to
  the caller's `property_filter` against the collection's declared
  schema; the system predicates are declared by construction.
- One plan in `query`. The vector store gets the declared part and the
  system predicates, evaluated during the search. When there is an
  undeclared part, `get_segments` with the same system values
  and the undeclared part is the post-filter: a seed whose segment the
  store does not return is dropped. The vector `limit` is widened, up
  to `max_overfetch`, while dropped seeds leave fewer than `limit`
  hits, `limit * max_overfetch_factor` at most; at the cap the search
  returns what survived (`b17fd0a1`, and #1602 for the store's own
  cap). With no undeclared part the first `query` is the last.
- Every count is a maximum: a filtered search returns fewer when the
  filter admits fewer, and nothing promises exactly `limit`.

## Tests

- Per backend: an undeclared key on `upsert` and in a filter raises;
  each node in `supported_filter_nodes` evaluates during a search and
  each node outside it raises `UnsupportedFilterError`; `NotEquals`
  excludes records lacking the key and `Not(Equals)` includes them;
  `IsMissing` matches absence, never null; a datetime bound at
  microsecond precision behaves the same on sqlite-vec, the engine
  store and pgvector.
- EventMemory: an undeclared predicate never reaches the vector store;
  widening stops at `limit * max_overfetch_factor` and returns what
  survived; a fully declared filter issues one query.
- Do not test ranking with a fake embedder that ties every score under
  cosine; use one whose vectors differ per text and assert the
  contract, not an exact list. Run each new store test against the
  unfixed store first.

## Not in this change

UUID keys and never-reused keys, the key registry and registry rows,
strict create, logical delete and batched purge, the check after a
remote operation, stateless handles and the removal of `open_*`, one
container per embedder with tenants as values, the SQLite stores on
shared tables and the `vec0` partition key, container provisioning by
the schema command, content-addressed names and Qdrant shard keys
going, the six surveyed remote backends, and the engines' durability
contract (#1588, #1599, #1602's engine allowlist). The interim
registry stack (#1526 to #1533,
being resliced) and #1537 cover the collection side until the lifecycle
work lands; nothing here assumes or forbids them.
