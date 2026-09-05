# Vector store

Existing component, `common/vector_store/`, reworked. Derived data: an
index of derivative embeddings, rebuildable from the event store.
Queries return ids and scores, never properties.

## Constructed with

- `<Backend>VectorStore(client, registry: KeyRegistry,
  settings: VectorStoreSettings)`, where `registry` is the scoped view
  from `key_registry.md` and the client is the backend's (an
  `AsyncQdrantClient`, a Milvus client, an engine for pgvector and the
  SQLite stores, for which the registry is a table beside the data).
- Settings: `containers: Mapping[str, ContainerSettings]`, one per
  embedder id, each with `dimensions` and any backend-specific option
  (the metric is cosine, always); `indexed_properties: Mapping[str,
  PropertyType]`, one schema for every container of the store, plus
  the system fields, which are always declared; `request_timeout`,
  required.

## Types

```python
class Record(BaseModel):
    uuid: UUID
    vector: list[float]
    properties: dict[str, PropertyValue]     # declared keys only

class QueryMatch(BaseModel):
    uuid: UUID
    score: float                             # cosine similarity

class QueryResult(BaseModel):
    matches: list[QueryMatch]                # descending score, at most `limit`
```

## API

Two ABCs: the store, the resource and the only place a key is named
(container provisioning, lifecycle, and constructing handles); and the
collection, the stateless handle every data consumer holds, bound to
one key and one container, no method on it taking either. Each backend
implements both, as the current code does with `VectorStoreCollection`,
minus the open and close and the stale state. `EpisodicMemoryManager`
holds the store: its hooks call lifecycle, and each request builds the
`VectorCollection` an `EpisodicMemory` receives.

```python
class VectorStore(ABC):                   # the resource: lifecycle, and handles
    async def provision_containers(self) -> None          # schema command only
    async def create_collection(self, key: UUID, container: str) -> None
    async def delete_collection(self, key: UUID) -> None
    async def purge_collection(self, key: UUID) -> Progress
    def collection(self, key: UUID, container: str) -> VectorCollection
        # stateless handle, no I/O
    @property
    def concurrency_scope(self) -> ConcurrencyScope

class VectorCollection(ABC):              # data, bound to one key and container
    @property
    def key(self) -> UUID
    @property
    def container(self) -> str
    async def upsert(self, records: Iterable[Record]) -> None
    async def delete(self, uuids: Iterable[UUID]) -> None
    async def query(self, vectors: Iterable[Sequence[float]], *,
                    limit: int, min_score: float | None,
                    filter: FilterExpr | None,
                    allowed_uuids: Iterable[UUID] | None) -> list[QueryResult]
    async def get_cosine_similarity(self, vector: Sequence[float],
                                    uuids: Iterable[UUID]) -> dict[UUID, float]
    @property
    def supported_filter_nodes(self) -> frozenset[type]
```

Scores are cosine similarity everywhere; there is no `SimilarityMetric`
(reference branch, commit 6ab12098): every container and every engine
is configured for cosine, the embedder exposes no metric, and `query`
takes `min_score` as a cosine similarity.

Semantics:

- `provision_containers`: idempotently create every container the
  settings declare, with the store's one indexed-property schema. Run by
  `memmachine schema upgrade`, never by a request.
- `create_collection(key, container)`: strict. Where a tenant is a value
  inside the container: one registry row, `live`, address `{container}`.
  Where a tenant is a native object (Chroma collection, Weaviate
  tenant): row as `creating`, create the object, set `live` with its
  address (`{container, collection_id}`); a caller resuming a
  `creating` row creates the object if absent and sets `live`.
- `upsert`, `delete`: a record carrying an undeclared property key
  raises `UndeclaredPropertyKeyError` before anything is sent. Read the
  registry row (not `live`: `KeyNotLiveError`); perform the remote
  write, acknowledged as applied (Qdrant `wait=True`; Milvus strong
  consistency); read the row again and raise `KeyNotLiveError` if it is
  no longer `live`.
- `query`: `filter` names declared keys only and raises
  `UndeclaredPropertyKeyError` otherwise, and uses only nodes in
  `supported_filter_nodes`, raising `UnsupportedFilterError` otherwise;
  evaluated during the search. `allowed_uuids` restricts the search to
  those records. Read the row, query, read the row again. Returns at
  most `limit` matches per vector, fewer when the filter admits fewer.
- `supported_filter_nodes`: the node classes the backend evaluates
  during a search, per the table in `filters_and_properties.md`; the
  subsystem routes any other predicate to the segment store.
- `get_cosine_similarity`: score given records against a vector,
  fenced the same way; the allowlist plan's scoring step.
- `delete_collection`: `set_state(key, DROPPING)`; O(1); idempotent.
- `purge_collection`: with a `dropping` row, delete records under the
  key in bounded steps (filter delete; a filtered-query loop and keyed
  delete on S3 Vectors; `delete_collection` on Chroma; remove the
  tenant on Weaviate); `MORE` while records remain; remove the row when
  none do and return `DONE`. With no row, delete by key in every
  container the store has; `DONE` when nothing is found. `purge` on
  a `live` row raises; the tenant service never calls it on one.
- `collection(key, container)`: builds the handle without I/O. Every
  operation through it reads the registry row and raises
  `KeyNotLiveError` when the row is not `live` or names another
  container, so a handle built for the wrong container cannot write.
  `EpisodicMemoryManager` builds one per request for the tenant's key
  and its embedder's container and hands it to `EpisodicMemory`.

## Backends

| Backend | Tenant inside the container | Address | Rejects a dead key itself | Purge |
| --- | --- | --- | --- | --- |
| Qdrant | payload value | container | no | filter delete |
| Milvus | partition-key value | container, loaded once | no | filter delete |
| pgvector | column value | table | yes, in-statement | keyed delete |
| Pinecone | namespace parameter | index host | no | delete all in namespace |
| S3 Vectors | filterable metadata value | bucket, index | no | filtered query, keyed delete, repeat |
| Weaviate | native tenant | collection, tenant name | yes | remove tenant |
| Chroma | collection per tenant | collection UUID | yes | delete collection |
| sqlite-vec | partition-key value in one `vec0` table per container | table | yes, in-statement | keyed delete |
| usearch | shared records table, one index file per tenant | table, file path | yes, in-statement | keyed delete, unlink |

## Concurrency scope

The minimum of the backend's and the registry's: networked backends
`cluster`; local-mode Qdrant and Milvus `process`; sqlite-vec `host`;
the usearch store `process`.

## Changes required

- The registry leaves the backend (`qdrant_vector_store.py:609`,
  `milvus_vector_store.py:477`, the `_CollectionRow` tables in both
  SQLite stores) for the key registry, or, for pgvector and the SQLite
  stores, a table beside the data.
- `VectorStoreCollection` (`vector_store.py:22`) becomes
  `VectorCollection`: the same data operations, bound to the key and
  the container at construction, stateless; `open_collection`,
  `open_or_create_collection` and `close_collection` (`:205`, `:235`,
  `:254`) go and `collection(key, container)`, which does no I/O,
  replaces them. `VectorStoreCollectionConfig`
  and the `config` parameter of `create_collection` (`:175`) go
  (#1573); the container's shape comes from `containers` settings.
- Keys are `UUID`; `validate_identifier` (`utils.py:31`) applies to
  property keys and container names only.
- Content-addressed native names (`_build_native_collection_name`,
  `qdrant_vector_store.py:619`) go; a container is named by its
  embedder id and provisioned by the schema command (#1572).
- Qdrant's shard key per collection and `_name_locks` go (#1564);
  payload partitioning by the key's hex is the one mode.
- `query` returns ids and scores only; `get` and `return_vector` go;
  `get_cosine_similarity` is added (reference branch, commit 2d5dc2b5).
- Undeclared property keys are rejected on write and query.
- Post-operation registry checks replace the absent fence (#1537,
  #1563); `purge_collection` and container retirement are added
  (#1565).
- The sqlite-vec store: one `vec0` table per container with the key as
  partition key, `chunk_size` a setting, instead of a table per
  collection (`sqlite_vec_vector_store.py:692`); the usearch store: a
  shared records table plus one index file per key, its `SQLiteVectorStore`
  per-collection tables (`sqlite_vector_store.py:1061`) go.
- Every client is constructed with `request_timeout`.

## Schema of the SQL-backed stores

pgvector, one table per container, created by `provision_containers`:

`vec_<container>`:

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key part |
| `uuid` | `Uuid` | primary key part |
| `vector` | `VECTOR(<dimensions>)` (the `pgvector` SQLAlchemy type) | not null |
| `memmachine_event_timestamp` | `DateTime(timezone=True)` | not null |
| `memmachine_event_session` | `Text` | null |
| `memmachine_event_source` | `Text` | null |
| `memmachine_block_kind` | `Text` | not null |
| `memmachine_event_uuid` | `Uuid` | not null |
| `memmachine_segment_uuid` | `Uuid` | not null |
| one column per declared user key | by declared type: `Text`, `BigInteger`, `Float`, `Boolean`, `DateTime(timezone=True)` | null |

Indexes: `vec_<container>__vector`, HNSW with `vector_cosine_ops`;
`vec_<container>__key_timestamp (key, memmachine_event_timestamp)`;
`vec_<container>__key_<field>` for each declared user key. Filtered
search is `WHERE key = ? AND ...` with pgvector's iterative index
scans, and the registry row for this store is `vector_store_pt` beside
the table, keyed by `key`, so the fence is in-statement.

sqlite-vec, one `vec0` virtual table per container plus a records
table:

```sql
CREATE VIRTUAL TABLE vec_<container> USING vec0(
    key TEXT PARTITION KEY,          -- 32 hex characters
    vector FLOAT[<dimensions>] distance_metric=cosine,
    memmachine_event_timestamp INTEGER,      -- metadata column, epoch seconds
    memmachine_event_session TEXT,
    memmachine_event_source TEXT,
    memmachine_block_kind TEXT,
    <declared user key> <TEXT|INTEGER|FLOAT|BOOLEAN>, ...
    chunk_size=<settings.chunk_size>
);
```

`vec_<container>_rec`: `key Uuid` and `uuid Uuid` primary key,
`rowid BigInteger` not null unique (the vec0 rowid), and
`memmachine_event_uuid Uuid`, `memmachine_segment_uuid Uuid` not null.
The vec0 table's metadata columns carry every declared filterable key;
the records table maps record uuids to rowids for `delete` and
`get_cosine_similarity`. The registry row is `vector_store_pt` in the
same file.

Engine-backed store (usearch, hnswlib, or turbovec engines, as the
reference branch's `VectorSearchEngine` family), one shared records
table and one index file per key:
`vec_<container>_rec` with `key`, `uuid`, `rowid` as above plus the
declared columns for post-filtering; the index file at
`<settings.index_dir>/<container>/<key hex>.usearch`, loaded into
process memory on first use and written back on change, which is why
the store is `process`-scoped. Two fixes from the reference branch are
folded in: an index file is published atomically (written beside, then
renamed over; commit 397a55cb), and engine row ids are reused after
deletion by default (commit cafc20c7).
