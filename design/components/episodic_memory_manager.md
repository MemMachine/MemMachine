# EpisodicMemoryManager

New component; the name is repurposed from the current
`episodic_memory/episodic_memory_manager.py`, which is not carried over.
The resource that stands in for a family of `EpisodicMemory` objects:
builds one per request, bound to the tenant's handles and
configuration, dispatches the request to it, fills per-request
defaults, validates tenant configuration,
and registers with the tenant service as the `episodic_memory`
component.

## Constructed with

```python
EpisodicMemoryManager(
    event_store: EventStore,
    segment_store: SegmentStore,           # lifecycle, and partition handles
    vector_store: VectorStore,             # lifecycle, and collection handles
    embedders: Mapping[str, Embedder],     # resources, all the deployment built
    rerankers: Mapping[str, Reranker],
    engine: AsyncEngine,                   # its per-tenant table
    settings: EpisodicMemorySettings,      # offered subsets, filter bounds, cache size
    metrics_factory: MetricsFactory | None,
)
```

The mappings hold resources; products of the manager never appear in
settings. `settings.embedders` and `settings.rerankers`, when given,
restrict the offered ids to a subset of the mappings.

## Tenant configuration model

```python
class EpisodicMemoryTenantConfig(BaseModel):
    embedder: str                       # immutable; an offered id
    segmenter: SegmenterOptions         # mutable; later events
    deriver: DeriverOptions             # mutable; later events
    reranker: str | None                # mutable; default for query
    search: SearchDefaults              # mutable; defaults for query
```

`SearchDefaults` has exactly the fields of `query`'s per-request
parameters (`limit`, `expand_context`, `min_score`, `include_events`),
checked at import time against `EpisodicMemory.query`'s signature. The
structural key of a configuration is `(embedder, segmenter, deriver)`.

## Schema

`episodic_memory_tenants`:

| column | type | constraint |
| --- | --- | --- |
| `tenant_id` | `Uuid` | primary key |
| `watermark` | `BigInteger` | not null, default 0; the last log position processed |
| `config` | `JSON` (`JSONB` on PostgreSQL) | not null |
| `config_version` | `Integer` | not null |
| `updated_at` | `DateTime(timezone=True)` | not null, `func.now()` |

No other index: every access is by primary key. The watermark is
written with `SET watermark = GREATEST(watermark, ?)` (`MAX` on
SQLite), so it moves only forward.

## API

Toward the tenant service (the `TenantComponent` protocol):

- `provision(tenant_id, section)`: `segment_store.create_partition` and
  `vector_store.create_collection(key, container=section.embedder)`,
  each treating its own `live` row as success and resuming `creating`;
  insert or update the per-tenant row with the section and version.
- `delete(tenant_id)`: `segment_store.delete_partition`,
  `vector_store.delete_collection`, remove the per-tenant row.
- `purge(tenant_id)`: `purge_partition` and `purge_collection`;
  `DONE` when both are.
- `validate_update`: `embedder` changed raises.
- `replay`: below.

Toward the routers:

```python
    async def search(self, tenant_id: UUID, request: SearchRequest) -> SearchResponse
    async def expand(self, tenant_id: UUID, request: ExpandRequest) -> Expansion
    async def status(self, tenant_id: UUID) -> SubsystemStatus   # watermark, head, lag
```

Each reads the per-tenant row (absent: `TenantNotFoundError`, which the router
turns into 404 or 409 by asking the tenant service), builds the tenant's
`EpisodicMemory` in one constructor call from
`segment_store.partition(tenant_id)`, `vector_store.collection(tenant_id, e)`,
`embedders[e]` and the segmenter and deriver objects for the row's options,
taken from the cache, and makes one call. `search` fills each request parameter
the request omits from the row's defaults, resolves `request.reranker` or the
default to an object (`InvalidTenantConfigError` for an id not offered), and
calls `query`. `replay` calls `encode` with the events of a batch's `added`
entries and `forget` with the uuids of its `deleted` entries, and advances the
watermark in the same transaction as the batch's last segment write where the
engines are shared, and after it otherwise.

## Cache

Segmenter and deriver objects, keyed by their options, never by tenant;
bounded by `settings.cache_size`. `EpisodicMemory` objects and handles
are not cached: each is a few references, built per request and
discarded, so nothing bound to a tenant outlives the request.

## What it does not do

No search or ingest logic, no model translation, no per-tenant state
beyond its table, no reading of the tenant table.

## Changes to existing code

Replaces `EpisodicMemoryManager` and `MemoryInstanceCache`
(`episodic_memory/episodic_memory_manager.py:63`,
`instance_lru_cache.py:32`), `EpisodicMemory` and `LongTermMemory` as
facades (`episodic_memory/episodic_memory.py:94`,
`long_term_memory/long_term_memory.py:177`), and
`long_term_memory/service_locator.py`. Nothing is carried over.

## Data-path races

The watermark and the `replay` job are the parts that need a rule. Two
processes may ingest into one tenant at once, positions are assigned
under the key's exclusive lock, and the job is one consumer per
(tenant, subsystem).

- Watermark semantics: every position at or below the watermark has
  been processed at least once. The watermark only moves forward: it
  is written with `SET watermark = GREATEST(watermark, ?)`.
- A `replay` step that fails leaves the watermark, records the error on
  the job, and the next attempt resumes from the watermark; processing
  is idempotent per event (forget first), so a re-run leaves one copy.
- Read-your-writes: an acknowledged ingest is durable in the event
  store; it is visible to search after the job has processed it and
  the vector backend has indexed it, which `?wait=` and the status
  endpoint expose.

| First | Concurrent | Outcome |
| --- | --- | --- |
| ingest batch A (positions 1..10) | ingest batch B (11..20) on another process | the event store serializes the two under the key's exclusive lock, so positions are contiguous and commit-ordered; each batch is processed by its own process; the watermark ends at 20 by `GREATEST` whatever the processing order |
| replay step fails mid-batch | | the watermark stays; the next attempt resumes from it and reprocesses the batch idempotently; nothing is skipped |
| ingest of event uuid U | ingest of U on another process | unique `(key, uuid)`: one stores, the other reports U skipped; the log holds one `added` entry |
| delete of event U | replay processing U's `added` entry | the delete appends a `deleted` entry after the `added` one; the single consumer replays them in order, so the derived rows are written and then forgotten; nothing depends on the client retrying |
| search | ingest | the search sees what the replay has processed and the backends have indexed; `?wait=` on the ingest is how a client sequences the two |
| replay, subsystem A | replay, subsystem B, same tenant | independent job rows; they run in parallel and touch different derived stores |
