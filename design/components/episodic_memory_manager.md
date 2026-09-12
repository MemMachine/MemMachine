# EpisodicMemoryManager

New component; the name is repurposed from the current
`episodic_memory/episodic_memory_manager.py`, which is not carried over.
The resource that stands in for a family of `EpisodicMemory` objects:
builds one per request, bound to the tenant's handles and
configuration, runs the search stages on it, fills per-request
defaults, validates tenant configuration, and registers with the tenant
service as the `episodic_memory` memory subsystem.

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
restrict the offered ids to a subset of the mappings; `settings.filter`
holds `max_overfetch_factor`, a multiple of `limit`; `settings.cache_size`
bounds the segmenter and deriver cache.

## Tenant configuration model

```python
class RerankOptions(BaseModel):
    reranker: str                       # an offered id
    candidates: int                     # hits the vector stage returns to it
    min_score: float | None             # on the reranker's score

class SearchOptions(BaseModel):
    limit: int
    min_cosine_similarity: float | None
    expand_context: int
    rerank: RerankOptions | None

class EpisodicMemoryTenantConfig(BaseModel):
    embedder: str                       # immutable; an offered id
    segmenter: SegmenterOptions         # mutable; later events; per block kind (blocks.md)
    deriver: DeriverOptions             # mutable; later events; per block kind
    format: FormatOptions               # mutable; later events
    eviction: EvictionOptions | None    # mutable; later batches; None: off
    search: SearchOptions               # mutable; defaults for a search
```

`eviction.cosine_similarity_threshold` is calibrated per embedder, so a
template sets it beside the `embedder` it names; the manager rejects
no value, since the design gives none.

`SearchOptions` is one model with two uses: in the tenant section every
field is set and is the default; in a search request every field is
optional and overrides the default. The request adds `query`, the
system filters, `filter` and `format`. There is no second list of
parameters to keep in step.

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
SQLite), so it moves only forward; `provision` writes `config` and
`config_version` only and never touches it.

## API

Toward the tenant service (`MemorySubsystem`):

- `provision(tenant_id, section)`: `segment_store.create_partition` and
  `vector_store.create_collection(key, container=section.embedder)`,
  each treating its own `live` row as success and resuming `creating`;
  insert the per-tenant row, or update its `config` and
  `config_version`.
- `delete(tenant_id)`: `segment_store.delete_partition`,
  `vector_store.delete_collection`, remove the per-tenant row.
- `purge(tenant_id)`: `purge_partition` and `purge_collection`;
  `DONE` when both are.
- `validate_update`: `embedder` changed raises.
- `replay`, `watermark`: below.

Toward the routers:

```python
    async def search(self, tenant_id: UUID, request: SearchRequest) -> list[QueryHit]
    async def expand(self, tenant_id: UUID, request: ExpandRequest) -> Neighborhood
    async def watermark(self, tenant_id: UUID) -> int
```

Each reads the per-tenant row (absent: `ComponentNotEnabledError`,
which the router turns into 404 or 409 by asking the tenant service),
builds the tenant's `EpisodicMemory` in one constructor call from
`segment_store.partition(tenant_id)`, `vector_store.collection(tenant_id,
e)`, `embedders[e]`, the row's `format` and `eviction`, and the
segmenter and deriver objects for the row's options, taken from the
cache, and makes one call.

`search` is the stages. It fills each `SearchOptions` field the request
omits from the row's defaults. Without `rerank`, it calls `query` with
`limit` and `min_cosine_similarity` and returns the hits. With `rerank`, it
calls `query` with `limit = candidates` and `min_cosine_similarity`, renders
each hit's window with the request's or the row's `format`, scores the
renderings with `rerankers[rerank.reranker].score`, drops those below
`rerank.min_score`, and returns the best `limit` in descending reranker
score with `score` replaced by it. Over-fetching is one limit set above
another, and each stage has its own threshold on its own scale; an id
not offered raises `InvalidTenantConfigError`. The router renders
`text` per hit with the same `format`.

`replay` reads the log after the watermark in batches, calls `encode`
with the `added` entries' events and `forget` with the `deleted`
entries' uuids, and advances the watermark to the batch's last position
in its own transaction after `encode` and `forget` have returned, that
is, after both the segment store and the vector store hold the batch.
A step that fails before that leaves the watermark, and the next step
redoes the batch from it. When the watermark is below the log's oldest
entry (a subsystem enabled on a tenant with history, or a compacted
log), the step reads `read_events_after` instead and encodes those
events until it reaches the log. `DONE` when the watermark is at the
head; `MORE` otherwise. A step that finds no per-tenant row returns
`DONE`.

## Cache

Segmenter and deriver tables (`blocks.md`, "Processing"), keyed by
their options, never by tenant;
bounded by `settings.cache_size`. `EpisodicMemory` objects and handles
are not cached: each is a few references, built per request and
discarded, so nothing bound to a tenant outlives the request.

## What it does not do

No segmenting, deriving or vector search of its own, no model
translation, no per-tenant state beyond its table, no reading of the
tenant tables.

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
(tenant, subsystem) except when a slow step is reclaimed, in which case
two run and idempotency makes the second harmless.

- Watermark semantics: every position at or below the watermark has
  been fully processed at least once, segments and vectors both. The
  watermark only moves forward: it is written with `SET watermark =
  GREATEST(watermark, ?)`, after the batch's writes.
- A `replay` step that fails leaves the watermark, records the error on
  the job, and the next attempt resumes from the watermark; processing
  is idempotent per event (forget first), so a re-run leaves one copy.
- Read-your-writes: an acknowledged ingest is durable in the event
  store; it is visible to search after the job has processed it and
  the vector backend has indexed it, which `?wait=` and the status
  endpoint expose.

| First | Concurrent | Outcome |
| --- | --- | --- |
| ingest batch A (positions 1..10) | ingest batch B (11..20) on another process | the event store serializes the two under the key's exclusive lock, so positions are commit-ordered; both reset the one `replay` row; one step processes both batches in order, and the other process's inline claim finds the row running and returns |
| replay step fails mid-batch | | the watermark stays; the next attempt resumes from it and reprocesses the batch idempotently; nothing is skipped |
| replay step crashes after the vector upsert, before the watermark write | | the next attempt redoes the batch: `encode` forgets each event's derived rows first, so one copy remains |
| ingest of event uuid U | ingest of U on another process | unique `(key, uuid)`: one stores, the other reports U skipped; the log holds one `added` entry |
| delete of event U | replay processing U's `added` entry | the delete appends a `deleted` entry after the `added` one; the consumer replays them in order, so the derived rows are written and then forgotten; nothing depends on the client retrying |
| search | ingest | the search sees what the replay has processed and the backends have indexed; `?wait=` on the ingest is how a client sequences the two |
| replay, subsystem A | replay, subsystem B, same tenant | independent job rows; they run in parallel and touch different derived stores |
| replay step | configuration update's provision step | `provision` writes `config` and `config_version` only, so the watermark is untouched; the next replay step uses the new options |
| replay step | tenant delete | `delete` removed the per-tenant row: the step returns `DONE`; rows it had written are purged by the sweep |
