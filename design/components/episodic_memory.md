# EpisodicMemory

Existing component, `episodic_memory/event_memory/event_memory.py`,
`EventMemory`, renamed. Processes events into segments and derivative
embeddings and answers vector searches over them. A configured object
bound to one tenant: built by `EpisodicMemoryManager` per request from
the tenant's handles and configuration, never by the composition, never
cached, interchangeable with any other built the same way; a library
user builds one directly.

## Constructed with

```python
EpisodicMemory(
    partition: SegmentPartition,    # handle: this tenant's segments
    collection: VectorCollection,   # handle: this tenant's records in its embedder's container
    segmenter: Segmenter,           # table from block kind to BlockSegmenter (blocks.md, "Processing")
    deriver: Deriver,               # table from block kind to BlockDeriver
    embedder: Embedder,
    format_options: FormatOptions,  # how the deriver renders dates and names into embedded text
    eviction: EvictionOptions | None,   # None: no eviction
    metrics_factory: MetricsFactory | None,
)

class EvictionOptions(BaseModel):
    cosine_similarity_threshold: float  # at or above it, eviction is considered
    search_limit: int               # stored neighbors consulted per new derivative
    target_size: int                # a cluster larger than this is trimmed to it
```

One tenant, one embedder, one segmenter, one deriver, one way of
formatting the text that is embedded. No reranker: reranking is a stage
the manager runs on top (`episodic_memory_manager.md`). No operation
names a key: the two handles are stateless bindings of the stores to
the tenant's key (`server_redesign.md`, "Vocabulary"), so past
construction nothing can route to another tenant, and a handle to a
deleted tenant raises `KeyNotLiveError` from the store's fence.
`format_options` is a constructor parameter, not a call argument,
because it decides what is written: the deriver formats a segment's
timestamp and author into the text it embeds, and a tenant's
derivatives must be formatted one way. It is a mutable tenant option
(`episodic_memory.format`), applying to events processed after a
change.

## API

```python
class QueryHit(BaseModel):
    score: float                    # relevance of the seed to the query; cosine similarity from `query`
    seed: Segment                   # the segment the query matched
    neighborhood: Neighborhood      # the segments around it, in the store's order
    def window(self) -> list[Segment]   # before, seed, after

class Neighborhood(BaseModel):
    before: list[Segment]           # in order, ending just before the anchor
    after: list[Segment]            # in order, starting just after it

class EpisodicMemory:
    async def encode(self, events: Iterable[StoredEvent]) -> None
    async def forget(self, event_uuids: Iterable[UUID]) -> None
    async def query(self, query: str, *,
                    limit: int, min_cosine_similarity: float | None,
                    expand_context: int,
                    since: datetime | None, until: datetime | None,
                    session_ids: Iterable[str] | None,
                    source_ids: Iterable[str] | None,
                    block_kinds: Iterable[str] | None,
                    filter: FilterExpr | None) -> list[QueryHit]
    async def expand(self, anchor: UUID, *,
                     before: int, after: int,
                     since: datetime | None, until: datetime | None,
                     session_ids: Iterable[str] | None,
                     source_ids: Iterable[str] | None,
                     block_kinds: Iterable[str] | None,
                     filter: FilterExpr | None) -> Neighborhood
    @staticmethod
    def render(segments: Iterable[Segment], *,
               format_options: FormatOptions) -> str
```

- `encode`: for each event, first `forget` its derived rows (so a
  repeat leaves one copy), then segment, derive, embed; with eviction
  on, decide which new derivatives are not worth keeping and which
  stored ones they displace (below); write segments and the surviving
  derivatives' links to the segment store; upsert the surviving
  derivatives to the vector store with the declared properties (system
  fields under reserved keys, plus the declared user keys); delete the
  displaced derivatives from both stores. Segment and derivative uuids
  are `uuid4`. The caller's batch is fully written when `encode`
  returns; the manager advances the watermark only then.
- `forget`: look up segments by event uuids and derivatives by segment
  uuids; delete vector records; delete segments.
- `query`: one stage, vector search. Embed the query; split `filter`
  into the declared part and the rest (`filters_and_properties.md`).
  `collection.query` with the declared part and the system filters
  (`since`, `until`, `session_ids`, `source_ids`, `block_kinds`) as
  predicates on reserved keys, evaluated during the search. Then
  `get_segments` for the seeds, then `get_segment_neighborhoods` from
  them with `expand_context` split as
  today (`event_memory.py:450`), the same system filters, and the
  undeclared part as `property_filter`, which bounds the window rows
  and is the post-filter for the seeds: a seed the store does not
  return is dropped. When seeds are dropped the vector `limit` is
  widened, up to `filter.max_overfetch_factor`, and at the cap the search
  returns what survived. Returns at most `limit` hits in descending
  score, one per matched derivative, each carrying its window and the
  index of the matched segment in it; windows of different hits may
  overlap, and each hit is returned whole. Every count is a maximum:
  a filtered search returns fewer when the filter admits fewer.
- `expand`: the neighborhood of an anchor in its session's one total order
  (`segment_store.md`), as claude-memory's `memory_expand` walks a conversation
  around a memory. The anchor is a segment uuid (from a hit) or an event uuid
  (its first segment). `before` and `after` count segments, the one unit the
  store has; a long event is several segments and is read inward by expanding
  from one of them. The walk stays in the anchor's session, and the same
  filters that bound a search bound it: `since` and `until` on the timestamp,
  `source_ids`, `block_kinds` and `filter`. Returns the two sides in the
  store's order and never the anchor: the caller named it and holds it, from
  the hit or the event, the filters apply to the neighbors only, and the
  anchor's place is between the lists. A caller walks further by calling again
  with the first of `before` or the last of `after` as the anchor and one side
  zero. Backed by `SegmentPartition.get_segment_neighborhoods` over the ordering
  index; no vector search and no embedding, so it is one indexed read.
- `render`: the reader's text for a run of segments, in their order:
  each segment's timestamp formatted by `format_options`, its context
  parts' contributions, and its block's rendering (`context.md`,
  `blocks.md`). What the API returns as `text`, and what a reranker
  scores.

## Eviction

From `agentic_expansion` (commit ed2c5702), where it runs in production
over agent transcripts. Why: an agent's stream repeats itself, the same
tool output, the same re-sent context, the same boilerplate, and every
repetition is another derivative with nearly the same vector. Left
alone they grow in proportion to the corpus, crowd a search's hits
with copies of one thing, and add nothing a reader did not already
have. Eviction is deduplication done lazily, at the moment a cluster
of near-duplicates gets too large, on derived data only.

What it does, per batch of derivatives in `encode`:

- Batch predecessors: for each derivative, the earlier derivatives in
  the same batch whose cosine similarity to it is at or above
  `cosine_similarity_threshold`. Only earlier ones count, so a batch evicts
  exactly what serial ingestion of the same events would have.
- Stored neighbors: one vector query per derivative against the
  tenant's collection, all sessions, `search_limit` results at or above
  the threshold.
- The cluster of a derivative is its stored neighbors not already
  displaced in this batch, its batch predecessors not already skipped,
  and itself. A cluster within `target_size` changes nothing. A larger
  one is trimmed from the temporal middle: the earliest half of
  `target_size` and the latest half are kept, ordered by the
  derivative's event timestamp, and the rest go: a stored member is
  displaced, a batch member is skipped.
- Displaced derivatives are deleted from the vector store and their
  links from the segment store; skipped ones are never written. The
  segment stays either way: it is reconstructed and expanded like any
  other, and is found by search only through its surviving
  derivatives, the same standing as a block kind the deriver does not
  handle.

What it guarantees and what it costs. The event store is untouched:
eviction is lossy for search and lossless for the record, and a
reprocessing into a new tenant starts from the full history. A redo of
a batch after a crash forgets the batch's own derivatives first and
runs eviction again over a store that has already lost what the first
run displaced, so it can displace more and never restores anything.
The cost is one bounded vector query per new derivative, which
`eviction: null` removes entirely. The threshold is a property of the
embedder, since two models put the same pair of texts at different
cosine similarities, so a template sets it beside the embedder it chooses and
a deployment calibrates it per embedder; the design gives no number.

## Context

Specified in `context.md`: a mapping from part kind to one registered
part, never `None`, composed by `with_part`, read by `get_part`, never
filtered. The deriver reads `Author` to format text, and the temporal
scorer reads `TimeRanges`.

## Blocks

Specified in `blocks.md`: a registered family of kinds, `text` built
in; the segmenter and the deriver are tables from kind to handler, a
later handler replacing an earlier one for the kinds it names, and a
kind with no handler passes through as one segment with no
derivatives; a segment is one block, so its kind is a system field
filtered by `block_kinds`; rendering calls `block.render`.

## Changes required

- Rename `EventMemory` to `EpisodicMemory`; `EventMemoryParams`
  (`event_memory.py:52`) to constructor parameters.
- `segment_store_partition` and `vector_store_collection` (`:76`, `:80`)
  stay as dependencies and become the stateless handles
  `SegmentPartition` and `VectorCollection`; no operation takes a key.
- `reranker` leaves the constructor (`:96`) and the class: reranking is
  the manager's stage.
- `encode_events` (`:200`) becomes `encode`, idempotent per event by
  forgetting first, taking `StoredEvent`s so a segment can carry the
  event's position; `forget_events` (`:680`) becomes `forget`.
- `query` (`:353`): `vector_search_limit` becomes `limit` with
  maximum semantics, the threshold becomes `min_cosine_similarity`; `since`, `until`, `session_ids`, `source_ids` and
  `block_kinds` are added as typed parameters (`until` exclusive, the
  reference branch's `before` renamed so that `before` counts segments
  everywhere); the reserved-key mapping
  `_to_vector_record_property` (`:340`) and the `m.` user prefix go,
  replaced by `filters_and_properties.md`'s reserved namespace; the plan
  split is added; the result is `list[QueryHit]`.
- `_SEGMENT_UUID_FIELD_NAME` and `_TIMESTAMP_FIELD_NAME` (`:111`, `:112`)
  become reserved keys built by `reserved_property_key`;
  `expected_vector_store_collection_schema` (`:118`) goes, since the
  store's schema is settings.
- Ingest order is unchanged (segments, then vectors).
- `ProducerContext` and `NullContext` (`data_types.py:49`, `:56`) and
  the `Context` discriminated union (`:64`) go; `Event.source_id: str |
  None` is added beside `timestamp` as the filterable identity, and
  `Event.context` becomes the keyed mapping of registered parts above,
  with `Author` and `TimeRanges` as the first kinds. `Segment` and
  `Derivative` carry both. Rendering prints the recorded name; a caller
  that wants current names or ids shown renders from the returned
  `source_id` and context itself. `FormatOptions` stays dates, times,
  locale and timezone. `produced_for` and the producer roles of the old
  episode model are not carried over and nothing replaces them.
- `expand` is added, with `get_segment_neighborhoods` on the segment store, on
  the rule of MemMachine #1498 and `agentic_expansion` commit 0c19942a:
  the neighbors, never the anchor; `string_from_segment_context` and
  `string_from_segment_contexts` become `render`.
- Eviction comes from `agentic_expansion` (commit ed2c5702):
  `_compute_batch_predecessors` and `_select_eviction_targets` as they
  are, cosine only; the three parameters become `EvictionOptions`; the
  displaced derivatives' link rows are deleted as well as their vector
  records, which the branch left dangling.
- Scores are cosine similarity; `SimilarityMetric` goes from the
  embedder, the vector store and the engines, as on the reference
  branch (commit 6ab12098): the embedder exposes `model_id` and
  `dimensions` only, every container and engine is configured for
  cosine, `higher_is_better` and the metric-dependent scoring branches
  in `event_memory.py` go.
- `Embedder.ingest_embed` and `search_embed` take `list[str]`, not
  `list[Any]` (reference branch, commit ae1d616a); the only inputs are
  derivative texts and the query.
- `TextSegmenter` (the `text` handler under `blocks.md`, "Processing")
  imports the standard-library port of
  `RecursiveCharacterTextSplitter` (`agentic_expansion`, commit 10ed25a6)
  in place of `langchain_text_splitters`.
