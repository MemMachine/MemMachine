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
    partition: EventMemoryPartition,    # handle: this tenant's segments
    collection: VectorCollection,   # handle: this tenant's records in its embedder's container
    segmenter: Segmenter,           # table from block kind to BlockSegmenter (blocks.md, "Processing")
    deriver: Deriver,               # table from block kind to BlockDeriver
    embedder: Embedder,
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
The format of what is embedded belongs to the deriver, not the memory:
a `BlockDeriver` handler owns its `DateTimeFormat` and `parts`, since it decides the
text it embeds and different kinds or handlers may want different
formats, and the `Deriver` table composes each derivative's text with
them. A tenant's deriver options (`episodic_memory_manager.md`) carry
the format per handler, mutable, applying to events processed after a
change. Rendering for display takes the caller's options per call.

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
                    vector_search_limit: int, min_cosine_similarity: float | None,
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
    def render_segments(segments: Iterable[Segment], *,
               datetime_format: DateTimeFormat,
               parts: Iterable[str] = ("author",),
               ids: Iterable[Literal["session", "segment"]] = ()) -> str
```

- `encode`: segment, derive, embed; with eviction on, decide which new
  derivatives are not worth keeping and which stored ones they displace
  (below); then one `write()` block on the segment partition: add the
  events with their segments and the surviving derivatives' links, and
  upsert the surviving derivatives to the vector store with the declared
  properties (system fields under reserved keys, plus the declared user
  keys) inside the transaction, so the segments commit only once the
  vector store holds their records and a failed upsert rolls them back;
  delete the displaced derivatives from both stores. An event is held
  once: a batch naming a held event is rejected whole, and `replay`
  forgets first. Segment and derivative uuids are `uuid4`. The caller's
  batch is fully written when `encode` returns; the manager advances
  the watermark only then.
- `forget`: look up derivatives by event uuids; delete vector records;
  delete the events, which cascades to segments and links.
- `query`: one stage, vector search. Embed the query; `collection.query`
  with the system filters (`since`, `until`, `session_ids`, `source_ids`,
  `block_kinds`) as predicates on reserved keys, evaluated during the
  search, and nothing else: `filter` is the caller's, over the
  user-defined properties, which the vector store does not index, and
  it never reaches the vector store. Then `get_segments` for the seeds
  with the same system filters and `filter`, which is the post-filter
  for the seeds, a seed the store does not return being dropped; then
  `get_segment_neighborhoods` from the seeds that passed, with
  `expand_context` split as today (`event_memory.py:450`) and the same
  filters on the neighbors. There is no widening: `vector_search_limit`
  bounds the vector stage, and a selective `filter` returns fewer hits
  rather than costing more work, which is what makes a search's cost
  predictable. Returns at most `vector_search_limit` hits in descending
  score, one per matched derivative, each its seed with the neighborhood
  around it; neighborhoods of different hits may overlap, and each hit
  is returned whole. Every count is a maximum: a filtered search returns
  fewer when the filter admits fewer. The split between system filters
  and `filter` is the API's, at the boundary where a request is read:
  the memory never splits a tree.
- `expand`: the neighborhood of an anchor in its session's one total order
  (`event_memory_store.md`), as claude-memory's `memory_expand` walks a conversation
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
  zero. Backed by `EventMemoryPartition.get_segment_neighborhoods` over the ordering
  index; no vector search and no embedding, so it is one indexed read.
- `render_segments`: the reader's text for any segments, a segment given twice
  rendered once: a block per session, the sessions in the order of
  their latest timestamps with the latest last and a blank line
  between; each block in the store's order, one line per run of
  adjacent pieces of one event: the timestamp written per
  `datetime_format`, the context parts' contributions, and the block
  renderings (`context.md`, `blocks.md`). Nothing in the text says
  whether two lines are adjacent in the store: a filtered walk can omit
  a neighbor and no segment carries a position, so no rendering can
  promise contiguity and this one claims none. What the API returns as
  `text`, and what a reranker scores, with `ids` off.
- Rendering for a reader who names things back. `ids` marks, each kind
  independently: `"session"` heads every block with `[session:"<id>"]`,
  the id JSON-quoted since a session id is any string; `"segment"`
  starts every line with `[segment:<hex>]`, or
  `[segments:<first>..<last>]` when the line holds more than one
  segment, a uuid as 32 hex digits. The plural and the range say which
  id opens the event and which closes it without prompting; a
  one-segment event carries one id. `session` and `segment` are spelled
  out, since the words cost what the abbreviations cost. Short ids are
  the client's: the server cannot resolve an id abbreviated per
  conversation, so the client translates before calling `expand`
  either way, and abbreviating is substitution on the markers, whose
  grammar is one marker per line,
  `\[segments?:([0-9a-f]{32})(?:\.\.([0-9a-f]{32}))?\]`, and
  `\[session:("(?:[^"\\]|\\.)*")\]` at a block's head. A returned or
  mutable mapping would add a type to the API and a request payload
  that grows with everything the model has seen. Minimal-unique
  prefixes, resolved against the store, are the stateless alternative
  and are not built.

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
  links from the event memory store; skipped ones are never written. The
  segment stays either way: it is reconstructed and expanded like any
  other, and is found by search only through its surviving
  derivatives, the same standing as a block kind the deriver does not
  handle.

What it guarantees and what it costs. The event store is untouched:
eviction is lossy for search and lossless for the record, and a
reprocessing into a new tenant starts from the full history. A redo of
a batch after a crash forgets the batch's events first and
runs eviction again over a store that has already lost what the first
run displaced, so it can displace more and never restores anything.
The cost is one bounded vector query per new derivative, which
`eviction: null` removes entirely. The threshold is a property of the
embedder, since two models put the same pair of texts at different
cosine similarities, so a template sets it beside the embedder it chooses and
a deployment calibrates it per embedder; the design gives no number.

## Context

Specified in `context.md`: a mapping from part kind to one registered
part, never `None`, composed by `with_part`, read by `get`, never
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
- `event_memory_store_partition` and `vector_store_collection` (`:76`, `:80`)
  stay as dependencies and become the stateless handles
  `EventMemoryPartition` and `VectorCollection`; no operation takes a key.
- `reranker` leaves the constructor (`:96`) and the class: reranking is
  the manager's stage.
- `encode_events` (`:200`) becomes `encode`, holding an event once (a
  repeated one is rejected whole; `replay` forgets first), taking
  `StoredEvent`s so a segment can carry the event's position;
  `forget_events` (`:680`) becomes `forget`.
- `query` (`:353`): `vector_search_limit` keeps its name, since it
  bounds the vector stage and the hits can be fewer; the threshold becomes `min_cosine_similarity`; `since`, `until`, `session_ids`, `source_ids` and
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
  `source_id` and context itself. `DateTimeFormat` is dates, times,
  locale and zone, and nothing else; which parts are composed is the
  composer's `parts`. `produced_for` and the producer roles of the old
  episode model are not carried over and nothing replaces them.
- `expand` is added, with `get_segment_neighborhoods` on the event memory store, on
  the rule of MemMachine #1498 and `agentic_expansion` commit 0c19942a:
  the neighbors, never the anchor; `string_from_segment_context` and
  `string_from_segment_contexts` become `render_segments`.
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
