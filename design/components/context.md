# Context

New shape for an existing type. A context is the typed, non-filterable
data attached to an event's content and carried to its segments, read
by the steps that process content (segmenters, derivers, scorers) and
by rendering. It replaces the closed `Context` union of
`event_memory/data_types.py` and the ordered `CompositeContext` of
#1436. It exists so that data a step reads has a home that is not a
property, because a property invites an index and the redesign
prohibits creating indexes dynamically (`server_redesign.md`,
"Properties and filtering").

## What is wanted from it

Collected from the discussions, and each answered below:

1. Carry a readable name for the content's source, as it was at the
   event, for rendering.
2. Carry data for processing steps that is not for filtering: the
   temporal signal of #1436, a tool's name, a document's title, a
   thread, a language, an "in reply to".
3. Let a library user add a kind, with its own processing, without
   editing anything in the core.
4. Compose: a step adds to what an event carried, and no step depends
   on the order parts were added in.
5. Represent "no context" and "a source with no good name" without a
   sentinel class and without `None` checks.
6. Never be filtered on; identity is `source_id`, a system field.
7. Round-trip through the codec, survive a kind the reader does not
   know, and evolve.
8. Be validated at the API with a generated schema.

## Types

```python
class ContextPart(BaseModel, ABC):
    kind: ClassVar[str]                      # unique, [a-z0-9_], bounded

class Author(ContextPart):                   # kind = "author"
    name: str

class TimeRanges(ContextPart):               # kind = "time_ranges"
    time_ranges: list[TimeRange]

class UnknownPart(ContextPart):              # never registered; see below
    kind_name: str
    data: dict[str, JsonValue]

type Context = Mapping[str, ContextPart]
    # part kind -> the one part of that kind

def get_part[P: ContextPart](context: Context, part: type[P]) -> P | None
def with_part(context: Context, part: ContextPart) -> Context
    # replaces the part of that kind
def without_part(context: Context, part: type[ContextPart]) -> Context
```

`Event.context: Context`, `Segment.context: Context`,
`Derivative.context: Context`. Never `None`: no context is the empty
mapping, which is what a caller omitting the field gets, what a
segment of such an event carries, and what every accessor handles
without a branch. That answers want 5 and the question "what goes
there without an author part": nothing, and `get_part(context, Author)`
is `None`. A `NullContext` class would be a second spelling of the
same absence that every reader would have to test for, and a nullable
field would be a third; the empty mapping is the one.

At most one part per kind; a kind that has several values holds a
list, as `TimeRanges` does. A part's fields are its own and validate
as a Pydantic model; a part carries no identifier of the source, since
that is `source_id` on the event.

## Registration

A kind is a `ContextPart` subclass registered under its `kind` in a
table, like a store kind: by import for the built-ins, through the
`memmachine.context_parts` entry-point group for a library user's own.
The table is the only registration point, and the API's `context`
schema is generated from it as an object whose keys are the registered
kinds and whose values are the corresponding models. A library user
therefore adds a kind by writing the class, registering it, and reading
it in their own segmenter, deriver or scorer with `get_part`; nothing
in the core changes (want 3).

## Composition

`with_part` returns a context with the part set under its kind,
replacing any part of that kind; `without_part` removes one. A
segmenter that extracts time ranges does `with_part(event.context,
TimeRanges(...))` for each segment, and what the event carried stays.
Because a context is keyed, there is no order to agree on and no
nesting to search: #1436's `CompositeContext` was an ordered list whose
order was load-bearing but expressed nowhere, and whose readers walked
it depth-first (want 4).

## Propagation

- An event's context is set at ingest and immutable, like its
  properties; changing it is a delete and a re-ingest.
- A segment's context is the event's, merged with whatever the
  segmenter adds; the segmenter contract says so, as it does for
  properties and `source_id`.
- A derivative's context is the segment's, merged with whatever the
  deriver adds, and exists at derive time only: it is what the deriver
  formats into the derivative's text before embedding, and it is not
  stored in the vector record, whose properties are the declared
  filterable keys and nothing else.

## Rendering

Rendering assembles a segment's text for a reader from the segment's
own fields and its parts: the timestamp, formatted by `FormatOptions`;
then each part's contribution; then the block's own rendering,
`block.render(options)` (`blocks.md`). A kind
contributes by implementing `render(self, options: FormatOptions) ->
str | None`; `Author` renders its name, `TimeRanges` renders nothing.
Placement is the renderer's, by a fixed order of kinds it knows, so
parts stay unordered; a kind the renderer does not know contributes
nothing. A source with no good name to render has a `source_id` and no
`author` part, and its segments render with the timestamp and the
text (wants 1 and 2). A caller that wants current names, or the source
id shown, renders from the returned `source_id` and context itself;
every hit and expansion returns both as data.

## Processing

A step reads the part it needs by kind and ignores the rest: the
temporal scorer of #1436 reads `TimeRanges` from segments, the deriver
reads `Author` to format text, a library user's step reads its own
kind. No step enumerates parts.

## Filtering

Never. A part is not a property, is not declared to any store, and
cannot appear in a filter; `source_id` and `timestamp` are the
filterable identity and time of an event, and anything else to filter
by is a property (want 6).

## Storage and compatibility

A context is codec-encoded, so it is encrypted wherever the codec
encrypts, as one object `{kind: part_fields, ...}` in the event row's
`context` column and the segment row's. Decoding a kind the running
server does not register (a plugin removed, a newer writer) yields an
`UnknownPart` that keeps the kind name and the data, round-trips
unchanged, renders nothing and is read by no step, with one log line;
nothing is dropped. A registered kind's model evolves additively (new
optional fields, defaults for old rows); a change that is not additive
is a new kind name (want 7). The API bounds a context by
`context.max_bytes` and rejects an unknown kind at ingest with
`InvalidContextError` (want 8), so unknown parts arise only from what
the server itself no longer knows, never from a caller.

## Changes to existing code

- `NullContext`, `ProducerContext` and the `Context` discriminated
  union (`data_types.py:49`, `:56`, `:64`) go; `Event.context`,
  `Segment.context` and `Derivative.context` become the mapping.
- `encode_context` and `decode_context` encode the mapping and produce
  `UnknownPart` for unregistered kinds.
- #1436's `TimeRangesContext` becomes the `TimeRanges` part, its
  `CompositeContext` and `find_contexts` become `with_part` and
  `get_part`, and its temporal segmenter merges instead of nesting.
- `string_from_segment_context` renders from parts; the deriver's
  `_format_with_context` reads `Author` by kind.
