# Blocks

New shape for an existing type. A block is one unit of an event's
content: the leaf the segmenter splits, the deriver embeds, expansion
returns and rendering prints. `Block` in `event_memory/data_types.py`
is a discriminated union closed over `TextBlock` (`block_type:
Literal["text"]`, `data_types.py:36`), so a second kind is an edit to
the core union. This makes block kinds a registered family, like
context parts, and makes the kind a system field of the segment.

## What is wanted from it

1. Content of more than one kind, each processed by its own policy:
   plain text first; structured data, HTML or an image reference
   later. Kinds are processing types, not modalities
   (`data_types.py:26`): plain text, JSON and HTML share a modality and
   are processed differently.
2. A library user adds a kind, with its own processing and rendering,
   without editing anything in the core.
3. The kind is filterable efficiently where a hit is a segment, since a
   segment is exactly one block; never on events, which hold several
   blocks of several kinds.
4. Round-trip through the codec, survive a kind the reader does not
   know, and evolve.
5. Validated at the API with a generated schema.
6. A step's policy for one kind is added or replaced without touching
   its policy for the others: the built-in text policy and a library
   user's policy for their kind compose in one segmenter and one
   deriver, the later registration winning for the kinds it names, so
   there is a base to start from and never a whole segmenter to rewrite.

## Types

```python
class Block(BaseModel, ABC):
    kind: str                                # a Literal on each subclass

    def render(self, options: FormatOptions) -> str | None: ...
        # the reader's text for this block; None renders nothing

class TextBlock(Block):                      # kind = "text"
    kind: Literal["text"] = "text"
    text: str

class UnknownBlock(Block):                   # never registered; see below
    kind: str
    data: dict[str, JsonValue]

Event.blocks: list[Block]                    # one or more
Segment.block: Block                         # exactly one
Derivative.block: Block
```

`kind` is a field, not a class variable as on `ContextPart`, because a
block travels in a list and carries its own discriminator, where a part
is keyed by the mapping it sits in. Kind names are `[a-z0-9_]`, bounded
like a property key, unique.

## Registration

As for context parts: a kind table, filled by import for the built-ins
and through the `memmachine.block_kinds` entry-point group for a
library user's own; the table is the only registration point. The
API's `blocks` schema is the discriminated union over the registered
kinds, built from the table at startup, and the codec decodes with the
same union. A library user adds a kind by writing the class,
registering it, and laying a `BlockSegmenter` and a `BlockDeriver` for
it over the tables ("Processing"), which leaves every other kind's
policy as it was (wants 2 and 6).

## Processing

A step's policy is per kind. `BlockSegmenter` and `BlockDeriver` are
the handler contracts, one kind each; `Segmenter` and `Deriver` are the
two objects a memory holds (`episodic_memory.md`), each a table from
kind name to handler. A table is built from handlers in order, a later
handler replacing an earlier one for its kind, so a library user's
table is the base handlers with their own listed after (want 6). Nothing is
recursive and nothing is chained: a step looks its block's kind up once
and calls one handler.

```python
@dataclass(frozen=True, slots=True)
class Piece:                                # a piece of one block: what one segment holds
    offset: int                             # position among the block's pieces
    block: Block

class BlockSegmenter[B: Block](ABC):
    kind: ClassVar[str]                     # the one kind this handler splits
    async def split(self, event: Event, block: B) -> list[Piece]: ...

class BlockDeriver[B: Block](ABC):
    kind: ClassVar[str]                     # the one kind this handler derives from
    async def derive(self, segment: Segment, block: B) -> list[str]: ...
        # content only; the memory composes the anchor around it

class Segmenter:
    def __init__(self, handlers: Iterable[BlockSegmenter[Any]] = ()): ...   # later wins for its kind
    async def segment(self, event: Event) -> list[Segment]: ...

class Deriver:                              # the same shape over BlockDeriver
    async def derive(self, segment: Segment) -> list[Derivative]: ...

Derivative.text: str                        # what the handler derived; always text
Derivative.block_kind: str                  # the segment's block kind, for the record
```

One kind per handler, typed by the block class. The table dispatched on
the kind, so a handler that took several kinds would only narrow again
what the table already decided, and its "other kind" branch would be
the `NotImplementedError` this design removes; a policy shared by two
kinds is a base class or a function, registered once per kind. The
kind and the class are one-to-one in the kind table, so "per kind" and
"per block class" are the same unit, and the handler's `B` gives it a
typed block.

The table builds the envelope; a handler returns only what varies by
kind. A segment's `uuid`, `event_uuid`, `index`, `timestamp`,
`session_id`, `source_id` and `properties` are the event's by contract
(`context.md`, "Propagation"; `event_store.md`), and a derivative's are
the segment's, so a handler that built them could only get them wrong.
`Piece` is what a segmenter decides: the sub-block and its offset; the
pieces of a block in offset order reconstruct it, the kind's join
contract. A segmenter that adds context parts (a temporal one adding
`TimeRanges`) extends `Piece` when it lands; nothing is built for it
before. A deriver decides texts, each embedded as one
derivative: a derivative is always text, whatever kind it came from, so
every derivative gets the same context processing. The memory, not the
handler, composes the embedded anchor from the timestamp, the parts
`format_options.parts` lists and the text, with the same `_header`
rendering uses (`context.md`, "Rendering"); a handler that formatted
parts itself would be one more place that knows the order. A handler
still sees the whole event or segment, so it reads context with
`get_part`; no handler takes format options.

The built-in handlers, all for `text` and keeping their names:
`TextSegmenter` (the recursive character splitter, `max_chunk_length`),
`WholeTextDeriver` and `SentenceTextDeriver`. `PassthroughSegmenter` is
gone, and so is its configuration name: one segment per block is the
identity segmentation, the complete answer for any kind, and a default
gets no name; `segmenter` omitted in configuration means it. A kind a
table has no handler for has one fixed outcome, and a step never
raises on it: the segmenter emits the block as one segment, unchanged,
so the join contract holds; the deriver derives nothing from it. The
deriver's outcome is a hole a kind's registration should close by
declaring its deriver or declaring none (below); it is not an error at
ingest, which would fail a batch on replayed history whose kind the
server no longer registers. Such a block is
stored, reconstructed, returned by expansion and rendered, and is found
by search only through its event's other blocks. That is the outcome
for a library user's kind until they lay their handlers over the table,
and for a built-in kind under a table with no policy for it.

Base and configuration (proposed; the decision above covers the tables
and the override order, not this). The base tables are the kind
table's defaults: each registered kind names a default segmenter
handler and a default deriver handler, `text` naming `TextSegmenter`
and `WholeTextDeriver`, and the base is those handlers in registration
order, built-ins first. Tenant options (`episodic_memory_manager.md`,
`SegmenterOptions` and `DeriverOptions`) are per kind: an entry names a
registered handler and its options for that kind, and a kind without
an entry keeps its default. Today's `segmenter` option, present or omitted, is the `text` entry
of `segmenter`; `whole_text` / `sentence_text` is the `text` entry of
`deriver`.

## Filtering

The kind is a system field of the segment: `block_kinds` on
episodic-memory search and expansion restricts hits, segment windows,
the selectivity probe and neighborhoods to segments of those kinds
(`episodic_memory.md`), evaluated on the vector record under the
reserved key `memmachine_block_kind` and on the segment row's
`block_kind` column. Event listing has no kind filter: an event has
several blocks, and "an event with a text block" is not a question the
server answers (want 3). The kind meets the criterion of "Properties
and filtering" for a system field because the server dispatches
processing and rendering on it.

## Rendering

Rendering a segment prints the timestamp, the context parts'
contributions, then `block.render(options)` (`context.md`,
"Rendering"); `TextBlock` renders its text. A kind decides its own
rendering, and `None` prints nothing.

## Storage and compatibility

Blocks are codec-encoded, as a list on the event row and as one block
on the segment row and the derivative; the segment row also carries
the kind name as a plain column, `block_kind`, since the encoded block
cannot be filtered. Decoding a kind the running server does not
register yields an `UnknownBlock` that keeps the kind name and the
data, round-trips unchanged, renders nothing and is processed by no
step, with one log line; nothing is dropped. A registered kind's model
evolves additively (new optional fields, defaults for old rows); a
change that is not additive is a new kind name (want 4). The API
rejects an unknown kind at ingest with `InvalidBlockError` and bounds
an event's blocks by `blocks.max_bytes` (want 5), so unknown blocks
arise only from what the server itself no longer knows, never from a
caller.

## Changes to existing code

- `block_type` (`data_types.py:36`) becomes `kind`, the discriminator
  every registered family uses; the closed `Block` union (`:40`)
  becomes the union built from the kind table; `encode_block` and
  `decode_block` go through it and produce `UnknownBlock`.
- `TextBlock` gains `render`; `EpisodicMemory.render` calls it instead
  of reading `.text`.
- `Segmenter` and `Deriver` (`segmenter/segmenter.py`,
  `deriver/deriver.py`) stop being the ABCs a whole policy implements
  and become the kind tables above; `BlockSegmenter` and `BlockDeriver`
  are the ABCs, and `Piece` is the segmenter handler's result.
  `TextSegmenter`, `WholeTextDeriver` and `SentenceTextDeriver` become
  `text` handlers returning pieces and blocks, their envelope building
  and their `NotImplementedError` on another kind gone, since a table
  never routes a handler another kind; `PassthroughSegmenter` goes.
  `service_locator.py` builds a table per configured option. Done in
  #1597.
- `Segment` rows gain `block_kind` (`segment_store.md`); vector records
  gain `memmachine_block_kind` (`vector_store.md`); search and expansion
  gain `block_kinds` (`episodic_memory.md`).
