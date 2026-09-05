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
registering it, and registering the segmenter or deriver kind that
handles it (want 2).

## Processing

The segmenter and the deriver dispatch on the kind: the built-in
segmenter splits `text` and the built-in deriver embeds it. A kind a
step does not handle has one fixed outcome, and a step never raises on
it: the segmenter emits the block as one segment, unchanged, so the
join contract holds; the deriver derives nothing from it. Such a block
is stored, reconstructed, returned by expansion and rendered, and is
found by search only through its event's other blocks. That is the
outcome for a library user's kind under the built-in steps until they
register their own, and for a built-in kind under a step with no policy
for it.

## Filtering

The kind is a system field of the segment: `block_kinds` on
episodic-memory search and expansion restricts hits, context windows,
the selectivity probe and neighbourhoods to segments of those kinds
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
- `TextBlock` gains `render`; `string_from_segment_context` calls it
  instead of reading `.text`.
- The segmenter and deriver contracts gain the unhandled-kind clause
  above.
- `Segment` rows gain `block_kind` (`segment_store.md`); vector records
  gain `memmachine_block_kind` (`vector_store.md`); search and expansion
  gain `block_kinds` (`episodic_memory.md`).
