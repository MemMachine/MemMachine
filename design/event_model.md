# The Event type

An event is the caller-provided unit of ingestion: one entry in a
stream, with the content to remember about it. The memory subsystems
process events and never define them, so the type is designed from what
callers send and from what every subsystem needs, not from any one
subsystem's storage. This document states the design of `Event` and the
types it is built from, `Block` and `Context`, with the reasons for each
choice and the alternatives that were considered, and then maps the
`Episode` type onto it, since the proposal is for `Event` to become the
top-level type of the server. How the episodic memory processes an
event, and the segments and derivatives it derives, come after that,
scoped to that one subsystem, with examples of what the model allows.

## Summary

- An event has an identity, a timestamp, a source, a context, a list of
  blocks, and properties. Its system fields are the ones the server
  gives semantics to; everything else a caller wants to filter by is a
  property, and everything a caller wants read but never filtered is a
  context part or a block.
- Content and attribution are two separate channels with one job each.
  Filtering runs on system fields and properties, which are typed,
  bounded, and declared. Rendering and processing run on context parts
  and blocks, which are typed, codec-encoded, and never filtered.
- Blocks and context parts are registered families keyed by a `kind`
  name. Adding a kind is writing a class and registering it, with its
  own processing and rendering, and nothing in the core changes.
- A stored event is immutable. A change is a forget and a re-encode.
- The event is the unit of ingestion. What the episodic memory derives
  from it for search, expansion, and indexing is that subsystem's
  business, described in its own section at the end.

## Goals

1. Carry content of more than one kind, each with its own processing
   and rendering: plain text first, then a coding agent's tool calls,
   tool results, injected text, and reasoning, and later structured
   data or image references. Kinds are processing types, not
   modalities: plain text, JSON, and HTML share a modality and are
   processed differently.
2. Carry attribution and other data that processing steps and rendering
   read, without making any of it filterable: such data is for the model
   and the reader, may be natural language, and may be sensitive enough
   that a deployment encrypts it, none of which a filter can work on.
   The only indexes are the ones the operator declares in the
   deployment's static configuration at setup time; nothing in a
   request, a tenant, or a project creates one, so data that is read but
   never filtered needs no place in that configuration.
3. Let a library user add a block kind or a context part kind, with its
   own processing, without editing the core.
4. Filter efficiently on the fields the server itself gives meaning to,
   and nothing more, so the system set is closed by a criterion rather
   than by guessing at applications, while the operator declares, in the
   deployment's static configuration at setup time, which user
   properties the stores index.
5. Hold everything an ingested stream contains, so that expansion from
   a hit reaches the whole timeline, while what is embedded is a policy
   that can change without a rebuild.
6. Round-trip through the payload codec, survive a kind the reader does
   not register, and evolve additively.
7. Be bounded by limits every supported backend accepts.

## The model

```python
class Event(BaseModel):
    uuid: UUID
    timestamp: AwareDatetime             # a naive value is rejected
    source_id: str | None = None         # bounded; None is one state
    context: Context = Context()         # at most one part per kind
    blocks: list[Block]                  # in order, each of a registered kind
    properties: dict[str, PropertyValue] = {}
```

Session is a system field of this design, the stream an event belongs
to and the unit the timeline's order is partitioned into, and it lands
as its own change on top of these: the model shown here carries none,
so a partition is one stream. The section on system fields states the
session's semantics so that the field is added to an existing design
rather than designed again.

### Identity

An event's identity is a UUID the caller supplies. UUIDs are unique
across nodes without coordination, so a tenant's rows move between
databases verbatim, and a UUID key keeps index entries narrower than a
composite string key. A caller whose own ids are strings maps them
with a version 5 UUID over its namespace, which is deterministic and
reversible when the original id is kept as a property.

The UUID is also the tie breaker in the timeline's order, so the order
is total with no sequence field: events with equal timestamps are
concurrent, and any deterministic order among them is acceptable. A
caller that wants a particular order among them sorts the ids it
supplies or uses time-ordered UUIDs; a server that mints ids for a batch
mints one per entry, sorts them, and assigns them in input order.

Segment and derivative identities are minted at ingest. Deriving them
from the event id and position was rejected: it makes the store's
uniqueness depend on the caller's convention, where the event row's key
lets the database own the invariant.

### Immutability

A stored event is never edited. An event is encoded once: a batch
naming an event the memory already holds is rejected whole, before
anything is stored, and a change is a forget followed by a re-encode
under the same UUID. Three things depend on the rule:

- A vector record carries a copy of its segment's filterable fields, and
  the vector stage filters on that copy. If a stored event could change,
  the copy could disagree with the segment row, and a search would admit
  or exclude a hit on a value the event no longer has; keeping the two
  equal would need an in-place update of every record on every backend.
  With no edits, the copy written at ingest is exact for the record's
  life, so the vector stage and the event memory store answer a filter
  the same way without reconciliation.
- Replay of a batch is either wholly rejected or wholly stored, so a
  client that retries after a timeout or a crash leaves one copy and
  never a partial second one.
- No path updates a record or a row in place, so the store contracts
  are insert, delete, and read, the operations every supported backend
  implements the same way.

### Timestamp

The timestamp is when the event happened, timezone-aware. It is the
event's time, not the record's: a `created_at` names when an object was
made, which for an event is either the same instant said less clearly or
the moment the row was written, and a memory orders, bounds, expands,
and renders by when things happened, so the time of the write is of no
use to it. A caller importing history supplies the historical time, and
an event with no better time carries the time it was observed. A naive
value is rejected on the model and on the typed bounds `since` and
`until`, since a naive datetime names no instant and a guessed zone
would silently shift an event in the order. The store holds the UTC instant
and the offset the value carried, so rendering in the event's own zone
is possible without a per-call zone.

Timestamp bounds are `since`, inclusive, and `until`, exclusive, so
ranges meet without overlap. `until` rather than `before`, because
`before` counts segments in expansion, and the two sit in one
signature.

### Source

The source is a bounded string the application owns that identifies the
originating entity of the event: a human, an agent, a tool, or an
import alike. It is the filterable identity; it never carries a name,
so identity survives any rename. Null is one state, encoded as a
missing record key, so every backend's "missing" implements it.

The source is a system field because nearly every event has an
originating entity and nearly every application filters by it, so
treating it specially pays: a typed `source_ids` parameter, a column in
the event memory store, and a reserved key every vector store indexes,
where a property gets that only if the operator declares it. The server
gives it no semantics beyond filtering, and an event may lack one,
which is why it is nullable. The readable name is the `author` part.
The alternative, one `author` field holding both an id and a name, was
rejected because
it puts filterable and rendered data in one field, and a name is data
about the event as it happened, not a property of the source: several
sources may share a name, and one source may carry different names over
time.

### Properties

Properties are user-defined values the event can be filtered by: keys
are identifiers within the bound every supported backend accepts;
values are scalars only, because a scalar is what every supported
backend can filter on and some reject nested values outright, with no
lists, no nesting, and no nulls, so absence is the only way a field
holds nothing. Long text is content and
goes in a block; small machine data such as an external id is a
property, which every hit returns and which stays filterable through the
event memory store.

Properties are the one place for a caller's key-value data; there is no
separate unfiltered metadata field. Every property is filterable
through the event memory store, with no index. Which properties the
vector store also indexes is declared once, by the operator, in the
deployment's static configuration at setup time; a request, a tenant,
or a project cannot add to that set, so "filterable" is not a property
of a field and no caller shapes an index. Every reader of such data
reads it as scalar key-values: a filter on a key, a routing rule keyed
by tags, or a round trip to the client with each hit, which is exactly
what properties
provide, so a second field would hold the same data under a second name
with no reader that needs the difference. Nested values have no reader
and no place.

Keys beginning with `memmachine_` belong to the server, and a caller key
under the prefix is rejected at validation, before any segment is
written; the system fields live under such keys, so stores index and
filter them with the same machinery as user properties and no caller
can spoof one. The reasons for that namespace and its shape are under
"The reserved key namespace" below.

### Context

The context is the typed, non-filterable data attached to an event's
content, read by processing steps and by rendering. It is a set of
parts, at most one per kind, built from parts and read by kind:

```python
class ContextPart(BaseModel, ABC):
    kind: ClassVar[str]                  # set once on each subclass, not a field
    def render(self, datetime_format: DateTimeFormat) -> str | None: ...

class Author(ContextPart):
    kind: ClassVar[str] = "author"
    name: str                            # as it was at the event

class UnknownPart(ContextPart):          # produced only by decoding
    kind_name: str
    data: dict[str, JsonValue]

class Context:
    def __init__(self, *parts: ContextPart)       # two parts of one kind are rejected
    def get(self, kind: str) -> ContextPart | None
    def with_part(self, part: ContextPart) -> Context   # replaces the part of its kind
    def encode(self) -> dict[str, JsonValue]            # {kind: fields}
    @classmethod
    def decode(cls, encoded) -> Context
```

The design decisions, each with its reason:

- **A class built from parts, keyed by no one.** A mapping the caller
  keys by hand carries the rule "the key equals the part's kind" in
  prose only, and a caller can write a wrong one. With the class, kinds
  are keys only inside it, a caller never writes one, and two parts of
  one kind are rejected at construction. The class owns its wire form,
  `{kind: fields}`, through a Pydantic core schema, so a model field of
  the type accepts an instance or the encoded form and serializes to
  the encoded form.
- **No context is the empty context, never `None` and never a sentinel
  class.** A source with no good name to render has a `source_id` and no
  `author` part; its segments render with the timestamp and the text. A
  null-context class would be a second spelling of the same absence
  that every reader must test for, and a nullable field a third; the
  empty collection is the one.
- **Composition is a merge.** `with_part` replaces the part of its kind
  and keeps the rest, so a segmenter that extracts data adds a part and
  what the event carried stays. An ordered list of contexts was
  considered, following an earlier composition design, and rejected:
  its order was load-bearing but expressed nowhere, composites nested,
  and its readers walked it depth-first. Keyed parts have no order to
  agree on and no nesting to search.
- **Never filtered.** A part is not a property and is declared to no
  store, for three reasons. A part exists to help the model and the
  reader: it is composed into the text a derivative embeds and a hit
  renders, and it may be natural language, a name or a title or a note,
  which a filter has no comparison for. A part may carry information
  more sensitive than a filter key, and the context is codec-encoded so
  a deployment can encrypt it; a store cannot filter on what it cannot
  read, and making a part filterable would mean holding it in plaintext
  beside the payload. And a filterable field invites an index, which
  only the operator's static configuration may create. Identity is
  `source_id`; time is `timestamp`; anything else to filter by is a
  property. This is the one rule that keeps the two channels apart.
- **Unknown kinds round-trip.** Decoding a kind the running process does
  not register yields an `UnknownPart` that keeps the kind name and the
  data, encodes back unchanged, renders nothing, and is read by no step,
  with one log line. Nothing stored is dropped by a reader that knows
  less than the writer. A registered kind evolves additively; a change
  that is not additive is a new kind name.
- **`UnknownPart` carries `kind_name` as a field and has no `kind`.** A
  registered part's kind is a class variable because one class is one
  kind and the kind is the key the part sits under in the context, never
  data inside the part. `UnknownPart` is one class standing for every
  kind the process does not know, so its kind varies per instance and
  must be a field. The field is not named `kind` because that name means
  "class constant" on every other part: a field under the same name
  would give the name two meanings across one family and leave the class
  attribute absent on the one class where a reader might reach for it.
  One function, `part_kind`, answers the kind for any part, reading the
  class variable or the field, and nothing else asks. The field stays
  out of the wire form: the context writes an unknown part's data under
  its kind name as it does for a registered part, so the kind is never
  stored twice.
- **A part carries no identifier of the source**, since that is the
  event's `source_id`.
- **The first part is `Author`**, the readable name of the content's
  author as it was at the event. The recorded name is what was true and
  what was embedded. Current names, or the id shown beside the name so a
  reader can tell two names are one entity, are the application's to
  render: every hit and expansion returns `source_id` and the context as
  data. A caller-supplied name directory on the format options was
  considered and rejected, since the server would then keep or receive a
  directory it has no other use for.
- **`producer`, `produced_for`, and roles are not carried over as
  fields.** A role is a readable attribution, which is an `author` part
  or another part, or a property where it is filtered by; an addressee
  is a property. The mapping section below covers the existing uses.

Further kinds are the "anything else worth rendering with the content or
read by a step" question, added when a use needs them: a tool's name, a
document's title, a thread, a language, an "in reply to", or the
temporal signal a scorer reads and never renders.

### Blocks

A block is one unit of an event's content: the leaf the segmenter
splits, the deriver embeds, expansion returns, and rendering prints.

```python
class Block(BaseModel, ABC):
    kind: str                            # a Literal on each subclass
    def render(self, datetime_format: DateTimeFormat) -> str | None: ...

class TextBlock(Block):                  # kind = "text"
    text: str
```

- **A kind is content the memory processes.** Every kind is segmented,
  derived, rendered, and reached by expansion under its own policy. A
  kind that exists only to carry data no step reads is not a block kind;
  that data is a property if it is filtered by and a context part if a
  step or rendering reads it.
- **`kind` is the one discriminator name every registered family
  uses.** It is a field on a block, not a class variable as on a context
  part, because a block travels in a list and carries its own
  discriminator, where a part is keyed by the context it sits in. Kind
  names are identifiers bounded like property keys.
- **Every kind renders itself.** `render` returns the reader's text for
  the block, or `None` to print nothing, so rendering never dispatches
  on kinds.
- **Model fields are typed as the abstract `Block`.** A before-validator
  decodes encoded data through the registered union, and the concrete
  kind is what serializes, so a handler's typed block assigns to the
  field without narrowing. Typing the field as the union itself was
  tried and left type errors wherever the codec's or a handler's block
  met the field.
- **The registered union is closed over the built-in kinds** until a
  kind table lands. Decoding an unregistered block kind is rejected,
  where an unregistered context part round-trips; the kind table and an
  `UnknownBlock` that round-trips the same way are open items below.

#### Registered kinds

The kinds below are examples of the mechanism: how a kind declares its
fields, its rendering, and its segmentation and derivation policy. The
design fixes that each kind owns those three things and that a message
and a tool event share one timeline. It does not fix the particular
rendered formats or the particular policies in the table, which are set
by evaluation of retrieval quality and are expected to change without
any change to the model. The tables that apply a kind's policy belong to
the episodic memory and are described in its section at the end.

| kind | fields | rendered | segmented | derived |
| --- | --- | --- | --- | --- |
| `text` | `text` | the text | by the text handler, when configured | by the text handler, when configured |
| `thinking` | `text` | `thinking: <text>` | one segment | nothing |
| `tool_call` | `name`, `input` | `tool_call <name>: <compact JSON>` | one segment | nothing |
| `tool_result` | `name`, `output`, `error` | `tool_result <name> [error]: <output>` | one segment | nothing |
| `injected` | `text`, `source` | `injected <source>: <text>` | one segment | nothing |

The four capture kinds are what a coding agent's transcript produces
beyond messages, each a kind of its own because each can carry a
distinct policy. The policy in the table is the starting point: none is
split, none is embedded, and each renders on one line of the same
timeline a message renders into, so a window mixing messages and tool
events reads as one timeline. Which kinds are embedded, how a tool
result is split, and what each kind's rendered line looks like are
evaluation questions, answered by registering a different handler or a
different `render`, never by changing the model.

- **A tool call carries its name and its input as a JSON object**, the
  shape a tool's arguments have, rather than as a mapping of property
  values: arguments nest, and a block is content the event carries, not
  something the event is filtered by. Serializing a call into a text
  block was rejected because the tool's name is then recoverable only
  by parsing the text, and because a call's rendering and its embedding
  policy differ from a message's. A tool name is bounded like a source
  id rather than like an identifier, since tool names are mixed case.
- **A tool result carries its name, its output, and whether the tool
  failed**; a result says nothing about failure unless it says so.
- **Injected text carries what put it into the conversation**: a hook, a
  skill, a compaction, a reminder, a command, or something else. Same
  modality and same producer as a message, text typed by nobody; the
  distinction is valuable and expressible neither as a context part nor
  as a property without inventing a vocabulary the server does not own.
- **Reasoning is a kind of its own** because capture holds everything the
  session log holds and nothing a model selected. Dropping reasoning on
  the way in would make ingestion a choice about content.
- **A tool result is one segment however long** under the starting
  policy, because the fallback segmentation is one segment per block and
  the kind declares no splitter. Expansion is counted in segments, so a
  client that captures bounds what it sends unless a splitter for the
  kind is registered, in which case the splitter decides.

An alternative kept the block as modality only, with the kind of act
(message, reasoning, call, result, injected) as a property on the event.
It was rejected because the act decides processing: whether the block
is split, whether it is embedded, and how it renders. A property would
leave the deriver and the renderer dispatching on a string the server
does not own, and tool-name filtering, the one thing a property buys,
is a property on the event in either design.

## System fields and properties

A field is a system field for one of two reasons: the server gives it
semantics beyond filtering, or it is expected on nearly every event and
filtered by nearly every application, so that treating it specially, as
a typed parameter and a key every store indexes, pays in filter
performance that a property would get only if the operator declared it.

- `timestamp` orders, bounds, and scores.
- `session_id` is the stream an event belongs to. Nearly every
  application has streams and filters by them, and the server also gives
  the field semantics: it bounds the total order that expansion walks,
  since a tenant holding many interleaved conversations would otherwise
  expand a hit into its neighbors' conversations. Because the order is
  partitioned by it, every event carries one.
- `source_id` is the originating entity. Nearly every event has one and
  nearly every application filters by it; the server gives it no
  semantics of its own, and an event may lack one, so it is nullable.
- A block's `kind` selects its processing and its rendering. It is a
  field of the segment, never of the event: a segment is one block, so
  search and expansion filter by `block_kinds`, and an event, which
  holds several blocks of several kinds, has no kind filter.

The other candidates fail the criterion and each maps onto one of these
or onto a property: a channel is a session; a workspace is a set of
sessions, parallel to the session rather than orthogonal like the
source, so a workspace filter is `session_ids` over its sessions and
membership is the application's; a user is a source, a session, a
workspace, or a tenant by application; importance, language, and their
like are features attached to content, arbitrarily many, so user
properties. The system set is closed by the criterion, and an operator
whose workload filters often on a property declares it indexed in the
deployment's static configuration, which is the answer to "too few" and
"too many" alike.

Three tiers of property, one mechanism underneath: system fields, which
every store indexes; user properties the operator declares indexed in
the deployment's static configuration at setup time, one schema for
every partition of a store; and every other property, filterable through
the event memory store without an index. The first two tiers are what
the vector stage filters on. Nothing moves a property between tiers at
runtime: not a request, not a tenant, not a project.

A vector record carries the reserved keys and the user properties the
deployment's configuration declares indexed, and nothing else; a store
that declares what it indexes and refuses the rest takes every record
the memory writes.

### The reserved key namespace

Underneath, each system field is a property under a reserved key,
`memmachine_<system>_<field>`, stored and filtered by the same machinery
as a user property. The namespace was chosen deliberately, against
several alternatives, for these reasons.

**Three parties share one flat key space.** A record's properties hold
the end user's keys, an untrusted and open set; the embedding
application's keys, the trusted closed set it stamps on every event; and
the memory's own keys. "System" here means everything that is not the
end user, the application that embeds the memory included. A scheme
that can express only one privileged tier leaves the other two sharing
a namespace that stays disjoint by luck: the application picks a name
the memory adds later, and existing deployments break silently.

**A leading underscore was rejected as the reserved prefix.** It is the
natural reservation when one party owns the namespace, which is how
document stores reserve a few exact names, and it fails when several
parties share one: it spends the one obvious prefix on a single memory
system, so a wrapping application or a second memory type has no
sanctioned way to reserve names of its own; it forbids the family of
names callers want for their own private keys; and the store that
reserved individual underscore names, among them a timestamp field it
later removed, taught its users to avoid the whole family while its
tooling treated such fields as second class. Namespaces shared by many
parties are reserved by a qualified prefix instead, the way a database
reserves its catalog prefix or an orchestrator namespaces annotations by
domain.

**The prefix is the distribution name, so its uniqueness is the package
registry's.** A short invented prefix is unique only by assumption, and
a third party embedding the memory beside its own would have to
coordinate prefixes with it. The package name is already unique by
construction; a reader of a payload dump knows at once who owns the
key; and a third party reserves under its own package name, never under
this one. The prefix is reserved as a whole and checked as a prefix, so
a field added later is protected without a list to keep in sync. A
double-underscore spelling for a stronger visual signal was rejected,
since the product name is the signal and leading underscores are what
tooling mistreats.

**The namespace is subdivided by the component that writes the key**, at
the lowest level that can be named, the memory or the store wrapper
rather than a data model such as the event or the block. A reader can
then attribute a key without knowing the codebase, a second memory type
gets its own token with no second caller-facing prefix, and there is no
central list of keys under the prefix: each component names its own,
and components do not share a vector store, so two components' keys
never meet in one schema. The event memory's token is `em`, short
because every key shares the identifier bound the stores agree on,
which the backends with the shortest limits set, and a token as long as
the memory's name would leave some keys no room. The key builder checks
each key against that bound when the module loads, so an overlong key
fails at startup and never at a write.

**Reservation, never rewriting.** The alternative of leaving every name
to the caller and mangling user keys with a prefix on the way to storage
was rejected: a key the caller writes is the key the caller filters on,
a mangling that is undone on every read is a round trip that cancels,
and a mangled user namespace can still be made to address a system key
through the filter language. Two more were rejected: reserving a length
rather than a prefix, which pads every system key to the bound and makes
stored data unreadable; and reserving nothing, by keeping every system
field out of the vector record and filtering time at the event memory
store alone, which costs selectivity at the vector stage on every
time-bounded query.

**Enforcement is validation at the boundary.** A caller key beginning
with the prefix is rejected when the event is validated, before any
segment is written, so no caller can stamp a system field and claim
another source's or session's identity. The system fields are typed
parameters of search that are never spelled inside the user filter, so a
caller cannot reach a system key through the filter either. The
reservation covers property keys only: a caller-named session or source
id may begin with the prefix, since ids are not keys.

**The reservation stands whichever way the API surface goes.** Whether
system fields are typed parameters outside the filter or routed out of a
caller's filter tree was contended when the namespace was fixed, and
the reservation was made first so that both remained available: with
the system keys in a namespace no caller can write, routing a reserved
key from a filter tree to a typed field is one call at a boundary, and
no collision can make either choice unsafe.

Search takes the system fields as typed parameters, `since`, `until`,
`source_ids`, and `block_kinds`, never spelled inside the user filter,
so no caller decides between a bare and a prefixed name. A typed id
list holds ids only: a list left `None` admits everything, and an empty
list admits nothing. The vector stage evaluates the typed predicates
and the declared conjuncts of the property filter; the event memory
store evaluates the whole property filter over the seeds and their
neighbors. The split between system and user fields is the API's, made
above the memory, so the memory never parses a filter tree to find its
own fields.

Every bound is one every supported backend accepts: kind names and
property keys are identifiers, ids and tool names fit the stores' key
columns, and property values are scalars.

## Episode as an event

The proposal is for `Event` to be the top-level type of the server, the
type the public API ingests and returns. Every field of `Episode` has a
home in it, and the uses that depend on each field are covered as
follows.

| Episode field | Event home | Existing uses and how they are met |
| --- | --- | --- |
| `uid: str` | `uuid: UUID` | A caller id that is not a UUID maps through a version 5 UUID over the caller's namespace, with the original kept as a property, so the mapping is deterministic and reversible. The API may mint UUIDs for callers that supply none. |
| `content: str` | `blocks: [TextBlock(text)]` | One text block. Rendering, chunking, and embedding are the text handlers' policy, as they are for episodes. |
| `created_at` | `timestamp` | Renamed for what it is, the time of the event rather than of the object. Ordering, time bounds, and rendering, with the format a caller or a deriver chooses. |
| `producer_id` | `source_id`, and `Author(name)` where the id is the only name available | Filtering by producer is a typed `source_ids` filter. Rendering keeps the `name: text` shape, which is what the declarative deriver's message prefix and the episode rendering print. |
| `producer_role` | a property, and a context part where it is rendered | The short-term filter on role reads the property. Semantic memory's reranker rendering, which prints the role before the content, reads a part, or the `Author` part when the role is the name to show. |
| `produced_for_id` | a property | Read only by the short-term filter. |
| `session_key` | `session_id` once sessions land; the partition until then | In the current API the partition is the session, so the field does work only for callers that put many conversations in one partition. |
| `sequence_num` | nothing; the order is `(timestamp, uuid)` | Removed, as the episode id change ([#1707](https://github.com/MemMachine/MemMachine/pull/1707)) settles it: entries with equal timestamps are concurrent, and the UUID breaks the tie deterministically with no extra field. A caller that supplies ids owns their order among equal timestamps, by sorting them or using time-ordered UUIDs; when the server mints ids for a batch it mints one per entry, sorts them, and assigns them in input order, so a batch stamped with one timestamp keeps its input order. A global counter was rejected: one row every add in the deployment must lock, a commit back on the critical path, and a value that leaks every tenant's write volume to every other. The field's one reader, a progress marker in short-term summarization that nothing reads back, goes with it. |
| `episode_type` | block kinds and context part kinds | Removed. What an episode is, a message or something else, is what its blocks and parts say: a message is a `text` block with an `Author` part, and it renders and embeds with an author header because the part is there, not because an enum says so. Every reader of the enum dispatches on a kind instead. |
| `content_type` | the block's `kind` | Removed. A content type is a block kind with its own processing, which is what the enum reserved room for and never filled. |
| `filterable_metadata` and `metadata` | `properties` | Whether two tiers exist at all is decided by the user-properties changes ([#1670](https://github.com/MemMachine/MemMachine/pull/1670), [#1702](https://github.com/MemMachine/MemMachine/pull/1702)): with them no project or request declares an indexed set, every property is filterable through the event memory store, and the vector store holds only the keys the operator declared in static configuration at setup, so "filterable" stops being a property of a field and the two fields are one. Every reader of either is a scalar key-value reader: the public API accepts string-to-string pairs and documents them as filter keys, the server copies them into the filterable set, the episode store and short-term memory filter on `metadata.<key>`, and semantic memory routes an episode to its sets by tag keys and string values. Reserved keys are rejected at validation instead of by prefix convention. |

So the Event type covers every existing use, with two adjustments that
are the proposal's cost: roles and addressees become properties and
parts rather than named fields, and callers that send string ids send
UUIDs or let the server mint them. The two metadata fields collapse into
one, since every reader of either is a scalar key-value reader. Each
subsystem then consumes one type: the event memory processes blocks by
kind, short-term memory filters on properties and renders from parts,
and semantic memory reads the text blocks, the author part, and the
properties its set types are keyed by.

## Open items

- A kind table for blocks, filled by import for the built-ins and
  through an entry-point group for a library user's, with the API's
  block schema and the codec's union built from it, and an
  `UnknownBlock` that round-trips an unregistered kind the way
  `UnknownPart` does. With the union closed, an unregistered block kind
  is rejected at decode.
- Each registered kind declaring its default segmenter and deriver, so
  a registered kind always has a stated policy and the table's fallback
  applies only to kinds from history, with tenant options per kind.
- The entry-point group for context part kinds.
- Session as a system field, with the session-led order and the walk
  confined to the seed's session.
- Names at render time. The server renders the recorded name; current
  names and ids shown beside names are the application's, from the data
  every hit returns.

## Processing in the episodic memory

Everything above is the general model: the primitives any subsystem
reads. This section is one subsystem's use of them, the episodic
memory that segments events, embeds derivatives, and answers searches
with expansion. Another subsystem reads the same events and processes
them its own way; nothing here constrains it. It is kept because it
shows what the model makes possible, and the examples at the end show
where the episode model fell short.

### Segments and derivatives

```python
class Segment(BaseModel):                # a piece of one of an event's blocks
    uuid: UUID
    event_uuid: UUID
    index: int                           # the block's position among the event's blocks
    offset: int                          # the piece's position among the block's pieces
    timestamp: AwareDatetime             # the event's
    source_id: str | None                # the event's
    context: Context                     # the event's, plus what the segmenter adds
    block: Block                         # exactly one
    properties: dict[str, PropertyValue] # the event's

class Derivative(BaseModel):             # text derived from a segment for embedding
    uuid: UUID
    segment_uuid: UUID
    timestamp: AwareDatetime             # the segment's
    source_id: str | None                # the segment's
    block_kind: str                      # the segment's block's kind
    text: str                            # the text to embed
```

A segment is one piece of one block and carries its event's fields
verbatim, which is a clause of the segmenter contract, because every
filter the event memory store evaluates reads them from the segment.
The timeline's total order is `(timestamp, event uuid, index, offset)`,
within a session once sessions land. Segments are the one unit: a long
event is several of them, read inward by expanding from one, and an
event is never an expansion seed.

A derivative is the text to embed plus the fields a vector record
carries. It has no context and no properties of its own: the deriver
reads the segment's context while composing the text, and the memory
reads the segment's properties when it builds the record, so a copy on
the derivative would be read by nothing.

### Kind-keyed segmenter and deriver tables

A step's policy is per kind. `BlockSegmenter` and `BlockDeriver` are
the handler contracts, one kind each; `Segmenter` and `Deriver` are the
two objects a memory holds, each a table from kind name to handler.

```python
@dataclass(frozen=True, slots=True)
class Piece:                                 # what one segment holds
    offset: int
    block: Block

class BlockSegmenter[B: Block](ABC):
    kind: ClassVar[str]
    async def split(self, event: Event, block: B) -> list[Piece]: ...

class BlockDeriver[B: Block](ABC):
    kind: ClassVar[str]
    def __init__(self, datetime_format: DateTimeFormat = ..., parts: Iterable[str] = ("author",))
    async def derive(self, segment: Segment, block: B) -> list[str]: ...

class Segmenter:                             # table from kind to BlockSegmenter
    def __init__(self, handlers: Iterable[BlockSegmenter[Any]] = ()): ...
    async def segment(self, event: Event) -> list[Segment]: ...

class Deriver:                               # table from kind to BlockDeriver
    def __init__(self, handlers: Iterable[BlockDeriver[Any]] = ()): ...
    async def derive(self, segment: Segment) -> list[Derivative]: ...
```

- **A table, not a chain.** A table is built from handlers in order, a
  later handler replacing an earlier one for its kind, so a library
  user's table is the base handlers with their own listed after, and a
  policy for one kind is added or replaced without touching the others.
  Lookup is one dictionary read and one call. A chain of responsibility
  is recursion over handlers; a visitor puts the kind list in the core;
  a process-global single dispatch has nowhere to hold per-memory
  options such as a chunk length. All three were rejected for the
  table.
- **One kind per handler, typed by the block class.** The table has
  dispatched on the kind, so a handler taking several kinds would narrow
  again what was decided and grow back the "other kind" error branch.
  Kind and block class are one-to-one, so the handler's type parameter
  gives it a typed block. A policy shared by two kinds is a base class
  or a function, registered once per kind.
- **Handlers return pieces and texts, never segments and derivatives.**
  Every field of a segment's envelope is the event's by contract, and
  every field of a derivative's is the segment's, so a handler that
  built them could only get them wrong. A segmenter decides pieces, the
  sub-block and its offset, and the pieces of a block in offset order
  reconstruct it. A deriver decides texts, each embedded as one
  derivative.
- **Unhandled kinds: identity for the segmenter, nothing for the
  deriver, never an error at ingest.** One segment per block is the
  complete answer for any kind, so the segmenter's fallback is correct
  rather than a gap, and a default gets no name: a passthrough handler
  class and a configuration value naming it were rejected, and a
  configuration that omits a segmenter means one segment per block.
  The deriver's fallback stores a block that search cannot reach, which
  a kind's registration should close by declaring its deriver or
  declaring none; raising at ingest was rejected because it would fail
  a batch on replayed history whose kind the server no longer registers,
  and a data-dependent raise fails a batch halfway.
- **A render-based default deriver was rejected.** Embedding each
  block's rendered text would make every renderable kind searchable with
  no registration, but it couples the embedded text to the display text,
  which retrieval work has found worth keeping apart.
- **A derivative is always text.** Whatever kind it came from, a
  derivative is the text to embed plus the segment's fields a record
  carries, so every derivative gets the same context processing, and
  a block kind and a context part kind never meet in a handler.

### Composition and rendering

The text before content, for the embedded text and for a rendered
segment alike, is composed at one point: the timestamp, written per a
`DateTimeFormat`, then the contribution of each context part a `parts`
list names, in that order, then the content.

- **Order is explicit and owned by the composer, not by the parts.**
  Timestamp first and content last are fixed. Between them come the
  parts listed, by kind, in that order; a part not listed contributes
  nothing; parts carry no order of their own. A new kind is placed by
  listing it, and a composition that needs another order lists another.
- **`parts` is a parameter of the composers, not a field of the format.**
  `DateTimeFormat` writes a timestamp and nothing else: date and time
  styles, locale, and zone.
- **A deriver handler owns the composition of what it embeds**, its
  format and its parts, so a kind or a handler embeds under its own. A
  memory-wide ingest format was rejected because different kinds and
  handlers may want different formats, and a per-call ingest format was
  rejected because construction captures every ingest behavior and no
  caller passed one. Display rendering takes the caller's format and
  parts per call, since display is a separate concern.
- **No handler formats parts itself.** The one composition point rules
  out a block-by-part matrix by construction: a part's contribution is
  per part kind, a block's content is per block kind, and there is one
  place they meet.
- **Rendering groups runs.** A header starts each run of adjacent pieces
  of one event, and the pieces' renderings are joined under it; a block
  that renders nothing shows its kind in brackets at the head of its
  run, so a reader sees that something is there.

### Storage

Blocks and contexts are codec-encoded, so they are encrypted wherever
the codec encrypts, and SQL cannot see inside them. Whatever the store
filters on is projected beside the payload: the segment row carries its
block's kind in a column of its own, filled by the store from the block
in the same insert and never accepted as a separate input, so the
column and the payload cannot disagree. Factoring the kind out is the
ordinary discriminator-column pattern, and the model does not change
for it: a segment keeps one block with its kind inside, and only the row
shape knows about the column. An unencoded JSON block column with a path
filter would take the block out of the codec, and a normalized blocks
table would be a join for a one-word tag; both were rejected.

### Results

A query answers hits, each the segment the query matched, its score,
and the neighborhood around it, which is the shape expansion returns:

```python
class Neighborhood(BaseModel):           # the open neighborhood of a seed
    before: list[Segment]
    after: list[Segment]

class QueryHit(BaseModel):
    score: float
    seed: Segment
    neighborhood: Neighborhood
    def window(self) -> list[Segment]    # before, seed, after
```

The seed is never inside its neighborhood: the caller named it and holds
it, the filters apply to the neighbors only, and the two sides come
back as two lists with the seed's place between them, so a neighborhood
is kept even when its seed would fail the filter, and walking further
from a hit composes with expansion.

### Single types against blocks and parts

The comparison that matters is not with the episode model as shipped
but with the episode model done well: an episode type and a content
type per episode, each a closed set with every value an application
needs defined. Against that, an event is a list of blocks of registered
kinds and a context of registered parts, and the two families mix
freely. Each example is a case met while designing, stated for both.

**One entry, several kinds of content.** A coding agent's turn carries
reasoning, a message, and several tool calls, and the next carries the
results. With one type per episode, the turn is either split into one
episode per piece, so the turn stops being the unit the caller sent and
the pieces need an order of their own, or stored under one catch-all
type whose processing has to fit all of them. With blocks, the turn is
one event with one block per piece, in order, under the turn's own id
and time, and each block is processed and rendered by its kind.

**Attribution orthogonal to content.** A message, a document, a tool's
output, and a transcript's reasoning can each have an author, a title,
a thread, or an "in reply to", and each can lack them. With types, a
combination is either its own type, so the type set grows as the product
of content kinds and attribution shapes, or a nullable field on every
type, so every type carries fields most of its values never use. With
parts, attribution is a set of parts on the event and content is a list
of blocks; a part composes with any kind and a kind with any parts, so
the definitions grow as the sum of the two families, and a new part
applies to every existing kind without touching it.

**Data for a step that is not content.** A temporal extraction produces
time ranges a scorer reads and nothing renders; a tool's name is read by
rendering and nothing else. With types, such data is a field on the
types it applies to, or a type of its own that is not content. With
parts, it is a part, set by the step that produces it with `with_part`,
read by kind by the step that consumes it, and ignored by every other
step and by rendering unless it chooses to render.

**Policy per kind times rendering per part.** Each content kind decides
its own splitting, embedding, and rendering; each part decides its own
contribution to a header. With types, every step switches on the type
pair, and a new type or a new attribution means a new case in every
switch. With kinds and parts, a step looks one kind up in a table and
rendering asks each part for its contribution, so a new kind brings its
handler and its `render` and inherits every part's rendering, and a new
part brings its `render` and applies under every kind.

**Extension by a library user.** A closed enum is edited in the core to
gain a value, and every switch over it is revisited. A registered family
is extended by writing a class and registering it, with its handlers laid
over the tables; a kind the running process does not register still
round-trips as data, so a reader older than a writer drops nothing.

**Evolution.** A type's shape is one shape for every episode of that
type, so a change to it is a change to the type. A kind or a part
evolves additively on its own, and a change that is not additive is a
new kind name, with the old kind still decodable beside it.

**Identity and name.** With one producer field on the episode, the id
and the readable name are one string: the filter and the rendering read
the same value, and a renamed author either breaks the filter or renders
under a name that was never true. With `source_id` as the identity and
`Author` as the name, the filter reads the id, rendering reads the name
as it was at the event, and an application that knows current names
renders them from the id, so each channel keeps its one job.

**Filtering on structure.** With types, the type pair becomes the filter
axis for every structural question, act, modality, and attribution at
once, and an episode is filterable only by the whole pair. With the
split, a block's kind is the one structural filter from content, applied
per segment because a segment is one block, attribution is `source_id`,
and everything else is a property, so a question is asked of the channel
that holds its answer.
