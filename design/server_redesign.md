# MemMachine server redesign

Status: proposal, under review in its own PR against `speedkick`. It
replaces the tenant lifecycle draft in this file's history: the lifecycle
cannot be fixed inside the current API, configuration and wiring, so the
document covers the server. Companion:
`design/segment_store_shared_tables.md` (the segment store as shipped in
#1548). Tracking: #1574. Line references are to `speedkick` at 7752e4cb,
paths under `packages/server/src/memmachine_server` unless given in full.
Settings are named, never given numeric defaults: none has been measured.

## Scope

In: the data model (tenants and events), the HTTP API, configuration,
startup and wiring, tenant lifecycle, the event store, the episodic
memory subsystem with its two derived stores, properties and filtering,
and schema management (initial setup and migrations). Breaking changes
to the API and to the store ABCs are accepted.

Out, and not wired into the new server: short-term memory (callers manage
their own near-term context), semantic memory (cost high, benefit
unproven), declarative memory and the graph stores behind it (Neo4j,
NebulaGraph), the retrieval agent. The MCP surface gets one note.

Requirements that shape everything below:

- Horizontal scaling without sharding, at cluster scope: any process on
  any host serves any tenant, no process owns a tenant, nothing is
  coordinated in process memory. Twenty thousand tenants now; millions
  at the next levels of scale.
- Every component declares its concurrency scope, the widest deployment
  boundary within which concurrent instances may manage the same
  resources (`process`, `host`, `cluster`). A deployment declares the
  scope it runs at, and startup refuses any component narrower than
  that. The SQLite stores declare `process` or `host`; within their
  scope they obey every contract, and nothing about scale is asked of
  them.
- DDL that is not tenant-specific runs only in a dedicated setup step
  that serves no requests and cannot race. Tenant-specific DDL is the
  only DDL allowed anywhere else, and it is avoided where it would be
  expensive at cluster scope. No store in this design creates a table
  per tenant.
- Garbage that is never collected is unacceptable. Wasted writes are
  acceptable.
- Rejection of operations on a deleted tenant is structural: a store
  reads its registry row in the same statement as the data operation
  where both are in one SQL database, and after the operation
  everywhere else. No rejection compares clocks, and no store holds a
  lock across remote I/O. A clock is compared in two places, both on
  the database side and neither about whether an effect is valid: to
  release a tombstone after a clean sweep and a retention orders of
  magnitude above any write's timeout, and to take over a job whose
  reconciler has stopped reporting; every remote client is constructed
  with a request timeout, and the composition refuses one without.
- One client per provider per process, shared by every tenant.
- The tenant layer neither routes data operations nor knows the stores
  or the options a component takes.
- Resources are plain classes, wireable by hand, unaware of any
  configuration or injection system. Composition is Python; settings
  are data.
- No component of the current server is carried over unless named under
  "What is reused".

## What is wrong with the current server

Named so the redesign can be checked against it.

- Identity. The scope of every request is `(org_id, project_id)`, joined
  into `session_key = "org/project"` (`server/api_v2/service.py:38`),
  split back by string (`server/api_v2/router.py:251`), and defaulting
  to `universal/universal` when omitted
  (`packages/common/src/memmachine_common/api/spec.py:58`). "Session"
  names a memory scope, not a conversation; the documentation says
  "project"; store keys are a hash of the string
  (`episodic_memory/long_term_memory/service_locator.py:166`).
- Layers. A request passes `MemMachine` (`main/memmachine.py`, 1726
  lines), `EpisodicMemoryManager`, `EpisodicMemory` and `LongTermMemory`
  (a `match` over two backends) before reaching `EventMemory`, with a
  model translation at each layer: `MemoryMessage` to `EpisodeEntry` to
  `Episode` to `Event`.
- Lifecycle. A search or add on an unknown scope creates it
  (`main/memmachine.py:767`, `:826`; #1576). Deletion returns 204 and an
  in-process queue with no persistence or retry does the work
  (`main/memmachine.py:343`; #1575, #1577).
- Vector stores. Each keeps its collection registry inside its own
  backend (a registry collection on Qdrant and Milvus, a table in the
  store's SQLite file), and a collection handle keeps writing after
  `delete_collection`. Both follow from #1201's model: one process owned
  a collection, so a stale handle could only be that process's own, and
  a store depending on one backend only was worth more than a fence
  nobody needed. Under any-process-serves-any-tenant the handle is
  stale across processes and nothing rejects it (#1537, #1563).
- Configuration. Resources are built lazily behind double-checked locks
  (`common/resource_manager/`), `DatabaseManager` repeats the pattern
  seven times, locator functions assemble params (`*/service_locator.py`),
  `ResourceManagerImpl.build()` is never called by the server, a
  misconfigured provider silently disables a subsystem
  (`common/configuration/__init__.py:384`), every YAML key is lowercased
  (`:586`), an opt-in HTTP API mutates the running process
  (`server/api_v2/config_router.py`), and the configuration models are a
  second description of the components, hand-mapped onto their
  constructors.
- Errors. Status mapping is hand-written per handler, two handlers have
  none, an unknown exception serializes its traceback into the response
  (`server/api_v2/exceptions.py:49`), and an unhandled exception drops the
  connection instead of answering.
- Storage. Eight `create_all` sites in serving processes and one Alembic
  chain (#1570). Every ingested message is stored twice, in the
  `episodestore` table and as segments, linked by an `_episode_uid`
  property with no transaction between the two
  (`episodic_memory/long_term_memory/long_term_memory.py:386`).
  Re-ingesting an id duplicates segment rows.
- Filtering. A string query language with its own grammar
  (`common/filter/filter_parser.py`), user keys mangled behind an `m.`
  prefix, system fields homogeneous with user metadata, and a
  per-collection indexed-properties schema (#1573, #1535) through which
  a caller creates filter indexes dynamically, which makes the
  deployment too difficult to maintain and its performance
  unpredictable. That is why `Event` and `Context` are reshaped below:
  a caller must have a place for every kind of data that does not
  create an index.

## Vocabulary

- Tenant: an isolated memory with its own configuration and lifecycle. It
  is not a user, a conversation or an agent run; an application maps
  those onto tenants by naming them. "Tenant" is chosen over "memory",
  "space" and "namespace" because it says isolation and lifecycle and
  nothing about what the application puts inside.
- Tenant name: the application's label. Any string up to 1024 bytes,
  unique among tenants that are not deleting or deleted, renamable.
- Tenant id: a UUID minted at creation, permanent, never reused. The
  tenant row outlives the tenant as a tombstone for as long as anything
  could remain under the id, which is what makes "never reused"
  enforced rather than hoped while it matters.
- Event: what a caller ingests: an id, a timestamp with its offset, an
  optional session id, an optional source id, an optional context, one
  or more content blocks, properties.
  Events are the caller's data; the event store records them and memory
  subsystems process them.
- Block: one unit of an event's content, of a registered kind; `text`
  is the first kind, and a library user registers others. A segment is
  exactly one block, so a block's kind is a system field of the
  segment, filtered by `block_kinds` on episodic-memory search and
  expansion, and of nothing else. Specified in
  `design/components/blocks.md`.
- Session id: the stream an event belongs to, a conversation or an
  interaction the application delimits, a bounded non-empty string it
  owns, required; events with the same session id form one ordered
  stream. Ids beginning with `memmachine_` are reserved for the server:
  `memmachine_default` is the stream of events the legacy API ingests
  without a conversation id, a stop-gap, since the new API requires a
  session.
  A system field beside `timestamp`, indexed, filtered by
  `session_ids`, and the field the server's own semantics depend on:
  expansion and segment windows walk the one total order confined to
  the seed's session, so they never cross from one conversation into
  another that happens to be interleaved in time. `session_ids` names
  sessions only. Inside a tenant "session" means exactly a
  conversation, which is why the tenant is not called one.
- Source id: the stable identifier of the entity responsible for an
  event's content, human, agent, tool or import, a bounded string the
  application owns. A system field beside `timestamp`, indexed and
  filtered by `source_ids`, so filtering is uniform over every event and
  "no source" is one state. Never rendered unless asked for.
- Context: the typed parts attached to an event's content for the
  steps that process it and for rendering, never for filtering. A
  context is a mapping from part kind to one part, each part a Pydantic
  model registered under its kind, so parts compose by merging and any
  step reads the one it needs by kind (`get_part(context, Author)`) without
  an order to agree on. `Event.context` is never `None`: no context is
  the empty mapping; a source with
  no good name to render has a `source_id` and no `author` part, and a
  name is a part's field rather than a property of the source, so
  several sources may share a name and one source may carry different
  names over time. The first kinds: `author` (`name`, the readable name
  as it was at the event) and `time_ranges` (the temporal signal of
  #1436, read by scoring, never rendered). A library user registers a
  kind the way a store kind is registered, and its part is stored,
  round-tripped and handed to that user's segmenter, deriver or scorer
  unchanged. Codec-encoded, stored with the event and copied to its
  segments. This replaces #1436's `CompositeContext`, an ordered list
  of a closed union: keyed parts need no ordering convention, no
  nesting, no depth-first search, and no edit to a core union to add
  one. `producer`, `produced_for` and the roles of the old episode
  model are not carried over, and nothing replaces them.
- Names at render time: the recorded name is what was true when the event
  happened, and what was embedded. Every hit and every expansion returns the
  segment's `source_id` and its context as data, so an application that knows a
  source's current name, or wants the id shown beside it so a reader can tell
  two names are one entity, renders that itself. The server's text rendering
  (`EpisodicMemory.render`, the `text` of a hit) is a convenience that prints
  what was recorded, formatted by `FormatOptions`, which stays what it is:
  dates, times, locale, timezone. The application holds the directory; the
  server keeps none.
- Event store: the component that records a tenant's events, in the
  order they were ingested. The system of record.
- Memory subsystem: a component that processes a tenant's events into
  derived data and serves queries over it. This design has one,
  episodic memory (the current `EventMemory`, renamed: "event" is the
  ingestion type, not the memory). Segment and derivative keep their
  meanings from `event_memory/data_types.py`.
- Component with per-tenant resources: the event store and each memory
  subsystem. Both register with the tenant service the same way.
- Key: a UUID a store takes as the identity of a partition or
  collection. It is the tenant id, in every store.
- Key registry: the one shared implementation of per-key bookkeeping for
  stores whose data is not in a SQL database, injected into each such
  store as a view scoped to that store.
- Handle: a store's data operations bound to one key (and, for the
  vector store, one container), held by a consumer in place of the
  store. Stateless: it holds the key and the store and nothing else, so
  constructing one does no I/O and checks nothing, there is nothing to
  open, close or evict, and nothing in it goes stale. Every operation
  through it is fenced by the store on the registry row as if the key
  had been passed, so a handle to a key deleted after it was built
  raises `KeyNotLiveError` on its next operation, from that fence and
  not from anything it remembers. What it buys is routing that cannot
  be wrong past the point it was built: a consumer holding a handle
  cannot name another key. `EventPartition`, `SegmentPartition` and
  `VectorCollection` are the three, each an ABC a backend implements
  beside its store. The incarnation-bound handles of the current
  stores, opened, closed and stale, are not these.
- Resource: any object the composition builds: a database engine, a
  provider client, a store, a component, the tenant service. A plain
  class or factory with a typed signature.
- Provider: a resource that is a process-wide model or backend client
  shared by every tenant: an embedder, a language model, a reranker, a
  database engine, a vector backend client.
- Template: a named set of component sections in the settings; a
  create request that names one enables those components on the new
  tenant with those sections. Nothing is built from a template.
- Tenant configuration: the resolved options recorded on a tenant's
  component rows, one section per enabled component, applied to the
  component by a job.
- Naming of values: settings (deployment), tenant configuration (per
  tenant, options), templates, overrides, defaults, request parameters,
  job arguments, partition configuration (per key). One word per kind,
  defined in `design/components/README.md`; in identifiers `Settings`
  and `Config`, Python's conventions, and never `conf`, `params`,
  `args` or `payload`.
- Job: a row describing one action for one component on one tenant.
  Four kinds, all defined and scheduled by the tenant service:
  `provision`, `delete` (the unlink, one call) and `sweep` (purging,
  batch by batch) for the tenant's lifecycle, `replay` for a subsystem's
  processing of the tenant's log. A component cannot define a kind.
- Reconciler: the role that claims and executes jobs and runs the
  tombstone pass. A process runs it when configured to; a deployment
  runs as many as it needs. A claim is a state transition on the job
  row, never a lock held across the step.
- Concurrency scope: `process` < `host` < `cluster`, the widest
  deployment boundary within which concurrent instances of a component
  may safely manage the same resources. Computed from a resource's
  constructor arguments at construction; a composition's scope is the
  minimum of its parts'. Declared and validated at startup, never
  enforced at runtime. "Host" is one operating system instance, which
  is what a file lock reaches; "node" is avoided as overloaded.

## Principles

1. Control plane and data plane are separate objects with separate
   endpoints. The tenant service creates, renames, configures and
   deletes tenants and records jobs, and is the only reader of the
   tenant table. Components serve data operations and own their
   per-tenant state, learning of tenants through their hooks.
2. A data operation names its tenant by id in the request, and every
   store operation names its key. Inside the server a consumer of a
   store's data holds a stateless handle, the store's operations bound
   to one key: constructing it does no I/O, nothing is opened or closed
   per tenant, a process holds no per-tenant state, and every operation
   through it is fenced by the store as if the key had been passed.
   There are no handles at the API. A store may keep a private per-key
   cache for a backend that needs one, bounded by the store and
   invisible above it.
3. The tenant service knows a component only through its registration: a
   name, a tenant configuration model to validate against, and the
   hooks behind the four job kinds (`provision`, `delete`, `purge` for
   `sweep`, `replay`). It never sees a store or an option.
4. Stores take a UUID key and nothing else, and fence on that key. Each
   store is two ABCs: the store, the resource and the only place a key
   is named, for lifecycle and for constructing handles; and the
   handle, the data surface, bound to one key at construction, with no
   method taking a key. A data consumer holds the handle only, so it
   cannot reach a lifecycle operation and cannot name a wrong key; a
   wrong key is unrepresentable past the handle's construction rather
   than checked on every call.
5. Every store rejects operations on a deleted key by itself, from a
   registry row keyed by the caller's UUID: in the same statement where
   the data is in the same SQL database, and by a check after the
   operation everywhere else. A crashed operation can leave a record
   under a dead key; purging and the tenant service's tombstone
   pass collect it. No clock is compared, and no store holds a lock
   across remote I/O.
6. Lifecycle steps are idempotent. There is no transaction across
   stores, so a step interrupted midway is completed by repeating it,
   and whatever a failed creation left behind is removed by the same
   path as a deletion.
7. Every lifecycle change is accepted, recorded as jobs, and reported
   on; none is refused for a failure the reconciler will retry. A caller
   polls or waits; it is never told "try again later".
8. Composition is Python: one function constructs every resource by
   calling constructors with the objects they depend on, in evaluation
   order, checked by the type checker. Settings are data: the scalar
   values a deployment sets, validated at startup, never mutated at
   runtime; a bad setting fails startup. A library user calls the same
   constructors.
9. Schema that is not tenant-specific is versioned migrations applied
   by an operator's command that serves nothing and races with nothing;
   serving and reconciler processes verify it and never run it.
   Tenant-specific DDL is the only DDL a job may run, and no store in
   this design runs any.

## Architecture

Every process runs the same binary and builds the same objects from the
settings, in this order:

1. Settings, read from the environment and an optional file, validated.
2. The composition, constructing every resource in evaluation order:
   database engines and backend clients, then embedders, language
   models and rerankers, then the key registry and stores, then the
   event store and each memory subsystem, then the tenant service,
   which takes them as its registrations, then the ingest service.
3. Schema verification: every component's version table is at head, or
   startup fails (see "Schema management").
4. The scope check: the minimum over every resource's concurrency scope
   must reach `server.concurrency_scope`, or startup fails naming the
   narrowest resource.
5. Roles, from `server.roles`: `api` binds the HTTP routers; `reconciler`
   starts the job loop and the tombstone pass. A single-host
   deployment runs both in one process; a cluster runs many `api`
   processes and as many `reconciler` processes as its job volume
   needs, one at least.

Who knows what:

- Tenant service: the tenant tables, the job table, the registrations.
- Event store: its tables, keyed by tenant id.
- Memory subsystem: its derived stores, its own per-tenant table
  (watermark, applied configuration and its version), the providers it
  was given, and the event store as a reader.
- Ingest service: the event store and the tenant service.
- Stores: their backend, their keys, and their own registry rows.
- Routers: the tenant service, the ingest service, the event store, the
  subsystems.

Dependency direction: the registration interface is defined by the
tenant package; component packages import it and nothing else from the
tenant package. The tenant package imports no component. No resource
imports the composition. Every contract is an abstract base class, for
the reasons in `design/components/README.md`; a `Protocol` describes
only an object this code does not define, of which there is none.

Horizontal scaling, at `cluster` scope: all shared state is in the databases
and the vector backend; per-process state is caches any process can rebuild;
concurrent creates are arbitrated by a unique index, concurrent jobs by an
atomic claim on the job row, concurrent data operations and deletes by store
fences. A `host` deployment is several processes on one host sharing SQLite
files; a `process` deployment is one process. The contracts are the same at
every scope; the scope says only which resources may share state with how many
others.

## Tenant registry

Three tables in the tenant database. Every transition is one transaction
on them. Only the tenant service reads or writes them. Specified with
types, constraints and indexes in `design/components/tenant_service.md`.

`tenants`: `id UUID PK`; `name TEXT NULL`, unique index, NULL from the
moment deletion starts so a new tenant can take the name at once under
a new id; `state` in `active`, `deleting`, `deleted`; `created_at`,
`updated_at`, `deleted_at`. A tenant is a name, an id and a state, and
holds nothing by itself. A `deleted` row is the tenant's tombstone: id
and `deleted_at`, nothing else, so no personal data outlives the tenant
in the registry.

`tenant_components`: one row per component enabled on a tenant, keyed
by `(tenant_id, component)`, with its own `state` in `provisioning`,
`active`, `deleting`, `deleted`, the requested section as `config` and
its `config_version`, `deleted_at` and `clean_at`. A component's
resources for a tenant are enabled, configured and disabled on their
own, each with its own jobs; the tenant and its components are as
separate as an account and the services used under it. A `deleted`
component row is that component's tombstone, and the tenant row stays
until every component row under it is gone, which is what keeps the id
reserved while anything could remain under it.

`tenant_jobs`: `id PK`, `tenant_id`, `component`, `action` in
`provision`, `delete`, `sweep`, `replay`, `arguments JSON`, unique on
`(tenant_id, component, action)`, cascading from the component row;
`state` in `pending`, `running`, `done`; `attempts`, `last_outcome`
(`more`, `error`, `clean`, `purged`), `last_error`, `last_run_at`,
`claimed_at`. The row records what happened; when a job is next
eligible is computed at claim time from those columns and the
reconciler's settings, so a change of schedule applies to every pending
job at once and rewrites no row, and an operator's retry is
`attempts = 0`.

Why the tombstones. Minting a tenant id is an insert on the primary key,
so a duplicate id, whether from a collision, a replayed id, or a
registry restored from a backup the stores were not restored to, fails
at the tenant service before any store is touched, for as long as the
tenant row exists. And a component tombstone drives the tombstone pass
under "Tenant lifecycle", which is what collects a record a crashed
operation left under the key after purging had finished. Neither is
kept forever: a component tombstone is removed after a sweep round has
found nothing, a retention of at least a day (`tombstones.retention`,
database clock) has passed since that clean round, and a verification
round after the retention has found nothing again; the tenant row goes
when its last component tombstone does. What the retention assumes is
that no write is in flight longer than it: a stale write is one whose
registry read preceded the delete's commit, and the request it belongs
to is issued at most a process stall after that read and lands within
the client's request timeout plus whatever the backend and the network
can queue, minutes at the outside; every remote client is constructed
with a request timeout, the composition refuses one without, and the
retention exceeds the sum by orders of magnitude. After the prune, a
reused id (a `uuid4` collision, or a registry restored from a backup)
creates cleanly and inherits nothing, since the stores hold nothing
under it. The case this leaves open, a write delayed longer than the
retention, is accepted as less likely than the failures the design does
not defend against either.

Rename is an update of `name`. No store key contains the name, which is
why names can be arbitrary strings while store keys are the 32 hex
characters every backend accepts.

## Tenant lifecycle

Create, `POST /v1/tenants` with `name`, an optional `template` and
per-component `config` overrides:

1. Insert the tenant row as `active`, the event store's component row
   as `provisioning` with its `provision` job, and, when a template is
   named, one component row and `provision` job per section of the
   template overlaid with the overrides, each validated by its
   component's tenant configuration model; one transaction. An unknown
   component name or an invalid option is 422. A duplicate name fails
   on the unique index: 409 `tenant_exists`, and nothing else. A caller
   that wants the existing tenant looks it up by name; an error
   response does not carry another tenant's record, and there is no
   get-or-create flag.
2. Respond 201 with the tenant and its component rows. The `api`
   process then executes this tenant's jobs itself, through the same
   claim as a reconciler, after the transaction has committed, so a
   single-process deployment completes the create inside the request;
   a failed step is left for a reconciler, never surfaced as an error
   to the creator. With `?wait=` the response is delayed until the
   tenant has no `provision` job pending or running, or the wait
   elapses.
3. `GET /v1/tenants/{id}` shows the tenant, each component row's state,
   requested section and version, and each job's attempts and last
   error. A component's section is applied when its `provision` job is
   done.

Enable a component, `PUT /v1/tenants/{id}/components/{name}` with
`config`: the tenant must be `active`; insert the component row as
`provisioning` with its `provision` job and execute it inline as for
create; 201, 202 with `?wait=` while it is still provisioning, 409
`component_exists` if the row exists. The event store is enabled at
creation and cannot be enabled or disabled by request.

Update a component's configuration, the same `PUT` on an existing row:
the row must be `provisioning` or `active`. The component validates the
change against its model, in which every option is mutable or
immutable. An option is immutable exactly when changing it would
require touching existing data (the embedder; anything that reshapes
stored rows); mutable options apply to events processed after the
change or to reads (a reranker, search defaults, segmenter options). An
immutable option in the request is 422, and there is no "expensive but
allowed" class. One transaction writes the section, increments its
version, and inserts or resets the `provision` job with the section and
version in its arguments; the row's state is unchanged. Respond 202;
`?wait=` blocks until the job is done. How it reaches the server
processes: the `provision` step calls the component's hook with the
section in the job, and the hook writes it, with its version, into the
component's own per-tenant row. Every request reads that row, so the
next request on any process uses the new options. No process is
notified; the row is the channel.

Disable a component, `DELETE /v1/tenants/{id}/components/{name}`: the
row must be `provisioning` or `active`; set it `deleting` with
`deleted_at`, insert its `delete` job; 202, `?wait=` blocks until the
row is `deleted`; a repeat while `deleting` responds the same 202.

Delete a tenant, `DELETE /v1/tenants/{id}`:

1. One transaction: `state = deleting`, `name = NULL`, `deleted_at =
   now()`, and every component row that is `provisioning` or `active`
   set `deleting` with a `delete` job. Allowed from `active`; a repeat
   while `deleting` responds the same 202.
2. Respond 202 with the tenant in `deleting`; `?wait=` blocks until it
   is `deleted`. A reconciler in the same process is woken; every other
   reconciler sees the jobs at its next poll.
3. A reconciler executes each `delete` job: one call to the
   component's `delete`, which makes the tenant unreachable in every
   one of the component's stores. Seconds, not minutes; nothing is
   purged yet.
4. The transaction that marks a component's `delete` job done sets its
   row `deleted`, inserts its `sweep` job, removes its `provision` and
   `replay` rows, and, when no component row of the tenant is outside
   `deleted`, sets the tenant `deleted`. `GET` then returns 404; the
   rows stay as tombstones, and the sweeps purge in the background.
5. A `sweep` job's step calls `delete` once more, idempotent and O(1),
   and then `purge`, one bounded batch at a time, until it reports
   nothing remains or the step's time budget
   (`reconciler.step_duration`) is spent; it is done when nothing
   remains. `purge` is idempotent, so a repeated batch is harmless.

Tombstone pass: a duty of the tenant service, run by reconciler
processes every `reconciler.tombstone_interval`, in every one of them
without exclusion, claiming rows so that each is handled by one pass at
a time, bounded per call. The rows record only when things happened;
the retention is a setting, and the comparison is datetime arithmetic
in the claim statement with the database's `now()`, never a stored due
time and never a process's clock. For each `deleted` component row
whose sweep is done, by the round's outcome and the row's `clean_at`:

- No `clean_at` yet: a round that purged something is followed by
  another; a round that found nothing on its first batch stamps
  `clean_at`.
- `clean_at` set, retention not yet elapsed: nothing.
- `clean_at` set, retention elapsed: a verification round runs; if it
  finds nothing the component row and its job rows are removed; if it
  finds something, `clean_at` is cleared and the rounds start over.

A `deleting` tenant whose component rows are all `deleted` becomes
`deleted`, and a `deleted` tenant with no component rows is removed.
This is what collects the one write that can land after purging (see
"How a store fences"); components whose stores cannot hold such a write
have clean rounds from the first. It is the only scheduled duty in the
system.

Rename, `PATCH /v1/tenants/{id}` with `name`: one update; 409 on a
duplicate; 409 `tenant_not_active` unless the tenant is `active`.

States. A tenant is `active -> deleting -> deleted`. A component row is
`provisioning -> active -> deleting -> deleted` and `provisioning ->
deleting`. There is no failed state. A job that raises is rescheduled
with exponential backoff (`reconciler.backoff`, capped by
`reconciler.max_backoff`) and keeps `attempts` and `last_error` on its
row for as long as it takes; a step that completes resets `attempts`.
An operator fixes the cause and the next attempt succeeds. A component
row in `provisioning` or `deleting` past `reconciler.stuck_after` is
logged with its jobs' last errors, and logged again each time the age
doubles.

Reconciler role:

- One loop per process that has the role. It polls every
  `reconciler.poll_interval` and when woken locally, and runs up to
  `reconciler.concurrency` steps at once, each on its own connection.
- Claims are state transitions, not held locks. A claim is one
  statement that sets the oldest eligible `pending` row `running` with
  `claimed_at = now()`; it is atomic on both dialects (`FOR UPDATE SKIP
  LOCKED` in the subselect on PostgreSQL, the statement's own write
  lock on SQLite). Eligibility is computed in the statement from the
  settings: after an error, `backoff` doubled per attempt up to
  `max_backoff`; after a `more` outcome, `sweep_pause` for a `sweep`
  and nothing for a `replay`, which has more log and continues at
  once. The hook then runs with no database lock held, on any row, for
  up to `step_duration`; completion is a second statement that records
  the outcome only if the row is still this claim's.
- Liveness is the one thing time decides: a `running` row whose
  `claimed_at` is older than `reconciler.reclaim_after` is claimable
  again, which is how a crashed or hung reconciler's job is taken over.
  A step that was slow rather than dead then runs twice, and because
  every hook is idempotent and every store fences, the second run
  wastes writes and changes nothing. No clock decides whether an
  effect is valid.
- Execute: the step re-reads the component row and calls nothing when
  the row's state has moved past the step (a `provision` on a row that
  is `deleting`, a `replay` on a row that is not `active`); otherwise
  the hook, with the job's arguments (for `provision`, the section and
  version the job carries; for `delete`, the one unlink call; for
  `sweep`, `delete` then `purge` until `DONE` or the time budget; for
  `replay`, the log). `DONE` marks the job done and resets `attempts`;
  a spent budget records `more`; an exception records `error` and the
  message. The transaction that marks a `provision` job done sets the
  component row `active` and inserts the `replay` row for a memory
  subsystem; the one that marks a `delete` job done sets the row
  `deleted`, inserts the `sweep` job and, when the tenant is `deleting`
  and every component row is `deleted`, sets the tenant `deleted`.
- Reset: the ingest service sets a tenant's `replay` jobs to pending,
  with `attempts` and the last outcome cleared so they are eligible at
  once, through the tenant service's `reset_replay(tenant_id)`, after
  its own transaction commits (in that transaction where the engines
  are shared); a running `replay` step keeps running and the row is
  claimable again when it ends. The reset updates rows that exist and
  inserts none.
- Ordering without locks. A request that disables a component while a
  `provision` step is running cannot wait for it, so the two may
  overlap: the provision hook may create a resource after the `delete`
  hook has unlinked the rest. Three things make that harmless: the
  strict create raises on a `dropping` registry row, so most late
  creates fail at the store; the `sweep` step calls `delete` before it
  purges, so a resource that was created anyway is unlinked and then
  removed; and a later `provision` step re-reads the row, finds
  `deleting`, and marks itself done. `purge` on a live key is still an
  error rather than a case, because the sweep step's own `delete` call
  is what guarantees it never sees one. Every concurrent pair on one
  tenant or component row and its outcome is tabulated in
  `design/components/tenant_service.md`; the data-path pairs (two
  ingests, a failed replay step, a delete racing a replay, a search
  racing an ingest) in `design/components/episodic_memory_manager.md`,
  where the watermark is defined to move only forward and a `replay`
  step to resume from it.
- Cost: one connection per running step, at most `concurrency` per
  reconciler process, and none held across a hook.

## Component contract

A component with per-tenant resources is passed to the tenant service
at construction as a registration:

- `name`: its section in tenant configuration and, for a memory
  subsystem, its path segment in the API (`episodic_memory`,
  `/episodic-memory`).
- `tenant_config`: a Pydantic model for its section, every field
  mutable or immutable, with defaults. Provider references in it are
  ids that the component validates against the providers it was
  constructed with. The tenant service validates a section against the
  model and calls `validate_update(old, new)` on a change, and never
  reads a field.
- Hooks, each idempotent, bounded per call, allowed to raise (the
  reconciler retries):
  - `provision(tenant_id, section) -> None`: create the component's
    resources for the tenant if absent, verify the immutable options if
    present, apply the mutable ones, record the section and its version
    in the component's per-tenant row.
  - `delete(tenant_id) -> None`: make the tenant unreachable in every
    one of the component's stores (each store's logical delete, a row
    flip) and remove the component's per-tenant row. Fast; no
    purging.
  - `purge(tenant_id) -> DONE | MORE`: remove a bounded amount of what
    the component's stores hold under the tenant id; `DONE` when
    nothing is found. Called by a `sweep` job, after that job's own
    `delete` call, for as long as the tombstone exists.
- A memory subsystem additionally implements `replay(tenant_id) ->
  DONE | MORE`, processing a bounded amount of the tenant's log beyond
  its watermark, its only processing path (see "Episodic memory"), and
  `watermark(tenant_id) -> int`, which `?wait=` on ingest polls.

The event store is a component with an empty section: its resources
are its partition, enabled on every tenant at creation and retired
with the tenant.

Data operations:

- Every request reads the component's own per-tenant row (absent: the
  tenant is unknown to this component) and builds the object for that
  tenant from the row and the stores' handles (see "Episodic memory").
  One indexed read; nothing per tenant is opened or held.
- Every store operation is fenced by the store's own registry row, which
  also supplies what the operation needs to address the tenant (the
  codec configuration, the container, the collection UUID). A key that
  is not live raises one error type.
- On an unknown tenant or a not-live-key error, the router asks the
  tenant service for the tenant's and the component row's state and
  answers 404 `tenant_not_found` (no tenant row, or `deleted`), 409
  `tenant_not_active` (`deleting`), 404 `component_not_enabled` (no
  component row, or `deleted`) or 409 `component_not_active`
  (`provisioning` or `deleting`). No component reads the tenant tables.

## Event store

The system of record. Shared tables in a SQL database, keyed by the
tenant id, with the fence under "Store contracts".

- Registry row per key.
- `events`: key, event id, position, timestamp with its offset, session
  id, source id, context and blocks (codec-encoded), properties. Unique
  on `(key, event id)`, which is what makes ingest idempotent per event
  id for a caller that supplies ids; a retry of a request that let the
  server mint them stores the events again.
- A log per key of additions and deletions, each entry a position, a
  kind and the event id. Positions are assigned under the key's
  registry row locked `FOR UPDATE`, so ingests to one tenant serialize
  inside that one short transaction and positions are contiguous and
  commit-ordered within the key. The log is what subsystems replay,
  from one watermark each, so every addition and deletion a client was
  acknowledged for reaches every subsystem at least once without the
  client retrying; processing is only ever a replay of the log by a
  `replay` job, executed inline in the request where a deployment wants
  the latency and by a reconciler otherwise. The log is the
  per-tenant data queue and the job table is the control queue; neither
  is a message broker, because each pairs with a row in one transaction
  that a broker could not join. Schema in
  `design/components/event_store.md`.
- `create_partition(key)` (strict; see "Create is strict"),
  `delete_partition(key)` (logical, O(1), idempotent),
  `purge_partition(key) -> DONE | MORE`, `purge_deleted_partitions()`
  for library users without a tenant service.
- Data operations, on the handle `event_store.partition(key)` and none
  taking a key: `add_events(events) -> IngestResult` (stored ids,
  skipped ids, the head position), `delete_events(ids)`,
  `get_events(ids)`, `list_events(filter, since, until, after, limit)`
  in position order with the last position as the cursor,
  `read_log(after, limit)`, `read_events_after(after, limit)`,
  `head()`.

Ingest, `POST /v1/tenants/{id}/events`, in the ingest service:

1. `add_events` on the event store, one transaction: the events and
   their `added` log entries; ids already present are skipped and
   reported.
2. Reset every memory subsystem's `replay` job for the tenant to
   pending, in the same transaction where the engines are shared and
   after its commit otherwise, so the log is processed by whichever
   reconciler claims it next. Where the process has the reconciler
   role, or `ingest.inline` is set, execute those jobs now through the
   same claim, after the commit, so a single-process deployment
   processes inside the request.
3. Respond 202 with stored ids, skipped ids and the head position;
   `?wait=` blocks until every subsystem's watermark has reached it and
   then responds 200, or 202 with the watermarks so far when the wait
   elapses. The client is acknowledged after the events are durable
   and the replay jobs are pending; processing is observable, never
   assumed.

Delete events, `POST /v1/tenants/{id}/events/delete`: `delete_events`
on the event store, which removes the rows and appends `deleted` log
entries, and resets the `replay` jobs the same way; 202, `?wait=` as
above. Subsystems apply the deletion by replaying the log; nothing
depends on the caller retrying.

## Episodic memory

The one memory subsystem: the current `EventMemory` renamed, processing
events into segments and derivative embeddings and answering searches.

Derived stores:

- Segment store: segments and their derivative links, as shipped in
  #1548.
- Vector store: derivative embeddings, an index.

Both are keyed by the tenant id. There is no rebuild of derived data: a
full reprocessing costs what an ingestion costs, so it is one, into a
new tenant. What the event store gives the subsystem is repair of
partial processing (`replay`) and processing of history for a
subsystem enabled on an existing tenant, whose watermark starts at zero.

Per-tenant row (the subsystem's own table): tenant id, watermark (the
last event position processed), applied configuration and its version.

Identifiers: segment and derivative ids are minted (uuid4). Idempotency
is per event, not per derived row: `encode` for an event first forgets
that event's derived rows, so repeating it after a crash, or from a
`replay`, leaves one copy.

Operations, in the order the stores are touched:

- `encode` and `forget`: for each log entry in order, an `added` entry is
  segmented, derived, embedded and written (segments, then vectors) and
  a `deleted` entry is forgotten; then, after both stores hold the
  batch, the watermark advances past it. A crash mid-batch leaves
  partial derived data above the watermark, which the next replay
  reprocesses (forget first, so one copy). A batch recorded as
  processed is processed in full.
- `replay` (job): the subsystem's only processing path. One job per
  memory subsystem enabled on a tenant, reset to pending by every
  ingest and deletion, claimed by one reconciler at a time, so a
  tenant's subsystems process in parallel while each is a single
  consumer and therefore processes in commit order. It reads the log
  after the watermark, processes the batch as above, and returns `MORE`
  until the log has nothing newer; a subsystem whose watermark is below
  the log's oldest entry, because it was enabled on a tenant with
  history or the log was compacted, reads the events table in position
  order until it reaches the log. There is no separate repair: a
  failure records an error on the job and the next attempt resumes from
  the watermark. Positions exist for this: they make "what has this
  subsystem processed" one integer per tenant, "what is left" a range,
  and the lag a subtraction.
- Search: embed the query; split the filter and choose the plan under
  "Properties and filtering"; vector query (checked against the store's
  registry row after the query, inside the vector store); segment
  contexts. Hits are returned in descending score, each a window with
  the matched segment marked; events are not returned, a caller that
  wants one fetches it by id. Scores are cosine similarity throughout;
  there is no metric option besides cosine similarity. Reranking is a second stage
  the manager runs over the rendered windows, with its own candidate
  count and threshold, so over-fetching is one limit set above
  another.
- Expand: the neighborhood of a segment or event in its session's
  one total order, `before` and `after` counted in segments, the way
  claude-memory walks a conversation around a memory; one indexed read
  on the segment store, no embedding. Segments are the one unit: a
  long event is several of them, read inward by expanding from one.
  The anchor is never returned: the caller named it and holds it, the
  filters apply to the neighbors only, and the two sides come back as
  two lists with the anchor's place between them, so the neighborhood is
  kept even when its anchor would fail the filter and nothing in it can
  be mistaken for the anchor (#1498). Specified in
  `design/components/episodic_memory.md`.
- `forget`: look up segments and derivatives; delete vector records;
  delete segments.

Objects per tenant and request, options per request. `EpisodicMemory`
(the current `EventMemory`) keeps its shape: one segment partition
handle, one vector collection handle, one embedder, one segmenter, one
deriver, the formatting the deriver embeds with, constructed with them,
no operation naming a key. It is a configured object bound to one
tenant, never built by the composition and never cached: a handful of
references, built per request and discarded. The resource is
`EpisodicMemoryManager`, the name repurposed: constructed with the
event store, the segment store, the vector store, and the embedders and
rerankers the composition built, as mappings from id to object. It is
registered with the tenant service, owns the per-tenant table and a
cache of segmenter and deriver objects keyed by their options, and
serves the routes. On a request the manager reads the tenant's applied
configuration from its per-tenant row and builds the object in one
constructor call from `segment_store.partition(key)`,
`vector_store.collection(key, e)`, `embedders[e]`, the row's format,
and the cached segmenter and deriver for the row's options, the
handles being those of composition rule 4, so past that call nothing
can name another tenant, and a dispatch error raises at the store's
fence instead of writing another model's vectors into a container;
then it fills each search option the request left out with the
tenant's default and runs the stages.

That is all the manager does: dispatch, defaults, validation, and the
stages. The vector stage is `EpisodicMemory.query`, which knows one
limit and one threshold, cosine similarity; the reranking stage, when
the request or the tenant asks for one, takes the vector stage's hits,
renders each window, scores the renderings with the named reranker,
applies the reranker's own threshold and cuts to the final limit.
Two stages with one limit and one threshold each are plainer than one
method with two of each, and the manager is the natural place for the
stage that needs the rerankers, which are providers the memory object
never sees. One layer that changes only scope, a tenant id to an object
and ids to objects, is the least a per-tenant configuration can need,
and it is not the layering this document removes, which was four layers
each translating models and branching on backends.

In the settings a deployment declares which embedders and rerankers
exist; the standard composition builds them and hands all of them to
the manager. Products never appear in settings. A deployment that
wants a memory type to offer a subset names the ids in that manager's
settings.

What a request may vary is not bound into objects at all; it is a
request parameter, passed as an argument of the call: `limit`,
`min_cosine_similarity`, `expand_context`, the reranking stage (reranker,
candidates, its threshold) and the rendering format. One model,
`SearchOptions`, is the tenant's `search` section with every field set
and the request's overrides with every field optional, so there is no
second list to keep in step. Nothing that ingest does varies per
request: the format the deriver embeds with is a tenant option, since
a tenant's derivatives must be formatted one way. The division is
composition rule 5's: an option that decides where or how records are
written is structural and bound; what only shapes an answer is a
request parameter.

A tenant naming an id the deployment did not build is rejected at
creation and at `PATCH` by the manager's own validation. Changing the
embedder is not a `PATCH`, because its container holds the data;
changing the segmenter or deriver options applies to later events;
changing the reranker or a search default takes effect on the next
request.

Tenant configuration section `episodic_memory`, with mutability:

- `embedder` (provider id): immutable; a different embedder is a new
  tenant and a new ingestion.
- `segmenter`, `deriver`: per-kind handler options (`blocks.md`,
  "Processing"); `format`: its options; all mutable, applying to
  events processed after the change.
- `eviction` (cosine similarity threshold, neighbors consulted, target
  cluster size, or none): mutable, applying to batches processed after
  the change; the threshold is calibrated per embedder, so a template
  sets it beside its embedder.
- `search`: default `limit`, `min_cosine_similarity`, `expand_context` and
  `rerank` (reranker id, candidates, threshold, or none); mutable; each
  overridable per request, the reranker within the ids the deployment
  offers.

Episodic memory uses no language model today (the segmenter and
deriver handlers are deterministic; the embedder is the only model
call). The section gains a `language_model` id when a deriver needs
one.

Eviction. An agent's stream repeats itself, and every repetition is
another derivative with nearly the same vector; left alone they grow
with the corpus and crowd a search with copies of one thing. With
eviction on, `encode` treats near-duplicates as a cluster: a new
derivative's cluster is its stored neighbors at or above the
cosine similarity threshold, the earlier derivatives of its own batch above
it, and itself; a cluster larger than the target size is trimmed from
the temporal middle, keeping the earliest and the latest, by deleting
stored derivatives and not writing new ones. Segments and events are
untouched, so the loss is to search only and a reprocessing starts
from the full record. Specified in
`design/components/episodic_memory.md`; from `agentic_expansion`,
where it runs in production.

Hooks:

- `provision`: create the partition and the collection under the tenant
  id; an existing `live` row is this component's own earlier attempt
  and is success, a `creating` row is resumed, a `dropping` row is a
  reused key and raises (see "Create is strict"); insert the per-tenant
  row if absent, else verify immutable options and write the section
  and its version, never the watermark.
- `delete`: `delete_partition` on the segment store,
  `delete_collection` on the vector store, remove the per-tenant row.
- `purge`: `purge_partition` and `purge_collection`; `DONE` when
  both report done.
- `replay` and `watermark`: as above; a `replay` step that finds no
  per-tenant row returns `DONE`.

## Properties and filtering

Why `Event` and `Context` are reshaped: to prohibit dynamic index creation.
Today a caller's metadata schema creates filter indexes per collection. That
makes the deployment too difficult to maintain, since a deployment holding many
dynamically created resources with potentially unstable ids becomes an
undiagnosable mess, and an index created from a request has no owner, no
migration and no place in the schema command's chain, so nothing upgrades or
retires it; and it makes performance unpredictable, since a query's cost
depends on which indexes a collection happens to have and on whether the
backend builds one online. The redesign gives every kind of data a place that
is not an index: what the server itself needs indexed is a system field; what a
deployment wants indexed is declared once, per store, at setup; what is read
but never filtered is a context part or a block; and the rest is a property,
filterable through the segment store without an index. Nothing a caller sends
creates an index.

Two tiers of fields, one mechanism underneath. The reference is the
`default` branch of edwinyyyu/MemMachine (commits 27b3279b, 822ccb6b,
2d5dc2b5), adjusted where noted.

System fields. Defined by the server, first-class in the API, typed: for an
event, `id`, `timestamp`, `session_id` and `source_id`; for a segment, its
event's fields and the `kind` of its one block. Search takes them as named
parameters, `since` and `until` (inclusive and exclusive, so ranges meet
without overlap; `until` rather than `before` because `before` counts segments
in expansion), `session_ids`, `source_ids` and `block_kinds` (lists; a source's
rendered name lives in the context and is never filtered). They are never
spelled inside the user filter, so no caller and no model decides between
`timestamp` and some prefixed form of it. Underneath, each system field is
stored as a reserved property key, `memmachine_<system>_<field>`, built by one
function that validates the key against the stores' naming contract at import
time; the prefix is the distribution name, so its uniqueness is the package
registry's. The prefix is reserved as a whole and there is no central list of
keys under it: each service names its own, and services do not share a vector
store, so two services' keys never meet in one schema. Stores therefore index
and filter system fields with the same machinery as user properties, and a
caller key beginning with the prefix is rejected on the way in.

Which fields are system fields is decided by one criterion: the server
gives the field semantics beyond filtering. `timestamp` orders,
bounds and scores; `session_id` bounds the total order that expansion
and segment windows walk; `source_id` is kept for universality and its
tie to the `author` part; a block's `kind` selects its processing and
its rendering, and is a field of the segment, never of the event. The
other candidates fail the criterion, and each maps onto one of these
or onto a property:

- A channel is a session: a stream the application delimits, named
  after the channel.
- A workspace is a set of sessions. It is parallel to the session, not
  orthogonal to it as the source is, so it is not a field of its own: a
  workspace filter is `session_ids` over the workspace's sessions, and
  which sessions a workspace holds, and whether two workspaces may
  share one, is the application's business logic, not the server's.
- A user is a source, a session, a workspace or a tenant, depending on
  the application; the server has no semantics for it.
- A block kind is a field of the segment, since a segment is one
  block, and never of the event, which has several blocks of several
  kinds: episodic-memory search and expansion filter by `block_kinds`;
  event listing does not.
- Importance, language and their like are features attached to content,
  not structure the ingested content has, and there are arbitrarily
  many of them. They are user properties; a deployment that filters on
  one declares it.

Nothing else needs to be a system field: a deployment that filters
often on a property declares it in the vector store's
`indexed_properties` and gets the same during-search filtering, typed,
without a core change, and a property it does not declare is still
filterable through the segment store. That is the answer to "too few"
and "too many" alike: the system set is closed by the criterion, and
the efficient set is open to each deployment.

User properties. `properties` on an event: keys `[a-z0-9_]`, at most 32
bytes (the identifier bound every backend accepts), not reserved;
values scalar only: string
(bounded by `properties.max_string_bytes`), integer, float, boolean,
datetime; no lists, no nesting, no nulls (absence is the only way a
field holds nothing); at most `properties.max_keys` keys per event.
Scalar-only is what every backend accepts (Chroma rejects nested values;
S3 Vectors caps filterable metadata at 2 KB per vector). Long text is
content, not a property: it goes in a block. Small machine data (an
external id, a source URL) is a property, which every hit returns and
which stays filterable through the segment store. There is no separate
opaque metadata field.

Readable metadata, a suggestion. Structured metadata that is meant to
be read rather than filtered could be a block of its own type,
`{"kind": "json", "data": {...}}`, bounded by `blocks.max_bytes`. For:
it rides the content path the event already has, codec-encoded and
stored with the event, reconstructed with it, and given a per-kind
policy by the segmenter and deriver, so it is never filterable, gets no
derivatives by default (a tenant option embeds it as text), and is
returned by context expansion or not as its kind decides. Whether it
occupies a segment is that policy's choice: a kind whose processing
produces no segment stays with the event, is returned with it, and
never enters an expanded context. Against: it is less obvious to a
user than a named field, which the API schema and example would have to
carry; and it invites overuse of a flexible pattern, where every new
need becomes a block kind rather than a designed field. Proposed, not
decided; see "Open questions".

Propagation. Properties are set at ingest and are immutable; changing an
event's properties is a delete and a re-ingest. The event store holds
them. Segments and derivatives carry a verbatim copy of their event's
properties; that is a clause of the segmenter and deriver contracts,
because filtering in the segment store depends on it. Vector records
carry only the declared subset, below.

Declared, not dynamic, indexes. The deployment declares, per vector
store, which user property keys are indexed inside its containers and
with which types (`indexed_properties`), one schema for every container
of that store, created by the schema command (#1573, #1535). Nothing
creates a filter index at runtime, per tenant, or from a request; the
current per-collection `indexed_properties_schema` goes. System fields
are always declared. A user property that is not declared is still
filterable, through the segment store, where a deployment names in its
settings the keys it wants expression indexes on, and the schema
command creates them, without touching the vector store or any
co-tenant.

The vector store rejects every key it has not been declared, on both
paths: a record carrying an undeclared property key is an error on
`upsert`, and a filter naming one is an error on `query`. An undeclared
key therefore does not exist in the vector store, rather than being
stored write-only or scanned for; an unindexed filter is unrepresentable
at the store instead of a silent full scan; and the split between
declared and undeclared predicates is the subsystem's job before the
call, never something a store guesses at. This is the rule the
reference branch arrived at, for the same reasons: a vector store's
properties have one purpose, being filtered on, since queries return
ids and scores and never properties, so an unfilterable property there
is data nothing gives back.

Filter representation. A filter is a tree, constructed, never parsed:
the closed union from the reference, `Equals`, `NotEquals`, `Ordering`
over numbers and datetimes only (ordering strings depends on how a
store encodes them; ordering booleans is meaningless), `In` over a
homogeneous list, `IsMissing`, n-ary `And` and `Or`, `Not`, compiled by
each store with an exhaustive `match`, so a node a store does not handle
is a type error rather than a query-time one. At the API and in MCP a
filter is a JSON object validated by the schema generated from that
union, for example
`{"and": [{"eq": {"field": "category", "value": "note"}},
{"gte": {"field": "score", "value": 3}}]}`. There is no string language,
nothing to learn beyond the schema, and nothing that parses successfully
into a different filter than intended; an MCP tool exposes the schema as
its parameter, which a model fills more reliably than a grammar. A field
in a filter must be a legal user key; system fields have their own
parameters. Semantics: a predicate matches only a record holding a value
of the compared type; `NotEquals` keeps records holding a differing
comparable value, and `Not(Equals)` also keeps records holding none.

Routing. Where a predicate is evaluated depends on what the backend can
do and on how selective the predicate is; the caller never chooses.

- Declared keys, system fields included, are evaluated inside the vector
  search on every backend that filters during the search (Qdrant,
  Milvus, pgvector, Pinecone, S3 Vectors, Weaviate, Chroma,
  sqlite-vec), which is what keeps a filtered search returning enough
  results instead of filtering away the ones it found.
- A store also declares the filter nodes it can evaluate during the
  search (`supported_filter_nodes`). Every backend evaluates equality,
  ordering on numbers and datetimes, and conjunction; some cannot
  evaluate a negation, a disjunction, a membership or a missing-key
  test (Chroma's `where` has `$ne` but no `$not` or `$exists`;
  sqlite-vec's KNN takes comparisons joined by `AND` only). A predicate
  a store cannot
  evaluate is routed exactly like an undeclared key: to the segment
  store, where SQL evaluates the whole language. One language, two
  places of evaluation; nothing diverges between SQL and vector
  filters. The table is in `design/components/filters_and_properties.md`.
- Undeclared keys, and predicates the store cannot evaluate, never
  reach the vector store. Episodic memory splits the filter: the part
  the store evaluates goes to the vector query; the rest is applied
  afterward by the segment store, which holds every segment's
  properties, when it builds the seeds' windows, and a seed it does
  not return is dropped. The vector query over-fetches with bounded
  widening up to `filter.max_overfetch_factor` while seeds are dropped, and
  at the cap the search returns what survived, which can be fewer than
  `limit`. The store scores nothing by id: no candidate set is ever
  assembled outside it.
- A backend that cannot filter during the search (the usearch engine)
  is handed an allowlist by its store, computed the same way over the
  store's own records table.

Limits are maximums. Every count a caller passes is a maximum: `limit`
on search is the most hits returned, and a filtered search may return
fewer. Nothing is called "top k", which promises exactly k. Order is
the operation's: a search returns hits in descending score, and an
expansion returns segments in the store's one total order.

## Store contracts

Every store takes a UUID key and nothing else. The string key contract
(charset, 32 bytes, validators, hashing in `partition_key_for_session`)
is retired, and so is the segment store's incarnation: with keys never
reused, the registry row keyed by the caller's UUID is the whole fence.

### How a store fences

A store keeps one registry row per key: a store whose data is in a SQL
database keeps it in a table beside the data, with a purge queue beside
it; every other store keeps it in the key registry (below), whose row
has three states, `creating`, `live`, `dropping`. A SQL store has two
conditions it can observe, a row (live) and a queue entry (dropping),
and they give it the same outward behavior without a second
component: the row's existence is checked in the same transaction as
the data statement, and the queue entry is written in the same
transaction as the delete, so what remains to purge is a durable,
ordered list and not a best effort. The row holds whatever the store
needs to address the key on its backend, so the fence read is also the
lookup and nothing is opened per tenant.

Stores whose data is in a SQL database (the event store, the segment
store, pgvector, the SQLite stores): the data statement carries the
liveness predicate, `EXISTS (SELECT 1 FROM <registry> WHERE key = ?)`;
a write's transaction pins the row for its duration
and the logical delete takes it exclusively, as shipped in #1548 (`FOR
SHARE` and `FOR UPDATE` on PostgreSQL; on SQLite, where the driver
defers `BEGIN` and a shared engine cannot be put under `BEGIN
IMMEDIATE`, the pin is a self-checking `UPDATE` of the registry row,
which takes the file's write lock for the statement and fails when
the row is no longer live). Those locks are in the same database as
the data, with no remote I/O inside the transaction, so they cost
nothing and give exactness: after the delete commits, no row can be
written under the key. The logical delete removes the row and enqueues
the key in that one transaction; the purge removes the data in batches
and the queue entry last.

Stores whose data is elsewhere (the vector stores on Qdrant, Milvus,
Pinecone, S3 Vectors, Weaviate, Chroma): read the row for the address;
perform the remote operation, a write acknowledged as applied (Qdrant
`wait=True`; Milvus with strong consistency); then read the row again
and raise the not-live error if the key is no longer live. Two indexed
reads per operation, no lock, no transaction spanning the remote call,
and a logical delete that is a row flip that waits for nothing. A
completed operation therefore never reports success on a dead key and
never returns a dead key's records.

What this does not guarantee, and what handles it. An operation that
crashes between its remote write and its check leaves a record under a
key that may be dead. The sweep's `purge` removes it, and if it
landed after that purge's last pass, the tombstone pass does. That is
the one residual, and it is inherent: the registry row and the backend
write are not one atomic action, so any ordering leaves a window in
which the check's outcome and the write's landing are separated by an
in-flight request of unbounded delay. Only the backend can close it, by
checking a fencing token per write, which no vector database offers per
tenant, or by making the tenant a native object whose drop fails later
writes (Weaviate tenants, Chroma collections, SQLite tables, Qdrant
shard keys, the last rejected for scale). Everywhere a metadata database
is paired with a non-transactional store the answer is the same as here:
a soft-delete marker, asynchronous purging, and periodic
reconciliation of orphans.

Why no lock across the remote write. A share lock held across the write
makes the delete wait for in-flight writers, so that for live writers no
write lands after the delete commits and purge's first pass is
complete. It costs a pooled connection held idle across every remote
write, a delete that waits, and settings that couple pool size to write
concurrency and the idle-in-transaction timeout to the longest remote
write. And it does not remove the residual, because a session that dies
with a request in flight releases the lock before the backend applies
the write, so the sweep is needed anyway. The check after the write
reaches the same end state with none of that cost. A compensating
delete, the writer removing what it wrote when its check fails, is not
done either: a crash before it defeats it, so it guarantees nothing the
sweep does not already, and it would be a second code path with its own
failures.

Why no lease, anywhere. A lease is a row saying "holder H owns this
until T". The holder cannot know it still holds it when it acts without
comparing a clock, and between that comparison and the act the lease can
expire and be taken over, so a late effect is prevented only if the
resource rejects a stale holder by a fencing token, which is the
backend cooperation just described as unavailable. The clock compared
must be the database's, never the process's, so every check is a round
trip anyway; acquire, renew and release are row writes; and there is no
queue, so after expiry whoever asks first wins. Where this design needs
exclusion inside one database, the SQL stores' pins, it uses a database
row lock: held in the lock manager's memory, released the instant the
holder's session ends, waiters queued, the database doing the
rejecting. The reconciler's job claim is not exclusion of that kind: it
is a state transition on the job row, with a liveness bound after which
a stalled claim is taken over, and it can afford that because no
effect's validity depends on the claim being exclusive, every hook
being idempotent and every store fencing. The lease once considered for
vector writes was not a mutual-exclusion lease at all but a shared,
time-limited validation with a purge delayed past its expiry; the check
after the write is that validation with the clock removed.

Alternatives rejected for the remote stores: accepting bounded leakage
(garbage that is never collected is unacceptable); delaying purging
past the longest possible write (a clock comparison); per-tenant native
structures on every backend (#1564: they do not reach the tenant
counts required).

Scope declarations, computed from constructor arguments:

| Resource | Scope |
| --- | --- |
| Tenant registry, event store, segment store, key registry | `cluster` on PostgreSQL; `host` on a SQLite file; `process` on in-memory SQLite |
| Vector store | the minimum of its backend's and its key registry's: networked Qdrant, Milvus, Pinecone, S3 Vectors, Weaviate, Chroma are `cluster`; local-mode Qdrant and Milvus are `process`; sqlite-vec is `host` (registry and data in one file, under the file lock); the usearch store is `process` (index state in process memory) |
| Reconciler, ingest service, routers | any; they hold no shared state of their own |

### The key registry

One implementation, `SqlKeyRegistry`: a table `key_registry(scope, key,
state, address, updated_at)` in a SQL database the deployment gives it,
`state` one of `creating`, `live`, `dropping`, `address` an opaque
document. It exists so that the per-key bookkeeping every non-SQL store
needs is written once, and so that those stores share a database
without sharing rows.

A store never receives the registry itself. Its constructor takes a
`KeyRegistry`, which exists only as a view scoped to one scope name:
`create(key, address)` (strict), `get(key)`, `set_state(key, state)`,
`remove(key)`, each constrained to the view's scope in the statement it
issues, so a store structurally cannot read or change another store's
rows. The composition produces the view, by
`registry.scoped("vector-store")`, and the registry refuses a second
claim on a scope at startup. A SQL-backed store does not use it:
its registry table sits beside its data, where the fence can be in the
statement.

### Create is strict

`create_partition(key)` and `create_collection(key, container)` insert
the registry row and raise `KeyExistsError` if any row exists under the
key, in any state. They are not idempotent, on purpose: a store cannot
tell a retry of its caller's own create from a create under a key
someone else chose, and only the caller can, so there is no
open-or-create at the store.

Idempotency lives in the component's `provision`, which knows the key's
provenance: the tenant service minted the id on the tenant table's
primary key before any job ran, so a `live` row under the key can only
be this component's own earlier attempt, and `provision` treats it as
success; a `creating` row is an interrupted attempt of its own, and
`provision` resumes it; a `dropping` row means the key had a previous
life, which the tenant service's tombstone makes impossible for the
server while anything remains under the key and which a library user
reusing keys can cause, and `provision` raises `KeyReusedError` for an
operator.

What every store operation does with a key whose row is present but
not `live` (`creating` and `dropping` in the key registry; a purge-queue
entry in a SQL store), and with no row at all. Only `provision`
proceeds on `creating`, as above:

| Operation | present, not live | no row |
| --- | --- | --- |
| create | `KeyExistsError` | creates |
| write | not-live error; a remote write already sent is garbage until purged | not-live error |
| read | not-live error | not-live error |
| logical delete | returns; idempotent | returns; idempotent |
| purge | proceeds on `dropping` or a queue entry, which goes when nothing remains; raises on `live` | deletes by key in every container; `DONE` when nothing is found |

### Segment store

As shipped in #1548, with these changes:

- Key type `UUID`; the `incarnation` column of every table becomes the
  key, the purge queue is keyed by the key, and the store mints nothing.
  The incarnation existed to keep a new life under a reused string key
  apart from the previous life's rows still awaiting purge; a key that
  is never reused is the life, and the strict create refuses a key whose
  purge is pending, so lives cannot mix. What it would leave behind is
  permanent from the first row written: a second identity per tenant, a
  mint per create, and a mapping consulted on every operation, for a
  capability (in-place replacement) the design rejects.
  `open_or_create_partition` goes.
- The incarnation-bound partition handle, opened and closed, becomes
  `SegmentPartition`, a stateless handle bound to the key at
  construction: the data operations live on it and none takes a key,
  and `partition(key)` on the store builds it without I/O. The
  registry read that fences each operation returns the codec
  configuration; codec objects are cached process-wide by
  configuration, not per key.
- `purge_partition(key) -> DONE | MORE`: purges this key's rows,
  bounded per call, while its queue entry exists; `DONE` when nothing
  remains and the entry is gone; raises on a live row. On SQLite the
  DELETE waits on the write lock up to the driver's busy timeout and
  raises past it; the reconciler retries.
- `purge_deleted_partitions()`: kept for library users without a tenant
  service; the server does not run it.

### Vector store

The collection registry leaves the vector backend and becomes the
store's rows in the key registry, or, for pgvector and the SQLite
stores, a row and a purge queue beside the data; that is the record of
every key that ever carried a record and what makes every record
purgeable on a backend that cannot list or reject keys.

- Native containers are not declared: one per embedder provider the composition
  built, per vector store, named by the embedder id, created by the schema
  command, never by a request. A container's dimensions and metric are the
  embedder's, so dimensions are declared once, on the embedder; its indexed
  properties are the store's, one schema for every container (#1573, #1572);
  backend-specific options per container are an optional settings map. A
  container is retired by the schema command when the configuration no longer
  declares its embedder and no registry row references it; `memmachine schema
  status` shows, per container, the registry rows referencing it by state,
  which is how an operator sees an old embedder's container drain. Until then
  it stays and serves the tenants pinned to it. Inside a container a tenant is
  a value or a native tenant object, per the table below.
- `create_collection(key, container)`: strict (see "Create is strict").
  Where the tenant is a value inside the container, one registry insert
  straight to `live`. Where the tenant is a native object (a Chroma
  collection, a Weaviate tenant), the object is named by the key's 32
  hex characters under the container, the row is inserted as
  `creating`, the object is created, and the row is set `live` with
  its address; a crash between the two leaves `creating`, which
  `provision` resumes by finding or creating the object by that name
  and setting `live`. Telling "already
  exists" from other failures is per backend; on Chroma it is by
  message, since its duplicate-create error is untyped (`InternalError`
  500 locally, `ChromaError` 400 over HTTP, never the
  `UniqueConstraintError` the module exports; chromadb 1.5.9).
- Write, on the handle `vector_store.collection(key, container)`
  (`upsert(records)`, `delete(uuids)`): a record carrying an undeclared
  property key is rejected before anything is sent. Read the row for
  the container and address; perform the remote write, acknowledged as
  applied; read the row again and raise the not-live error if the key
  is not live. No row before the write: the not-live error, and no
  write creates a collection.
- Read, on the handle (`query(vectors, limit, min_cosine_similarity,
  filter)`): read the row for
  the address, query, read the row again, raise the not-live error if
  the key is not live. `filter` names declared keys only and raises on
  any other; it is evaluated during the search where the backend can.
  Queries return record ids and scores, never properties.
- `delete_collection(key)`: set `dropping`. O(1), idempotent, waits for
  nothing.
- `purge_collection(key) -> DONE | MORE`: with a `dropping` row,
  delete records under the key in bounded steps by the backend's means
  in the table below, `MORE` while records remain, and remove the row
  when nothing remains. With no row, delete by key in every container
  the store has, which is how the tenant service's tombstone pass
  reaches a record that landed after the row went; `DONE` when nothing
  is found. Raises on a `live` row; the sweep step's own `delete` call
  precedes every purge. Containers are few (one per embedder), so a
  no-row purge is a bounded number of filter deletes that mostly find
  nothing.

Why nothing escapes. Every record carries a key whose registry row
existed before the record could, because a write reads the row first
and no write creates a row. A completed write is checked after it lands
and raises if the key died meanwhile. A write whose check never ran
lies under a key that is `dropping`, purged by the sweep, or gone,
purged again by a sweep the tombstone pass reset, through the no-row path. No
rejection compares timestamps, and no key is forgotten while a record
under it could exist: the tombstone outlives the last possible stale
write by the retention margin under "Tenant registry".

Backends at the tier that scales to the stated tenant counts. "Addressed
by" is what an operation needs beyond the shared client, all of it held
in the registry row or equal to the key; no backend requires an object
opened per tenant and kept across operations.

| Backend | Tenant inside the container | Addressed by | Rejects a write to a dead tenant | Lists tenants | Purge |
| --- | --- | --- | --- | --- | --- |
| Qdrant | payload value (#1564) | container name; the key as the payload filter | no | no (`facet` is approximate) | filter delete |
| Milvus | partition-key value | container name, loaded once; the key as the partition-key value | no | no | filter delete |
| pgvector | column value | table name; the key as the column value | yes, in-statement (registry in the same database) | yes | keyed delete |
| Pinecone | namespace, a call parameter; created implicitly on first upsert | index host; the key as `namespace` | no | yes | delete all in the namespace, O(1) |
| S3 Vectors | filterable metadata value (10,000 indexes per bucket, so not one per tenant) | bucket and index names; the key as the metadata filter (`$eq`; filters are evaluated during the search) | no | no | no delete by filter: filtered query (top-K up to 10,000), `DeleteVectors` by key (500 per call), repeat |
| Weaviate | native tenant, one shard each, activity tiers | collection name; the key as the tenant name (the client's `with_tenant` wrapper is built per call, no request) | yes (tenant not found) | yes | remove tenant, O(1) |
| Chroma | collection per tenant (Chroma's own write-up warns that metadata filtering "can become slow" as users and documents grow) | the collection's UUID, recorded in the registry row at creation; operations go to the HTTP API by that UUID | yes: operations route by the UUID, and a stale one raises `NotFoundError` instead of reaching a replacement (chromadb 1.5.9) | yes (`list_collections`, paged) | `delete_collection`, O(1) |
| sqlite-vec | partition-key value in one `vec0` table per container | table name; the key as the partition key | yes, in-statement (registry in the same file) | yes | keyed delete |
| usearch store | rows in a shared table; one index file per tenant | table name; the key as the column value; the index file path | yes, in-statement (registry in the same file) | yes | keyed delete; unlink the file |

Pinecone's implicit namespace creation and any backend's inability to
reject are covered the same way: the registry row exists before the
first record, and purge deletes by the tenant's value. Chroma's Python
client only offers a `Collection` object obtained by `get_collection`,
one round trip resolving a name to the collection's UUID, after which
every operation addresses the UUID (`chromadb/api/fastapi.py`). The
store records that UUID in the registry row at creation and calls the
HTTP API by it, so no operation makes the lookup and no object is
held. Where a client library only offers per-tenant objects, the store
uses the backend's HTTP API directly, as here, or builds the object per
call where that is free, as for Weaviate. Verified against chromadb
1.5.9 (comment on #1579): a stale UUID raises `NotFoundError` rather
than writing into a replacement, so the UUID is the fence on this
backend; eight clients racing `create_collection` on one name produce
one winner every round, the constraint living in the persisted sysdb;
metadata values must be scalars, which the only thing stored on a
collection, its container reference, is. The per-collection cost on a
single Chroma node (an index each) and the collection count Chroma
Cloud supports remain unverified and are checked before an
implementation. S3 Vectors' limits are from its documentation as of
this writing: filterable metadata is 2 KB per vector, and timestamps
must be stored as numbers to be range-filtered, since comparisons apply
to numbers only.

The SQLite stores. Both are rewritten away from a table per collection,
like the segment store: the sqlite-vec store keeps one `vec0` table per
container with the tenant key as its partition key, the usearch store
keeps records in one shared table and one index file per tenant (a file
it owns, not schema; index state lives in process memory, which is why
it stays at `process` scope), and both keep their registry in the same
file, so their fence is in-statement and the sqlite-vec store reaches
`host` scope. The O(1) drop that a table per collection gave becomes a
keyed delete run by the purge job, as in the segment store;
unreachability is immediate through the registry either way. A `vec0`
partition key is not a filter over a shared scan: vec0 stores each
partition value's vectors in chunks of its own, and a KNN constrained on
the key reads only those chunks. Measured on the pinned 0.1.9 with 400
tenants of 25 vectors each at 1024 dimensions, the constrained query
cost two orders of magnitude less than an unconstrained one over the
same table. The cost of a partition is the chunk allocation: every
partition value holds at least one chunk of `chunk_size` vectors,
`chunk_size` times dimensions times four bytes, which a table per
collection paid identically (measured: the same size per tenant either
way). `chunk_size` is a per-table setting the store exposes, and a
deployment with many small tenants sets it low.
`sqlite_vec_vector_store.py:4` records that partition keys were avoided
in case a future sqlite-vec ANN index does not support them; if that
comes to pass, the store stays brute-force, which is what it is today
and what its scope is for.

## Composition and settings

Five rules, then the mechanics.

1. Fixed topology, pluggable slots. The standard server is a fixed graph
   of roles: database engines, the key registry, the event store, the
   segment store, the vector store, embedders, rerankers, language
   models, the episodic memory manager, the ingest service, the tenant
   service, the routers. Each role is an ABC. The graph never varies
   between deployments; what fills each slot does. This is the shape of
   every server that supports many backends from one image (a database
   `ENGINE` setting, SQLAlchemy dialects, the OpenTelemetry Collector's
   receivers and exporters, an AI gateway's model list).
2. Composition is Python. One function, `compose(settings) -> Server`,
   constructs every resource by calling constructors and factories with
   the objects they depend on. Wiring order is evaluation order; a wrong
   argument is a type error the type checker reports before anything
   runs; a library user's own class needs nothing but an import. The
   server ships the standard composition, which `memmachine serve`
   runs; `memmachine serve --compose my_package.wiring:compose` runs
   another. A deployment that needs a different graph writes Python,
   which is what a wiring file would have been describing anyway.
3. Settings are data. Everything a deployment sets without changing the
   graph: which kind fills each slot and that kind's own values (URLs,
   credentials, pool sizes, pragmas, model names, dimensions), plus
   bind address, roles, scope, reconciler intervals, property and
   filter bounds, and the tenant templates. Read from environment
   variables and an optional YAML or TOML file of the same shape,
   validated at startup against Pydantic settings models, with a JSON
   Schema and a documented example generated from them. Secrets are
   environment variables. A settings file carries no wiring; it cannot
   express a graph.
4. Scoped views. A dependency that could be misused across a boundary
   is scoped before it is injected, so the misuse is unrepresentable
   rather than checked: the key registry is handed to a store as a view
   over that store's rows only; the segment store and the vector store
   are handed to an episodic memory object as stateless handles bound
   to one key, and to one container. A store cannot touch another
   store's rows, an object built for one tenant cannot name another,
   and an object built for one embedder cannot write into another
   embedder's container. The rule applies wherever a shared resource
   serves several holders.
5. Three scopes, three kinds of object, told apart by identity. A resource is
   built by the composition, has identity and a lifecycle, and holds
   connections or bounds the deployment: stores, the key registry, the
   providers that exist. A configured object is built by a factory from
   resources and a tenant's structural configuration, is interchangeable with
   any other built from the same configuration, is cached by configuration when
   building it costs something and never by tenant, and has no lifecycle: a
   segmenter built from its options, a handle, an `EpisodicMemory` bound to a
   tenant's handles and embedder. A call argument belongs to one request.
   Factories are resources and their products are not; a factory is the
   resource that stands in for a family of configured objects, which is the
   metrics factory's model already. An option is classified by the same line:
   per server if it holds identity or bounds the deployment; per tenant and
   structural if it decides where or how records are written; per tenant and a
   default if it only shapes answers but a caller wants one setting for all its
   calls; per request if it may vary per call. The separation is enforced by
   signature, not by convention: a structural option is a constructor parameter
   of the configured object and a parameter of no operation; a per-request
   option is a parameter of the operation and of no constructor; a tenant
   default is a field of the tenant configuration model whose name is an
   operation parameter, checked at import time, and the manager fills it into
   the call. An option cannot exist in two places, so it cannot silently change
   scope.

Kinds. Each slot family has one table from kind name to callable, a
class or a factory, and the slot's settings type is a discriminated
union over the registered kinds, keyed by `kind`, so validation and
schema generation cover plugins. Built-in kinds register by being
imported; out-of-tree kinds register through `importlib.metadata`
entry points, one group per family (`memmachine.engines`,
`memmachine.vector_stores`, `memmachine.embedders`, ...), which is
Python's standard plugin mechanism and keeps the core ignorant of
backends it did not ship. Heavy client libraries are optional extras
imported inside the kind's factory, so a deployment that never selects
Milvus never imports it; one image carries every built-in kind.

Why not YAML for wiring. A wiring document is Python written in YAML,
with a loader to turn it back: a registry of ids, reference resolution,
topological ordering and runtime type checks, all to avoid a file the
type checker and the interpreter already handle. The usual reasons for a
separate wiring file do not apply here: nothing is reloaded, and
arbitrary code execution is not a new exposure, since the composition
runs as the server's own code at the trust of the image. What YAML is
good at, values templated by Helm or set from a ConfigMap and the
environment, is exactly what settings are.

Why not Python for settings. Values change per environment without a
code change, are templated by deployment tooling, and must be validated
and documented as data. A URL in a Python file is a URL that needs a
code review to change.

Where the standard composition is data-driven: at every slot, through
the kind tables, and in how many providers exist, since embedders,
rerankers and language models are named settings entries and the
composition instantiates one object per entry. The standard composition
decides the graph; settings decide what fills it and how many of each
provider there are.

The standard composition, sketched:

```python
def compose(s: ServerSettings) -> Server:
    main = ENGINES[s.databases.main.kind](s.databases.main)
    metrics = MetricsFactory(s.metrics)
    embedders = {name: EMBEDDERS[e.kind](e) for name, e in s.embedders.items()}
    rerankers = {name: RERANKERS[r.kind](r) for name, r in s.rerankers.items()}
    registry = SqlKeyRegistry(main)
    events = SqlAlchemyEventStore(main, s.event_store)
    segments = SQLAlchemySegmentStore(main, s.segment_store)
    vector_store = VECTOR_STORES[s.vector_store.kind](
        s.vector_store, embedders=embedders,
        registry=registry.scoped("vector-store"), engine=main,
    )   # a kind takes the registry or the engine, whichever holds its rows
    episodic = EpisodicMemoryManager(
        events, segments, vector_store, embedders, rerankers,
        main, s.episodic_memory, metrics,
    )
    tenants = TenantService(
        main, components=[events, episodic], templates=s.tenant_templates,
        settings=s.tenants,
    )
    ingest = IngestService(events, tenants, s.ingest)
    return Server(tenants, ingest, events, episodic, s)
```

The settings a deployment writes, each key a field of a settings model:

```yaml
databases:
  main:
    kind: postgres
    url: ${MEMMACHINE_DATABASE_URL}
vector_store:
  kind: qdrant
  url: http://qdrant:6333
  request_timeout: 10s
  indexed_properties:      # once per store; system fields implicit
    category: string
    score: integer
segment_store:
  property_indexes: [category]   # expression indexes the schema command creates
embedders:
  openai-large:
    kind: openai
    model: text-embedding-3-large
    dimensions: 1024
    api_key: ${OPENAI_API_KEY}
rerankers:
  bm25:
    kind: bm25
tenant_templates:          # data: the sections a create enables at once
  default:
    episodic_memory:
      embedder: openai-large
      search:
        limit: 10
        expand_context: 4
        rerank:
          reranker: bm25
          candidates: 40
bind: 0.0.0.0:8080
concurrency_scope: cluster
roles: [api, reconciler]
```

Tenant templates are validated at startup against each component's
tenant configuration model, including their provider ids against the
providers the composition built, and nothing is built from them. A
template edit changes future tenants only; existing tenants keep their
recorded configuration, which an operator changes with `PATCH`.

Provider ids are stable identities. A provider's model or dimensions
are not changed under an id; a new model is a new id. Removing an id
from the settings fails startup while any tenant of that component
references it, which the component checks from its own per-tenant table
when the server starts, before a role runs; removing a component from
the composition fails startup the same way while any tenant has a
component row for it, since only a running component can delete or
sweep its resources.

Routing stores. A slot may be filled by an implementation that routes
over several children of the same ABC, placing each key on one child
and recording the choice in its own key registry row's address; every
later operation reads that row, as the fence already does, and
dispatches. No new key is passed anywhere: the tenant id is the key,
and placement is server-side state. That is how tenants are placed
across several vector clusters once one is full, and how a tenant moves
between backends; it does not conflict with "no sharding", which is
about process ownership, not data placement. Not in the first
deployment; a custom composition wires it when needed.

The earlier `resource_initializer.py` proposal (edwinyyyu/MemMachine,
e134c531) is superseded rather than improved: its three problems,
custom logic per builder, no dynamic registration, and stringly typed
dependencies, dissolve when there is no builder layer, because the
constructor is the builder, a class is available by being imported, and
a dependency is a typed parameter.

## Startup and wiring

`memmachine serve [--compose MODULE:FUNCTION]` reads the settings, runs
the composition, verifies schema and scope, then starts the roles.
Startup fails on the first constructor that raises, naming the resource
and the cause, before the socket is bound. Shutdown runs in reverse:
routers stop accepting, a reconciler finishes its current job,
resources close.

Nothing per tenant is created after startup: a request reads rows and
calls stores with the tenant id, and a process holds no per-tenant
state beyond a store's private cache where a backend needs one.

## Server API

Prefix `/v1`. Tenant ids in paths are UUIDs; names are looked up
explicitly. Bodies are JSON. `?wait=<seconds>` on a lifecycle request
delays the response until the request's jobs are done or the wait
elapses, and on ingest and event deletion until every subsystem has
processed the batch; the status tells which.

Tenants and their components:

| Method and path | Effect | Status |
| --- | --- | --- |
| `POST /v1/tenants` | create; body `name`, optional `template`, per-component `config` overrides | 201 with the tenant and its component rows, their jobs run inline; with `wait`, once no `provision` job is pending; 409 `tenant_exists`; 422 |
| `GET /v1/tenants?name=` | look up by name | 200; 404 |
| `GET /v1/tenants?prefix=&after=&limit=` | list by name, `after` the last name returned | 200 |
| `GET /v1/tenants/{id}` | the tenant, each component row's state, requested section and version, and jobs with attempts and last error | 200; 404 |
| `PATCH /v1/tenants/{id}` | rename; body `name` | 200; 404; 409 `tenant_exists` or `tenant_not_active` |
| `DELETE /v1/tenants/{id}` | start deletion | 202, or 204 with `wait` once deleted; 404 |
| `PUT /v1/tenants/{id}/components/{name}` | enable the component with `config`, or update its configuration | 201 enabled and provisioned inline, 202 while provisioning or applying, 200 with `wait` once applied; 404; 409 `tenant_not_active` or `component_not_active`; 422 |
| `DELETE /v1/tenants/{id}/components/{name}` | disable the component | 202, or 204 with `wait` once deleted; 404; 409 |

Events, under `/v1/tenants/{id}`:

| Method and path | Effect | Status |
| --- | --- | --- |
| `POST .../events` | ingest a batch | 202 with stored ids, skipped ids and the head position, or with `wait` 200 once every subsystem has processed it and 202 with the watermarks so far otherwise; 404; 409; 422 |
| `GET .../events/{event_id}` | one event | 200; 404 |
| `GET .../events?filter=&since=&until=&after=&limit=` | list in ingestion order; `filter` a JSON tree, URL-encoded; `after` the last position returned | 200 |
| `POST .../events/delete` | body `ids` | 202 with the head position, or 200 with `wait` |

Episodic memory, under `/v1/tenants/{id}/episodic-memory`:

| Method and path | Effect | Status |
| --- | --- | --- |
| `POST .../search` | body `query`; the search options, each optional with the tenant's default: `limit`, `min_cosine_similarity`, `expand_context`, `rerank` (`reranker`, `candidates`, `min_score`, or `null` for none); the system filters `since`, `until`, `session_ids`, `source_ids`, `block_kinds`; `filter` (JSON tree); `format` (dates, times, locale, timezone for `text`) | 200 with up to `limit` hits in descending score |
| `POST .../expand` | body `anchor` (segment or event uuid), `before`, `after` (segments), `since`, `until`, `source_ids`, `block_kinds`, `filter`, `format` | 200 with `before` and `after`, the segments on each side in order within the anchor's session, never the anchor itself, and their `text` |
| `GET ...` | `watermark` and `head`, the lag being their difference | 200 |

Event body: `id` (optional UUID; a caller that retries a request
supplies ids, since a retry without them stores the events again),
`timestamp` (optional, with offset; server time if absent),
`session_id` (optional string), `source_id` (optional string),
`context` (an object of parts keyed by kind, for example `{"author":
{"name": "Alice"}}`), `blocks` (list of blocks, each `{kind, ...}` of a
registered kind; `{kind: text, text}` is built in), `properties`
(scalar values under legal keys; what `filter` sees). An event is
returned as it was ingested, plus `position`.
Search hit: `score`, `seed` (the index in `segments` of the matched
segment), `segments` (each with `uuid`, `event_id`, `position`,
`index`, `offset`, `timestamp` with offset, `session_id`, `source_id`,
`context`, `block`, `properties`), `text` (the window rendered with
`format`). An expansion returns `before` and `after`, each a list of the
same segment shape, and their `text`.

Errors: one handler for the domain error hierarchy maps to a status and
a body `{error: {code, message}}` with a closed set of codes:
`tenant_not_found`, `tenant_not_active`, `tenant_exists`,
`component_not_enabled`, `component_not_active`, `component_exists`,
`invalid_request`, `provider_unavailable`, `internal`. No traceback
leaves the process. Everything unmapped is 500 `internal` with the
traceback logged. Every request is answered; no path drops the
connection.

MCP: rebuilt over the same component objects, with the tenant id taken
from a header; not designed here, except for one rule carried over from
claude-memory: the tools expose no expansion counts to the model.
Segments are not an intuitive unit for an agent, so a tool's expansion
is a fixed maximum the deployment or the tenant's defaults choose, and
the model asks for context, not for a number of it. Programmers keep
the numbers in the API, whose unit is learned once.

Clients: the Python and TypeScript clients are generated from the OpenAPI
document, not mirrored by hand.

## Schema management

Two kinds of schema:

1. Component schema: the tenant registry tables, each store's registry
   and data tables, the key registry, each component's per-tenant
   table, and the native containers of the vector backends. Static,
   versioned, migrated by an operator's command.
2. Per-tenant resources: rows, in every store. Tenant-specific DDL is
   the only DDL allowed outside `memmachine schema upgrade`, and no
   store in this design uses it; the usearch store's index file per
   tenant is a file, not schema.

Component schema:

- Each SQL-backed component owns an Alembic script directory beside its
  code and its own version table (`schema_version_<component>`), so a
  library user composing some components migrates only those.
  Migrations are written from Alembic autogenerate diffs against the
  component's metadata, with the environment's `include_object`
  limited to the component's own table prefix so the diff never sees
  another component's tables in the shared database; the metadata is
  never applied with `create_all`.
- Each vector store owns `provision_containers()`, which
  idempotently creates one container per embedder the composition
  built.
- `memmachine schema upgrade --settings PATH` is the only thing that runs
  component DDL. It runs the composition to learn which components use
  which database, then per database, under `pg_advisory_xact_lock` on
  PostgreSQL or the file's write lock on SQLite, upgrades each
  component assigned to it to head in registration order; then, outside
  that transaction, creates the property expression indexes the
  settings name (`CREATE INDEX CONCURRENTLY IF NOT EXISTS` on
  PostgreSQL, which cannot run inside a transaction, plain `CREATE
  INDEX IF NOT EXISTS` on SQLite) and provisions containers. Initial
  setup is an upgrade from an empty database; there is no separate
  path. It runs from a deploy job, an init container, or a shell
  before `serve`.
- `memmachine serve` verifies at startup that no component's version
  table is behind the head its code carries: a version the code knows
  and that is not its head is behind, and startup fails naming the
  component and both versions; the head passes; a version the code
  does not know is a later release's, is allowed with a log line, and
  is what a rollout produces (below). `memmachine schema status
  --settings PATH` prints the same comparison, the per-container
  registry counts, and the component tombstone count by stage (rounds
  in progress, clean and within retention, awaiting verification).
- Rolling deployments: a migration must keep the previous release's
  code working (expand and contract: add before the code that reads,
  remove after the code that writes is gone), because during a rollout
  processes of both releases run against one schema, and a process of
  the previous release may restart after the migration. A migration
  that cannot is a release note that requires a stop.
- `memmachine events compact --settings PATH` compacts each live
  tenant's log below the minimum of its subsystems' watermarks, bounded
  per tenant per run (`design/components/event_store.md`).

## What is reused and what is removed

Reused, with the change named:

- `episodic_memory/event_memory/`: `EventMemory` renamed to episodic
  memory, the segmenters, the derivers, the data types, and the segment
  store (UUID keys in place of incarnations, `purge_partition`, a
  stateless handle in place of the incarnation-bound one). The text
  segmenter splits with the standard-library port of
  `RecursiveCharacterTextSplitter` from `agentic_expansion` (commit
  10ed25a6), verified against the original by a differential test, so
  `langchain-text-splitters` leaves the dependencies; with the Bedrock
  embedder an optional extra, as every backend client is,
  `langchain-core` and the `langsmith` tree it pulls in leave the base
  install with it.
- `common/vector_store/`: the four implementations, with the registry
  replaced by rows in the key registry checked after each operation,
  the `config` parameter removed from `create_collection`, containers
  provisioned by the schema command, and the two SQLite stores on
  shared tables.
- `common/filter/`: the filter expression tree, as the closed union on
  the reference branch; the string parser goes.
- `common/embedder/`, `common/language_model/`, `common/reranker/`,
  each class taking its settings model in its constructor in place of
  the separate configuration model.
- `common/metrics_factory/`, `common/payload_codec/`.
- `enable_sqlite_foreign_keys` and the engine construction in
  `common/resource_manager/database_manager.py`, as the `sqlite` and
  `postgres` factory kinds.

New: the event store, the key registry, the ingest service, the tenant
service and reconciler, the standard composition and the settings
models, the schema command, the routers.

Removed: `main/memmachine.py`; `episodic_memory/episodic_memory.py`,
`episodic_memory_manager.py`, `instance_lru_cache.py`,
`long_term_memory/`, `declarative_memory/`, `short_term_memory/`;
`common/session_manager/`, `common/episode_store/`,
`common/vector_graph_store/`, `common/neo4j_utils.py`;
`common/resource_manager/` (the managers, the locator functions, the
`CommonResourceManager` protocol); `common/configuration/`;
`common/filter/filter_parser.py`; `server/api_v2/` including the config
router and the traceback-carrying error model; `semantic_memory/`;
`retrieval_agent/`; `installation/`; `memmachine_common/api/spec.py`.

## What must be built first

The first deployment of this design carries no data forward, so nothing
is urgent because existing data would be expensive to change. What is
urgent is what the first deployment freezes: anything written into
stored records, and anything a client integrates against. Those parts
must be in place before the first tenant is created, whatever else is
still partial. Everything else can follow without touching a record or
a client.

In the first deployment, because the first records freeze it:

1. The tenant id as the store key, minted per life, and the tenant table
   with its tombstones, with create, delete and status recorded there.
   A key scheme written into every record cannot be changed later on
   backends that cannot rename; the reconciler that executes deletions
   may land after, but the rows and jobs it needs must exist before the
   first delete. This includes the key being the physical key inside
   each store's own tables: the segment store's rows keyed by the
   tenant id with no incarnation column, because a store's schema is
   permanent from its first row, and a store-private identity written
   then could never be removed.
2. The event store as the system of record, with positions. Derived
   data written without it can never be reprocessed or handed to a
   subsystem added later, because no other copy of the events exists in
   this server.
3. What goes into stored records: system fields under the reserved
   `memmachine_` keys, scalar-only bounded properties, the declared
   index schema per vector store, and containers per embedder. Every
   vector payload and segment row written freezes them, and most
   backends cannot rewrite a payload key in place.
4. A registry row for every vector key, with strict create and the check
   after each operation. The row is what makes a key's garbage
   addressable at all; purging and the tombstone pass can follow,
   but a key that was never registered can never be purged.
5. The public surface: the v1 API, the event shape, the filter as a JSON
   tree, limits as maximums, one error handler. Clients freeze it, and
   changing it later is a client migration.
6. Alembic from the first migration, which is the schema itself. Free at
   the start and painful once `create_all` has run anywhere.

Before churn, not before the first byte: the reconciler role executing
durable deletions, sweeps and replays, and the tombstone pass. None
writes anything into records, and a deployment with few deletions can
run with them recorded as jobs until it lands. "No implicit creation"
is not a task in a new server; there is no path to remove.

At any time: the generated settings schema and example, the
concurrency scope check, generated clients, the SQLite stores on shared
tables, MCP. Internal or reversible.

The first deployment's minimum is therefore narrower than this
document: the tenant table and jobs, the event store, episodic memory
over the shipped segment store and one registry-backed vector store, the
property conventions, the v1 API for tenants, events and search, and
Alembic. Everything else is additive, and the risk to guard against is a
partial implementation that starts writing records before one of the
six items above is in place.

## Component specifications

One specification per component, under `design/components/`, each with
its API, storage, fencing, settings, and the changes it requires of an
existing component: `README.md` (conventions), `tenant_service.md`
(registry, jobs, reconciler, sweep), `key_registry.md`,
`event_store.md`, `segment_store.md`, `vector_store.md`,
`episodic_memory.md`, `episodic_memory_manager.md`, `context.md`,
`blocks.md`, `ingest_service.md`, `filters_and_properties.md`,
`server_and_settings.md`. Where a specification and this document
disagree, the specification is the newer statement and this document
is corrected to it.

## Relation to open issues

- #1574: this document is the target for every row.
- #1548: kept; UUID keys replace incarnations, `purge_partition` is
  added, the incarnation-bound partition handle gives way to a
  stateless handle whose operations take no key, and
  `open_or_create_partition` goes.
- #1530: agrees on the outcome, the store ABCs keeping only the strict
  create, and on the reason: only the caller knows why an existing row
  is acceptable. The two prove different things at the same signature.
  #1530's names are reused by design and successive lives are separated
  by incarnations, so its create raising means "this exists"; here keys
  are never reused, the tenant service's tombstone enforces it, and a
  row under a key in a store is evidence of an interrupted attempt or
  a library caller's reuse.
- #1531: the reference for the concurrency scope idea. The levels,
  the declarations and the deployment-side check above are this
  design's own; they differ from #1531 where the key registry changes
  what a store can promise (Qdrant, Milvus and sqlite-vec widen).
- #1571: the router's 404/409 mapping under "Component contract";
  handles are stateless and built per request, so there is nothing to
  evict.
- #1575, #1576, #1577: resolved by construction: declared components, no
  implicit creation, durable jobs with retry.
- #1572, #1573, #1564, #1565, #1537, #1563: the vector store contract
  above: containers from configuration, the key registry as the record
  of every key, checks after each operation, tenants as values,
  single-use keys, purging plus the tenant service's tombstone
  pass. In-flight registry PRs are measured against that section.
- #1535: one declared, typed `indexed_properties` schema per vector
  store, in configuration, under "Properties and filtering".
- #1570: "Schema management".
- #1542: SQLite pragmas become fields of the SQLite engine's settings
  (`busy_timeout`, `journal_mode`); the reconciler's retry covers the
  busy-timeout raise.

## Open questions

- Hierarchy: flat tenants with prefix listing, proposed, or a parent
  column with cascading delete as jobs. If semantic memory returns, an
  org-level share across tenants is a real requirement the flat model
  cannot express.
- Event size limits: the values of `blocks.max_bytes`,
  `context.max_bytes` and the property bounds, which are settings
  without measured defaults.
- Readable metadata as a `json` block kind, or a designed field.
- Retention: deleting events by age, by source or by session, as a job
  kind.
- Which vector backends the first release implements beyond the four
  existing ones and pgvector; the six remote backends are surveyed for
  the contract only.
- Log compaction as a scheduled duty rather than an operator command,
  if uncompacted logs ever matter.
