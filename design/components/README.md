# Component specifications

One file per component of `design/server_redesign.md`. Each states the
component's purpose, what it is constructed with, its types and API, its
storage and fencing where it has any, its settings, and, for a component
that exists today, the changes required of it with file references to
`speedkick` at 7752e4cb under `packages/server/src/memmachine_server`.

Conventions shared by every specification:

- Identifiers are `uuid.UUID` values in every signature, row and model,
  never strings. A backend that wants a string receives the 32 lowercase
  hex characters of the UUID, rendered at the store's boundary and
  nowhere else.
- A key is a tenant id. Stores take it and nothing else, fence on it,
  and never mint an identity of their own.
- `Progress` is an enum with two members, `DONE` and `MORE`, returned by
  every bounded, repeatable step. `MORE` means "call again"; `DONE`
  means "nothing remains". A step may raise instead, and its caller
  retries.
- Vocabulary of removal: `delete` is the logical unlink, O(1); `purge`
  is one bounded batch of physical removal, returning `Progress`; a
  sweep is a run that calls `purge` until `DONE` or until a step's time
  budget (`reconciler.step_duration`) is spent. A `sweep` job sweeps
  one tenant for one component; the tombstone pass resets the sweeps of
  deleted tenants so late garbage is purged too.
- Errors are one hierarchy under `MemMachineError`, named for the
  condition, never for a driver: `KeyExistsError`, `KeyNotLiveError`,
  `KeyReusedError`, `TenantNotFoundError`, `TenantNotActiveError`,
  `TenantExistsError`, `InvalidTenantConfigError`,
  `UndeclaredPropertyKeyError`, `ProviderUnavailableError`,
  `AttemptsExhaustedError`. The HTTP layer maps the hierarchy once.
- Every method that is a hook or a bounded step is idempotent: calling
  it again after any partial outcome completes it.
- A resource's constructor takes its dependencies as typed parameters
  (engines, clients, stores, other resources) and its settings as one
  Pydantic model, `<Component>Settings`. The line between them is
  serializability: a settings model holds everything that can arrive
  from configuration or a request (scalars, names, bounds, options)
  and is validated by Pydantic; a dependency is a live object that
  never appears in a document. The current `EventMemoryParams`
  (`InstanceOf[Embedder]` beside scalar fields) mixes the two, which is
  what this rule removes. No numeric default is given here, since none
  has been measured.
- Every store has two ABCs: the store (`EventStore`, `SegmentStore`,
  `VectorStore`), the resource and the only place a key is named, for
  lifecycle and for constructing handles; and the handle
  (`EventPartition`, `SegmentPartition`, `VectorCollection`), the data
  surface, bound to one key at construction, with no method taking a
  key. A data consumer holds the handle only, so it cannot reach
  lifecycle and cannot name a wrong key; the store is held by the
  component that owns lifecycle. Each backend implements both.
- A handle is stateless: constructed by the store without I/O, holding
  the key and the store's shared resources, fenced by the registry row
  on every operation; nothing in it goes stale, and there is nothing to
  open, close or evict.
- Nothing per tenant is opened, closed or held: an operation takes the
  key, reads what it needs, and returns; a handle is a binding of the
  key, not a thing opened.

Files:

- `tenant_service.md`: the tenant registry, jobs, the reconciler role,
  the tombstone pass, the component registration protocol.
- `key_registry.md`: per-key bookkeeping for stores whose data is not
  in SQL, and the scoped view a store receives.
- `event_store.md`: the system of record for events.
- `segment_store.md`: the segment store as shipped in #1548 and what
  changes.
- `vector_store.md`: the vector store contract, its registry rows,
  containers, declared properties, and per-backend notes.
- `episodic_memory.md`: `EpisodicMemory`, the current `EventMemory`.
- `episodic_memory_manager.md`: the resource that builds and dispatches
  to `EpisodicMemory` objects and registers with the tenant service.
- `context.md`: the typed, non-filterable parts attached to content,
  their registration, composition, propagation and rendering.
- `blocks.md`: block kinds as a registered family, and the kind as a
  system field of the segment.
- `ingest_service.md`: the write path from the API to the event store
  and the subsystems.
- `filters_and_properties.md`: the filter expression tree, property
  keys and values, and the reserved namespace.
- `server_and_settings.md`: the `Server` object, roles, routers, error
  mapping, and the settings models.

## Naming of values

One word per kind of value, and the kind is decided by where the value
comes from and where it lives:

| Word | What it names | Type or field |
| --- | --- | --- |
| settings | deployment-level values from the environment and the settings file; one Pydantic model per resource class, nested into `ServerSettings`; read at startup, never mutated | `<Resource>Settings`, `settings: ...`, `memmachine serve --settings PATH`, `memmachine settings schema` |
| tenant configuration | per-tenant values recorded on the tenant row, one section per component, validated by the component's model; each field is an option, mutable or immutable | `<Component>TenantConfig`, `tenants.config`, `config_version` |
| template | a named tenant configuration in the settings, copied at create | `tenant_templates` |
| overrides | the sections a create or update request supplies on top of a template or the recorded configuration | request field `config` |
| defaults | the options of a tenant configuration that fill request parameters a request omits | `SearchDefaults` |
| partition or collection configuration | per-key values a store records in its registry row at create (codec configuration, container) | `<Store>PartitionConfig`, column `config` |
| request parameters | fields of a request model; a request may set any of them | `SearchRequest`, `ExpandRequest` |
| parameters and arguments | a parameter is in a signature, an argument is the value passed; a resource's constructor takes its dependencies and one settings model as parameters | |
| job arguments | what a job's hook is called with, recorded on the job row | `tenant_jobs.arguments` |

In identifiers the short forms are Python's conventions and the only
ones used: `Settings` for deployment values, as `pydantic_settings`
names them, and `Config` for tenant and per-key configuration, as
`logging.config`, `configparser` and Pydantic's `ConfigDict` do; the
prose says "configuration" as a word. Not used as identifiers in new
code: `conf`, `cfg`, `configuration`, `params`, `args`, `payload`
(except in "payload codec", the existing component that encodes blocks
and context), `options` other than for the fields of a tenant config
section.

## Contracts are ABCs

Every contract a component implements is an abstract base class, never
a `Protocol`: abstract methods are enforced when an object is
instantiated, not only when a type checker runs; `isinstance` holds,
which the composition's scope check and Pydantic's `InstanceOf` rely
on; `@override` is checked against a real base; and shared behaviour
(a formatting helper, a default) has a home. A third-party
implementation imports the base class, which it does anyway to register
a kind. A `Protocol` is used only to describe the shape of an object we
do not define, of which the design has none.

## SQL type mapping

Every SQL-backed component's schema below uses these SQLAlchemy types,
with the dialect mappings the schema command emits:

| SQLAlchemy | PostgreSQL | SQLite | Note |
| --- | --- | --- | --- |
| `Uuid(native_uuid=True)` | `UUID` | `CHAR(32)`, hex | keys and identifiers |
| `Integer` | `INTEGER` | `INTEGER` | small counters; SQLite integer primary keys alias rowid |
| `BigInteger().with_variant(Integer, "sqlite")` | `BIGINT` | `INTEGER` | positions, autoincrement ids |
| `Text` | `TEXT` | `TEXT` | names, opaque strings |
| `String(n)` + `CheckConstraint` | `VARCHAR(n)` | `VARCHAR(n)` | enumerations, never a native enum type, so a migration adds a value without an `ALTER TYPE` |
| `DateTime(timezone=True)` | `TIMESTAMP WITH TIME ZONE` | `DATETIME` as ISO text, offset discarded | values are normalized to UTC before persisting |
| `JSON().with_variant(JSONB, "postgresql")` | `JSONB` | `JSON` (text) | properties, configuration, addresses |
| `LargeBinary` | `BYTEA` | `BLOB` | codec-encoded payloads |
| `Boolean` | `BOOLEAN` | `INTEGER` 0/1 with check | |
| `Float` | `DOUBLE PRECISION` | `REAL` | |

Constraints are declared in metadata (primary keys, unique constraints,
check constraints, foreign keys with `ON DELETE CASCADE`), so Alembic
autogenerate sees them. Indexes are named `<table>__<columns>`.
Timestamps stamped by the database use `func.now()`.

## Code conventions

New code follows the reference branch's rules, and old code is brought
to them where the change is mechanical: every overriding method is
marked `@override` and carries no docstring (the contract is on the
ABC); `D102`, `D213` and `RET504` are enforced; lint ignores are pruned
to load-bearing rules with a stated rationale; directly imported
dependencies are declared explicitly and unused ones removed. Nothing
here changes behaviour.
