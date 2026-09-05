# Tenant service

New component. Control plane: creates, renames, configures and deletes
tenants; records jobs; runs the reconciler role and the tombstone pass;
knows components only through their registrations. The only reader and
writer of the tenant tables.

## Constructed with

- `engine: AsyncEngine`, the tenant database.
- `components: Sequence[TenantComponent]`, the registrations (below).
- `templates: Mapping[str, TenantTemplate]`, validated at construction
  against every component's `tenant_config` model.
- `settings: TenantServiceSettings`: `reconciler.poll_interval`,
  `reconciler.jobs_per_pass`, `reconciler.step_duration`,
  `reconciler.purge_interval`,
  `reconciler.backoff`, `reconciler.stuck_after`,
  `reconciler.sweep_interval`, `tombstones.retention`.

## Component registration protocol

```python
class TenantComponent(ABC):
    name: str
    tenant_config: type[BaseModel]   # fields marked mutable or immutable

    @abstractmethod
    async def provision(self, tenant_id: UUID, section: BaseModel) -> None: ...
    @abstractmethod
    async def delete(self, tenant_id: UUID) -> None: ...
    @abstractmethod
    async def purge(self, tenant_id: UUID) -> Progress: ...
    @abstractmethod
    def validate_update(self, old: BaseModel, new: BaseModel) -> None: ...
    @abstractmethod
    async def replay(self, tenant_id: UUID) -> Progress: ...
```

- `provision(tenant_id, section)`: idempotent by provenance. Create the
  component's resources under the key if absent (a `live` row is its
  own earlier attempt; a `creating` row is resumed; a `dropping` row
  raises `KeyReusedError`), verify immutable options if present, apply
  mutable ones, record the section and its version in the component's
  own per-tenant table. Complete on return.
- `delete(tenant_id)`: idempotent; make the tenant unreachable in every
  one of the component's stores (each store's logical delete) and
  remove its per-tenant row. Fast; no purging.
- `purge(tenant_id) -> Progress`: remove one bounded batch of what the
  component's stores hold under the key; `DONE` when nothing is found.
  One batch per call; the sweep job loops. Never called on a live key
  (see "Serialization").
- `validate_update(old, new)`: raise `InvalidTenantConfigError`
  if `new` changes an immutable field.
- `replay(tenant_id) -> Progress`: process a bounded amount of what
  the event store's log holds for this tenant beyond the component's
  watermark; `DONE` when the watermark is at the head. The component's
  only processing path. The tenant service defines the kind and its
  semantics; the component implements the hook. There are no
  component-defined job kinds: every kind is defined, scheduled and
  given its semantics by the tenant service, so two components cannot
  mean different things by one name.

The tenant service imports nothing from a component; a component
imports this protocol and nothing else from the tenant package.

## Schema

Types per the mapping in `README.md`.

`tenants`:

| column | type | constraint |
| --- | --- | --- |
| `id` | `Uuid` | primary key; minted `uuid4` at create; never reused |
| `name` | `Text` | null; unique index `tenants__name` (partial on PostgreSQL, `WHERE name IS NOT NULL`; SQLite treats NULLs as distinct) |
| `former_name` | `Text` | null |
| `state` | `String(16)` | not null; check in (`provisioning`, `active`, `deleting`, `deleted`) |
| `config` | `JSON` (`JSONB` on PostgreSQL) | not null |
| `config_version` | `Integer` | not null, default 1 |
| `created_at`, `updated_at` | `DateTime(timezone=True)` | not null, `func.now()` |
| `deleted_at`, `swept_at` | `DateTime(timezone=True)` | null |

Indexes: `tenants__name` above; `tenants__state_swept_at (state,
swept_at)` for the sweep's "oldest `swept_at` first among `deleted`";
`tenants__name_prefix`, a `text_pattern_ops` index on PostgreSQL for
prefix listing.

`tenant_jobs`:

| column | type | constraint |
| --- | --- | --- |
| `id` | `BigInteger` (autoincrement; `Integer` on SQLite) | primary key |
| `tenant_id` | `Uuid` | not null; foreign key to `tenants.id` (no cascade: rows are removed by the transition that sets `deleted`) |
| `component` | `String(64)` | not null |
| `action` | `String(32)` | not null; check in (`provision`, `delete`, `sweep`, `replay`) |
| `arguments` | `JSON` (`JSONB` on PostgreSQL) | not null |
| `state` | `String(16)` | not null; check in (`pending`, `done`) |
| `attempts` | `Integer` | not null, default 0 |
| `last_outcome` | `String(8)` | null; check in (`more`, `error`) |
| `last_error` | `Text` | null |
| `last_run_at` | `DateTime(timezone=True)` | null |
| `created_at`, `updated_at` | `DateTime(timezone=True)` | not null, `func.now()` |

The row records what happened, never what should happen next: when a
job is eligible is computed at claim time from `last_run_at`,
`last_outcome` and `attempts` with the reconciler's settings, so a
change of `purge_interval` or `backoff` applies to every pending job
at once and rewrites no row, and an operator's retry is `attempts = 0`.
A `next_run_at` column would be a prescription written by one version
of the policy and honoured by another.

Constraints and indexes: unique `tenant_jobs__tenant_component_action
(tenant_id, component, action)`; `tenant_jobs__pending (last_run_at)`
partial on PostgreSQL `WHERE state = 'pending'`, the claim's scan
(pending rows are few, so eligibility is evaluated per row, not
indexed);
`tenant_jobs__tenant (tenant_id)` for a tenant's job listing and the
delete request's cancellation of pending jobs.

## Job kinds

Four, all defined by the tenant service, which is the only thing that
defines, schedules and gives semantics to a kind. Three are
control-plane: `provision`, which creates or updates a component's
resources for a tenant; `delete`, which unlinks them, one call, done in
seconds; and `sweep`, which purges them batch by batch until nothing
remains, and again whenever the tombstone pass resets it. One is
data-plane: `replay`, one row per (tenant, component), created by
`provision`, reset to pending by every ingest and deletion, and the
component's only processing path (`episodic_memory_manager.md`). A
component cannot invent a kind.

What guarantees that a client's write is executed fully at least once
is the event store's log and the `replay` job together: every addition
and deletion a client was acknowledged for is an entry, and the job
replays entries from the component's watermark until none remain,
resuming from the watermark after any failure. A client's read has
nothing to guarantee. Lifecycle requests are guaranteed by their jobs.

## Queueing

The job table is a queue: `pending` rows ordered by last run, or by
creation for a job never run, claimed under `FOR UPDATE SKIP LOCKED`,
and serialized per tenant by
the tenant row's lock. There is no per-tenant FIFO beyond that; the
only ordering a tenant's jobs need, `provision` before `delete`, is the
state machine's. The event store's log is the per-tenant data queue,
with one watermark per subsystem.

Neither is a message broker, on purpose. A lifecycle transition and its
jobs are one transaction with the tenant row, and a log entry and the
event it describes are one transaction with the event row; a broker
cannot join a transaction, so it would need an outbox table anyway, at
which point the table is the queue. Volumes are lifecycle events and
repairs, not the request rate. A broker becomes worth having only if
ingest is made asynchronous for every request and fanned out to
consumers outside this process, which the main document leaves open
and this design does not need.

## API

```python
class TenantService:
    async def create(self, name: str, template: str,
                     overrides: Mapping[str, Mapping]) -> Tenant
    async def get(self, tenant_id: UUID) -> Tenant
    async def get_by_name(self, name: str) -> Tenant
    async def list(self, prefix: str | None, cursor: Cursor | None,
                   limit: int) -> Page[Tenant]
    async def rename(self, tenant_id: UUID, name: str) -> Tenant
    async def update_config(self, tenant_id: UUID,
                                   overrides: Mapping[str, Mapping]) -> Tenant
    async def delete(self, tenant_id: UUID) -> Tenant
    async def wait(self, tenant_id: UUID, until: TenantState,
                   timeout: timedelta) -> Tenant
    async def reset_replay(self, tenant_id: UUID) -> None
    async def replay_position(self, tenant_id: UUID) -> Mapping[str, int]
    async def state_of(self, tenant_id: UUID) -> TenantState | None
    async def run_reconciler(self) -> None      # the role's loop
    async def reconcile_tenant(self, tenant_id: UUID) -> None
```

`Tenant` carries the row plus, per component, the applied configuration
version reported by the component, and the jobs with attempts and last
error. `state_of` is what a router calls on a subsystem miss to answer
404 or 409; it is the one method of the service a router calls on the
data path.

Semantics:

- `create`: resolve the template overlaid with overrides, validate each
  section with its component's model (unknown component or invalid
  option: `InvalidTenantConfigError`), insert the row as
  `provisioning` and one `provision` job per component in one
  transaction; a duplicate name raises `TenantExistsError`, nothing
  more. Then `reconcile_tenant` inline, so a single process finishes the
  create in the request; a failing step is left pending.
- `update_config`: validate with `validate_update` per changed
  section; one transaction writes the configuration, increments the version,
  inserts or resets a `provision` job per changed component with the
  version in its arguments; the tenant stays `active`.
- `delete`: one transaction, under the tenant row's lock: `state =
  deleting`, `former_name = name`, `name = NULL`, `deleted_at = now()`;
  mark every pending `provision` and component-defined job done; insert
  one `delete` job per component. Allowed from `provisioning` and
  `active`; idempotent while `deleting`.
- `wait`: poll the row until the state is reached or the timeout
  elapses; used by `?wait=`.

## Reconciler role

- Claim: `SELECT ... FROM tenant_jobs WHERE state = 'pending' AND
  (last_run_at IS NULL OR last_run_at + :delay <= now()) ORDER BY
  COALESCE(last_run_at, created_at) LIMIT n FOR UPDATE SKIP LOCKED`,
  where `:delay` is a `CASE` over `last_outcome` and `attempts` bound
  from the settings (`purge_interval` after `more`; `backoff` raised
  to `attempts` after `error`). A `provision` or `delete` step then
  also takes `SELECT ... FROM tenants WHERE id = ? FOR UPDATE`; a
  `replay` step holds only its own job row, so a tenant's subsystems
  replay in parallel while each is one consumer. Locks are held for the
  step. On SQLite, `BEGIN IMMEDIATE` and no `SKIP LOCKED`.
- Serialization: every lifecycle transition and every lifecycle step
  holds the tenant row's lock, so transitions and lifecycle steps for
  one tenant are totally ordered; a `replay` step holds its job row
  only, and a delete request's cancellation of that row waits for a
  running step, which orders the two. A step re-reads the state under the lock:
  a `provision` or
  component-defined step on a `deleting` tenant marks itself done and
  calls nothing. A delete request waits for a running step, then
  cancels pending `provision` jobs. No `provision` hook runs after a
  `delete` hook; `purge` never sees a live key.
- Execute: `provision` calls the hook with the section at the arguments'
  version and marks done. `delete` calls `delete` once and marks done.
  `sweep` calls `purge` repeatedly until it returns `DONE`, which marks
  the job done, or until the step has run for
  `reconciler.step_duration`, which records `last_outcome = more` and
  `last_run_at`. `replay` calls the hook the same way, and its row is
  never removed while the tenant lives: `DONE` sets it `done` until the
  next reset. An exception records `error`,
  `last_error`, `last_run_at` and `attempts + 1`; eligibility follows at the
  next claim. The transaction that marks the last `provision` job done sets
  `active`; the one that marks the last `delete` job done sets
  `deleted`, inserts one `sweep` job per component, and removes the
  `provision`, `delete` and `replay` rows; the `sweep` rows stay until
  the tombstone is pruned.
- Reset: the ingest service sets a tenant's `replay` jobs to `pending`
  through `reset_replay(tenant_id)`, in the ingest's own transaction
  where the engines are shared and after its commit otherwise. A
  running `replay` step keeps running; the reset only sets `pending`,
  so the row is claimable again when the step ends.
- Tombstone pass, every `sweep_interval`, in every reconciler process
  without exclusion: claim `deleted` rows with `FOR UPDATE SKIP LOCKED`,
  oldest `swept_at` first, bounded per call; reset each one's `sweep`
  jobs to `pending` and stamp `swept_at`. The sweeps then run as jobs.
  When a tombstone's `sweep` jobs have all completed with `purge`
  finding nothing on their first batch, and `deleted_at < now() -
  tombstones.retention` on the database clock, the pass deletes the
  row and its job rows. The retention assumes every remote client has a
  request timeout; the composition refuses one without.
- Cost: one connection per executing job, bounded by `jobs_per_pass`.

## Concurrency scope

`cluster` on PostgreSQL; `host` on a SQLite file; `process` on in-memory
SQLite.

## Changes to existing code

Replaces `common/session_manager/` (`SessionDataManager`, the `sessions`
and `short_term_memory_data` tables), `MemMachine.delete_session` and
`_delete_session_worker` (`main/memmachine.py:343`, `:635`), and the
implicit creation in `add_episodes` and `_search_episodic_memory`
(`:767`, `:826`). Nothing is carried over.

## Race matrix

Every concurrent pair on one tenant, and the defined outcome. "Serialized"
means the tenant row's lock orders the two and the second re-reads state.

| First | Concurrent | Outcome |
| --- | --- | --- |
| create, name N | create, name N | unique index: the second raises `TenantExistsError`; no store touched |
| create (id from a restored registry or a library caller) | any | primary key: `TenantExistsError`; no store touched |
| provision step running | delete request | delete waits for the step (tenant row lock), then marks pending `provision` jobs done and inserts `delete` jobs; later `provision` steps see `deleting` and mark themselves done; created resources are purged by the sweeps |
| delete request | delete request | serialized; the second finds `deleting`, inserts nothing, responds the same 202 |
| provision step | delete or sweep step, same tenant | cannot overlap: serialized, and the delete request already cancelled pending `provision` jobs; no `provision` hook runs after a `delete` hook, and no `sweep` exists before `deleted` |
| configuration update | delete request | serialized; update after delete raises `TenantNotActiveError`; update before delete leaves a `provision` job that sees `deleting` and marks itself done |
| configuration update | configuration update | serialized; both versions recorded in order; the `provision` job row is reset to the later version (a running step finishes first, then the reset lands); applied version is the latest |
| rename | create with that name | unique index; the loser raises `TenantExistsError` |
| rename | delete | serialized; rename on `deleting` raises `TenantNotActiveError` |
| any step | reconciler crash mid-step | the database releases the job and tenant row locks; the job stays `pending` and is claimable at once; hooks are idempotent; a half-done `provision` resumes from `creating` rows |
| create request executing steps inline | api process crash | same as above; a reconciler finishes; the caller's `?wait=` fails and it polls |
| claim | claim, other reconciler | `FOR UPDATE SKIP LOCKED`: a job has one executor; on SQLite the file lock serializes |
| sweep job | tombstone pass | the pass only resets a `sweep` row to pending; a running sweep finishes its step first, since the reset waits for the row lock |
| tombstone pass | tombstone pass, other reconciler | `SKIP LOCKED` on tombstone rows; a row is reset by one pass at a time; sweep jobs are claimed like any job |
| `replay` step | delete request | serialized; a `replay` step on `deleting` marks itself done |
| `replay` step | tenant still `provisioning` | rescheduled with backoff; `replay` runs only on `active` |
| data operation | delete commit | store fences: in-statement for SQL stores, check-after for the rest; the operation raises `KeyNotLiveError`, the router answers 409 while the row exists and 404 after; a remote write already sent is purged by the sweep or the sweep |
| data operation | tenant `provisioning` | the component's per-tenant row is written at the end of `provision`, so the operation finds no row and the router answers 409 `tenant_not_active` |
| data operation | configuration update | the operation uses the row it read at its start; the next request uses the new configuration |
| `schema upgrade` | `serve` starting | `serve` verifies and fails if not at head; a deployment orders upgrade before serve; during a rollout, expand/contract keeps both releases valid |
