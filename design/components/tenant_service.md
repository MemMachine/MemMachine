# Tenant service

New component. Control plane: creates, renames and deletes tenants;
enables, configures and disables components on a tenant; records jobs;
runs the reconciler role and the tombstone pass; knows components only
through their registrations. The only reader and writer of the tenant
tables.

A tenant and the components enabled on it have separate lifecycles. A
tenant is a name, an id and a state; it exists the moment its row is
inserted and holds nothing by itself. Each component's per-tenant
resources are enabled, configured and disabled on their own, each with
its own state, jobs and tombstone. Deleting a tenant disables every
component enabled on it and then retires the id. A template is a set of
component sections a create request enables at once; it is a
convenience, not a coupling.

## Constructed with

- `engine: AsyncEngine`, the tenant database.
- `components: Sequence[TenantComponent]`, the registrations (below).
- `templates: Mapping[str, TenantTemplate]`, validated at construction
  against every component's `tenant_config` model.
- `settings: TenantServiceSettings`: `reconciler.poll_interval`,
  `reconciler.concurrency`, `reconciler.step_duration`,
  `reconciler.reclaim_after`, `reconciler.sweep_pause`,
  `reconciler.backoff`, `reconciler.max_backoff`,
  `reconciler.stuck_after`, `reconciler.tombstone_interval`,
  `tombstones.retention` (a deployment sets it to a day or more).

## Component registration

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

class MemorySubsystem(TenantComponent):
    @abstractmethod
    async def replay(self, tenant_id: UUID) -> Progress: ...
    @abstractmethod
    async def watermark(self, tenant_id: UUID) -> int: ...
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
  One batch per call; the sweep step loops. Raises on a live key; the
  sweep step calls `delete` first, so it never sees one.
- `validate_update(old, new)`: raise `InvalidTenantConfigError`
  if `new` changes an immutable field.
- `replay(tenant_id) -> Progress` (memory subsystems only): process a
  bounded amount of what the event store's log holds for this tenant
  beyond the subsystem's watermark; `DONE` when the watermark is at the
  head. The subsystem's only processing path. A step that finds no
  per-tenant row returns `DONE`.
- `watermark(tenant_id) -> int` (memory subsystems only): the last log
  position the subsystem has fully processed.

The event store is a `TenantComponent` with an empty section: its
resources are its partition, its `provision` creates it, and it is
enabled on every tenant at creation and cannot be disabled while the
tenant lives. Memory subsystems are `MemorySubsystem`s. The tenant
service imports nothing from a component; a component imports this
module and nothing else from the tenant package.

## Schema

Types per the mapping in `README.md`.

`tenants`:

| column | type | constraint |
| --- | --- | --- |
| `id` | `Uuid` | primary key; minted `uuid4` at create; never reused |
| `name` | `Text` | null; unique index `tenants__name` (partial on PostgreSQL, `WHERE name IS NOT NULL`; SQLite treats NULLs as distinct) |
| `state` | `String(16)` | not null; check in (`active`, `deleting`, `deleted`) |
| `created_at`, `updated_at` | `DateTime(timezone=True)` | not null, `func.now()` |
| `deleted_at` | `DateTime(timezone=True)` | null |

Indexes: `tenants__name` above; `tenants__state (state)` for the
tombstone pass; `tenants__name_prefix`, a `text_pattern_ops` index on
PostgreSQL for prefix listing. The name is released (`NULL`) the moment
deletion starts, so a new tenant can take it at once under a new id. A
`deleted` row keeps its id and `deleted_at` and nothing else, so no
personal data outlives the tenant in the registry.

`tenant_components`, one row per component enabled on a tenant:

| column | type | constraint |
| --- | --- | --- |
| `tenant_id` | `Uuid` | primary key part; foreign key to `tenants.id` |
| `component` | `String(64)` | primary key part |
| `state` | `String(16)` | not null; check in (`provisioning`, `active`, `deleting`, `deleted`) |
| `config` | `JSON` (`JSONB` on PostgreSQL) | not null; the requested section |
| `config_version` | `Integer` | not null, default 1 |
| `created_at`, `updated_at` | `DateTime(timezone=True)` | not null, `func.now()` |
| `deleted_at` | `DateTime(timezone=True)` | null |
| `clean_at` | `DateTime(timezone=True)` | null; when a sweep round last found nothing |

Index: `tenant_components__state_clean (state, clean_at)` for the
tombstone pass. A `deleted` row is the component's tombstone; the
tenant's row is pruned when no component rows remain under it, which
is what keeps the id reserved while anything could remain under it.

`tenant_jobs`:

| column | type | constraint |
| --- | --- | --- |
| `id` | `BigInteger` (autoincrement; `Integer` on SQLite) | primary key |
| `tenant_id` | `Uuid` | not null |
| `component` | `String(64)` | not null; `(tenant_id, component)` foreign key to `tenant_components` `ON DELETE CASCADE` |
| `action` | `String(32)` | not null; check in (`provision`, `delete`, `sweep`, `replay`) |
| `arguments` | `JSON` (`JSONB` on PostgreSQL) | not null |
| `state` | `String(16)` | not null; check in (`pending`, `running`, `done`) |
| `attempts` | `Integer` | not null, default 0 |
| `last_outcome` | `String(8)` | null; check in (`more`, `error`, `clean`, `purged`) |
| `last_error` | `Text` | null |
| `last_run_at` | `DateTime(timezone=True)` | null; when the last step ended |
| `claimed_at` | `DateTime(timezone=True)` | null; when the running step was claimed |
| `created_at`, `updated_at` | `DateTime(timezone=True)` | not null, `func.now()` |

Constraints and indexes: unique `tenant_jobs__tenant_component_action
(tenant_id, component, action)`; `tenant_jobs__state_last_run (state,
last_run_at)` for the claim's scan; the foreign key's cascade removes a
component's job rows with its tombstone, so no job row can outlive the
row it acts on.

The row records what happened, never what should happen next: when a
job is eligible is computed at claim time from `last_run_at`,
`last_outcome` and `attempts` with the reconciler's settings, so a
change of a setting applies to every pending job at once and rewrites
no row, and an operator's retry is `attempts = 0`. A `next_run_at`
column would be a prescription written by one version of the policy
and honored by another.

`arguments` carries what the hook is called with. For `provision` it is
the section and its version: the job carries the section itself
because the component row holds only the current section and no
history, so a job that ran later than the request that made it would
otherwise have nothing to apply; the section is small and is the
request's own content. For the other kinds it is empty.

## Job kinds

Four, all defined by the tenant service, which is the only thing that
defines, schedules and gives semantics to a kind. Three are
control-plane, one row each per component row: `provision`, which
creates or updates a component's resources for a tenant; `delete`,
which unlinks them, one call, done in seconds; and `sweep`, which
purges them batch by batch until nothing remains, and again whenever
the tombstone pass resets it. One is data-plane: `replay`, one row per
memory subsystem enabled on a tenant, inserted by the transaction that
marks the subsystem's `provision` job done, reset to pending by every
ingest and deletion, and the subsystem's only processing path
(`episodic_memory_manager.md`). A component cannot invent a kind.

An unfinished ingestion and a replay are one thing. A client's write is
acknowledged when the events are durable in the event store; what
remains is processing, and the `replay` job is the record that
processing remains: the subsystem's watermark says how far it got, the
log says what is left, and the job row carries nothing but the tenant
and the subsystem. No content and no event ids are copied into a job,
because the log already holds them in order, so nothing can go stale
between the queue and the store. A step that fails leaves the watermark
where it was, and the next step redoes the batch from there; a step
advances the watermark only after every store has the batch, so a batch
recorded as processed is processed.

## Queueing

The job table is a queue: `pending` rows ordered by last run, or by
creation for a job never run, claimed by a state transition (below).
There is no per-tenant FIFO; the only ordering a component's jobs need,
`provision` before `delete`, is the state machine's, and the store
fences make a late step harmless. The event store's log is the
per-tenant data queue, with one watermark per subsystem.

Neither is a message broker, on purpose. A lifecycle transition and its
jobs are one transaction with the component row, and a log entry and
the event it describes are one transaction with the event row; a broker
cannot join a transaction, so it would need an outbox table anyway, at
which point the table is the queue. Volumes are lifecycle events and
repairs, not the request rate.

## API

```python
class TenantService:
    async def create(self, name: str, template: str | None,
                     overrides: Mapping[str, Mapping]) -> Tenant
    async def get(self, tenant_id: UUID) -> Tenant
    async def get_by_name(self, name: str) -> Tenant
    async def list(self, prefix: str | None, after: str | None,
                   limit: int) -> list[Tenant]
    async def rename(self, tenant_id: UUID, name: str) -> Tenant
    async def delete(self, tenant_id: UUID) -> Tenant
    async def enable_component(self, tenant_id: UUID, component: str,
                               section: Mapping) -> Tenant
    async def update_component(self, tenant_id: UUID, component: str,
                               section: Mapping) -> Tenant
    async def disable_component(self, tenant_id: UUID, component: str) -> Tenant
    async def wait(self, tenant_id: UUID, timeout: timedelta) -> Tenant
    async def reset_replay(self, tenant_id: UUID) -> None
    async def watermarks(self, tenant_id: UUID) -> Mapping[str, int]
    async def state_of(self, tenant_id: UUID) -> TenantState | None
    async def component_state(self, tenant_id: UUID,
                              component: str) -> ComponentState | None
    async def run_reconciler(self) -> None      # the role's loop
    async def reconcile_tenant(self, tenant_id: UUID) -> None
```

`Tenant` is the row plus its component rows, each with state, the
requested section and version, and the component's jobs with attempts
and last error; a section is applied when its `provision` job is done.
`TenantState` and `ComponentState` are the two state enumerations.
`state_of` and `component_state` are what a router calls on a subsystem
miss to answer 404 or 409; they are the only methods of the service a
router calls on the data path. `watermarks` asks each enabled memory
subsystem for its watermark; the ingest service's `wait_processed`
polls it.

Semantics:

- `create`: insert the tenant row as `active` (a duplicate name raises
  `TenantExistsError`, nothing more), the event store's component row
  as `provisioning` with its `provision` job, and, when a template is
  named, one component row and `provision` job per section of the
  template overlaid with the overrides, each validated by its
  component's model (unknown component or invalid option:
  `InvalidTenantConfigError`); one transaction. Then `reconcile_tenant`
  inline, so a single process finishes the create in the request; a
  failing step is left pending.
- `enable_component`: the tenant must be `active`; insert the component
  row as `provisioning` with its `provision` job; a row already present
  raises `ComponentExistsError`. Then `reconcile_tenant` inline.
- `update_component`: the row must be `provisioning` or `active`;
  `validate_update` against the recorded section; write the section,
  increment the version, insert or reset the `provision` job with the
  section and version in its arguments; the row's state is unchanged.
- `disable_component`: the row must be `provisioning` or `active` and
  the component must not be the event store: set the row `deleting`,
  `deleted_at = now()`, insert the `delete` job; idempotent while
  `deleting`. A memory subsystem is disabled without the event store;
  the event store goes only with the tenant.
- `delete`: one transaction: `state = deleting`, `name = NULL`,
  `deleted_at = now()`, and every component row that is `provisioning`
  or `active` set `deleting` with a `delete` job. Allowed from `active`;
  idempotent while `deleting`. `rename`, `enable_component` and
  `update_component` on a tenant that is not `active` raise
  `TenantNotActiveError`.
- `wait`: poll the tenant until it has no `provision` or `delete` job
  pending or running, or the timeout elapses; `?wait=` on the lifecycle
  requests. A caller then reads the states it cares about.
- `reset_replay`: `UPDATE` every `replay` row of the tenant to
  `pending` with `attempts = 0`, `last_outcome = NULL`, `last_run_at =
  NULL`, so it is eligible at once; it inserts nothing, so a tenant
  whose replay rows are gone gets none back.

## Reconciler role

Claims are state transitions, not held locks. A step runs a hook that
does remote I/O for up to `reconciler.step_duration`, and no database
lock is held across it: not on the job row, not on a tenant or
component row. What orders steps is the state machine on the component
row, the idempotency of every hook, and the store fences.

- Claim: one job per claim, on its own connection; a reconciler runs up
  to `reconciler.concurrency` steps at once. The claim is one
  statement, `UPDATE tenant_jobs SET state = 'running', claimed_at =
  now(), attempts = attempts + 1 WHERE id = (<the oldest eligible
  row>) AND state = 'pending'`, which is atomic on both dialects; on
  PostgreSQL the subselect uses `FOR UPDATE SKIP LOCKED` so concurrent
  claims do not contend, and on SQLite the statement's own write lock
  serializes them. Eligible means `state = 'pending'` and
  `last_run_at IS NULL OR last_run_at + delay <= now()`, with `delay` a
  `CASE` computed in the statement from the settings: after `error`,
  `backoff` doubled per attempt and capped by `max_backoff`; after
  `more`, `sweep_pause` for a `sweep` and nothing for a `replay`, which
  has more log to process and continues at once; otherwise nothing.
  Ordering is `COALESCE(last_run_at, created_at)`.
- Liveness: a `running` row whose `claimed_at` is older than
  `reconciler.reclaim_after` is eligible again, which is how a crashed
  or hung reconciler's job is taken over. This is the one place time
  decides anything in the reconciler, and it decides liveness only: a
  step that was not dead but slow may run twice, and every hook is
  idempotent and every store fences, so the second run wastes writes
  and changes nothing. `reclaim_after` exceeds `step_duration`; the
  settings model refuses the reverse.
- Execute: re-read the component row (plain read). A `provision` step
  on a row that is not `provisioning` or `active` marks itself done and
  calls nothing; a `delete` or `sweep` step on a row that is not
  `deleting` or `deleted` does the same; a `replay` step runs only on
  an `active` row and marks itself done otherwise. Then the hook:
  `provision` with the section and version in the job's arguments;
  `delete` once; `sweep` calls `delete` once, which is idempotent and
  O(1), and then `purge` repeatedly until `DONE` or the budget is
  spent, so a resource created late by a step that raced the delete is
  unlinked before it is purged; `replay` the same way without the
  `delete`. Completion is a second `UPDATE ... WHERE id = ? AND state
  = 'running' AND claimed_at = ?`, so a step that was reclaimed
  meanwhile records nothing. `DONE` marks the job `done`, `attempts =
  0`, `last_run_at = now()`, and for a `sweep` records `last_outcome =
  clean` when the round's first `purge` call returned `DONE` (the step
  started with `last_outcome` null and its first call found nothing)
  and `purged` otherwise; a spent budget records `more`; an exception
  records `error`, `last_error` and `last_run_at`. The transaction that
  marks a `provision` job done also sets the component row `active` if
  it is `provisioning`, and inserts the `replay` row for a memory
  subsystem if absent. The transaction that marks a `delete` job done
  sets the component row `deleted`, inserts the `sweep` job, removes
  the component's `provision` and `replay` rows, and, if the tenant is
  `deleting` and no component row of it is outside `deleted`, sets the
  tenant `deleted`.
- Reset: the ingest service calls `reset_replay(tenant_id)` after its
  own transaction commits (in the same transaction where the engines
  are shared). A running `replay` step keeps running and the row is
  claimable again when it ends.
- Inline execution: `reconcile_tenant(tenant_id)` claims and executes
  the tenant's eligible jobs through the same claim, after the request's
  transaction has committed. `create`, `enable_component` and, with
  `ingest.inline` or the reconciler role, ingest use it.
- Tombstone pass, every `tombstone_interval`, in every reconciler
  process without exclusion, claiming rows with `FOR UPDATE SKIP LOCKED`
  on PostgreSQL and under the write lock on SQLite, bounded per call.
  The rows record only what happened and when (`deleted_at`,
  `clean_at`, the sweep job's `last_run_at`); the retention is policy,
  a setting, and every comparison below is evaluated in the claim
  statement as datetime arithmetic with the database's `now()`, never
  on a process's clock and never stored as a due time. For each
  component row in `deleted` whose `sweep` job is `done` (a pending or
  running sweep is a round in progress, skipped):
  - `clean_at` null and the round `purged`: reset the sweep, a new
    round.
  - `clean_at` null and the round `clean`: `clean_at = now()`.
  - `clean_at` set and `now() < clean_at + retention`: skip.
  - `clean_at` set and the retention elapsed, and the round ended
    before it (`last_run_at < clean_at + retention`): reset the sweep,
    the verification round.
  - `clean_at` set, the retention elapsed, and the round ended after
    it: `clean` removes the component row, and the cascade its job
    rows; `purged` sets `clean_at` null and resets the sweep.
  Then every `deleting` tenant whose component rows are all `deleted`
  is set `deleted`, and every `deleted` tenant with no component rows is
  removed. Rows are visited oldest `COALESCE(clean_at, deleted_at)`
  first. What this collects is the one write that can land after
  purging (see "How a store fences" in the main document): a round that
  finds something after a clean round is that write, and the verification
  round after the retention is what makes the id safe to release. The
  retention assumes every remote client has a request timeout; the
  composition refuses one without.
- Stuck jobs: a component row in `provisioning` or `deleting` past
  `reconciler.stuck_after` is logged with its jobs' last errors, and
  again each time the age doubles. There is no failed state.
- Cost: one connection per running step, at most `concurrency` per
  reconciler process; no lock is held across a hook.

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

Every concurrent pair on one tenant or component row, and the defined
outcome. No step holds a lock across a hook; "ordered" below means the
state machine, the idempotent hooks and the store fences produce the
same end state whichever order the two land in.

| First | Concurrent | Outcome |
| --- | --- | --- |
| create, name N | create, name N | unique index: the second raises `TenantExistsError`; no store touched |
| create (id from a restored registry or a library caller) | any | primary key: `TenantExistsError`; no store touched |
| enable component C | enable C | primary key on the component row: the second raises `ComponentExistsError` |
| provision step running | disable or tenant delete request | the request sets the row `deleting` and inserts the `delete` job; the provision hook may still create a resource; the `delete` step unlinks what exists, and the `sweep` step calls `delete` again before purging, so a resource created after the `delete` hook is unlinked and purged too; a later provision step re-reads `deleting` and marks itself done |
| delete step | provision step, same row | ordered as above; the strict create raises on the `dropping` row and the provision step re-reads and marks itself done |
| disable request | disable request | idempotent: the second finds `deleting` and inserts nothing |
| update component | disable or delete request | the update on a row that is not `provisioning` or `active` raises `ComponentNotActiveError`; an update before the disable leaves a `provision` job that re-reads `deleting` and marks itself done |
| update component | update component | the later request's section and version win: the job is reset with them; a running step finishes applying the earlier section, and the reset row runs again with the later one; the component's applied version is the latest |
| rename | create with that name | unique index; the loser raises `TenantExistsError` |
| rename | delete | rename on a tenant that is not `active` raises `TenantNotActiveError` |
| any step | reconciler crash mid-step | the row stays `running` until `reclaim_after` passes, then is claimed again; hooks are idempotent; a half-done `provision` resumes from `creating` rows |
| any step | the same step reclaimed after `reclaim_after` while still running | both run; idempotent hooks and store fences make the second a wasted write; the completion of the reclaimed run records nothing |
| create request executing steps inline | api process crash | as above; a reconciler finishes; the caller's `?wait=` fails and it polls |
| claim | claim, other reconciler | the claim `UPDATE` is atomic: one wins, the other's rowcount is 0 and it moves on |
| sweep step | tombstone pass | the pass skips a component whose sweep is `pending` or `running` |
| tombstone pass | tombstone pass, other reconciler | `SKIP LOCKED` on the rows; each row is handled by one pass at a time |
| `replay` step | delete request | the delete hook removes the subsystem's per-tenant row; a `replay` step that finds no row returns `DONE`; the `deleted` transition removes the `replay` row afterward; a `replay` that had already written derived rows is purged by the sweep |
| `replay` step | `provision` step (configuration update) | `provision` writes the section and version columns only, never the watermark, so both land; the next `replay` step uses the new options |
| data operation | delete commit | store fences: in-statement for SQL stores, check-after for the rest; the operation raises `KeyNotLiveError`, and the router answers 409 while the component row is `deleting` and 404 once it is `deleted` or gone; a remote write already sent is purged by the sweep, or by the round the tombstone pass resets |
| data operation | component `provisioning` | the component's per-tenant row is written at the end of `provision`; an operation that finds none is answered 409 `component_not_active` while the row is `provisioning` and 404 `component_not_enabled` when there is no component row |
| data operation | configuration update | the operation uses the row it read at its start; the next request uses the new configuration |
| ingest | tenant delete | an ingest that committed before the event store's `delete` appended its entries; the `replay` row is reset only if it still exists; the sweep purges whatever the subsystem wrote |
| `schema upgrade` | `serve` starting | `serve` verifies and fails if the database is behind the code; a deployment orders upgrade before serve; during a rollout, expand/contract keeps both releases valid |
