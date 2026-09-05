# Server and settings

New component. The `Server` object the composition returns, the roles,
the routers, error mapping, and the settings models.

## Server

```python
class Server:
    def __init__(self, tenants: TenantService, ingest: IngestService,
                 subsystems: Sequence[MemorySubsystem], settings: ServerSettings)
    async def start(self) -> None     # verify schema, check scope, start roles
    async def stop(self) -> None
```

- `start`: `schema verify` for every SQL component; the scope check
  (the minimum over every resource's `concurrency_scope` reaches
  `settings.concurrency_scope`); then, per `settings.roles`, bind the
  routers (`api`) and start `tenants.run_reconciler()` (`reconciler`).
- `stop`: routers stop accepting; the reconciler finishes its current
  step; resources close in reverse order.

## Routers

- `/v1/tenants`: `TenantService`.
- `/v1/tenants/{id}/events`: `IngestService` and `EventStore`.
- `/v1/tenants/{id}/episodic-memory`: `EpisodicMemoryManager`.

One exception handler maps the `MemMachineError` hierarchy:
`TenantNotFoundError` 404, `TenantNotActiveError` 409,
`TenantExistsError` 409, `InvalidTenantConfigError`,
`InvalidPropertyKeyError`, `InvalidPropertyValueError`,
`UndeclaredPropertyKeyError` 422, `ProviderUnavailableError` 503,
anything else 500 `internal` with the traceback logged and never
returned. A `KeyNotLiveError` or an unknown tenant on the data path is
resolved by `tenants.state_of(id)`: no row or `deleted` is 404, else
409.

## Settings

One model per resource class, nested into `ServerSettings` for the
standard composition; read from environment variables and an optional
YAML or TOML file of the same shape; `memmachine settings schema` and
`memmachine settings example` are generated from them.

- `ServerSettings`: `bind`, `roles`, `concurrency_scope`,
  `request_timeout`, `databases`, `vector_store`, `embedders`,
  `rerankers`, `language_models`, `event_store`, `segment_store`,
  `episodic_memory`, `tenants`, `tenant_templates`.
- Slot settings are discriminated unions over registered kinds, keyed
  by `kind`: `DatabaseSettings` (`postgres`, `sqlite`, with the SQLite
  pragmas as fields), `VectorStoreSettings` (`qdrant`, `milvus`,
  `pgvector`, `sqlite_vec`, `usearch`, ...), `EmbedderSettings`,
  `RerankerSettings`, `LanguageModelSettings`.
- Every client settings model has a required `request_timeout`.

## Adding a service type

A provider family is an ABC (`Embedder`, `Reranker`, `LanguageModel`,
`Database` engines), a settings union over its kinds, a kind table, and
an entry-point group `memmachine.<family>`. Adding a family is adding
those four and a slot in the standard composition; adding a kind to a
family is registering one callable and one settings model. Language
models stay in the design as a family with the current
`common/language_model/` implementations behind it, wired to nothing
until a component takes one; the family exists so that wiring one later
is a constructor parameter, not a redesign.

Context part kinds and block kinds are families of the same shape (a
Pydantic model per kind, a kind table, the `memmachine.context_parts`
and `memmachine.block_kinds` entry-point groups), without a slot in the
composition, since parts and blocks are data, not resources.

## Composition

`compose(settings: ServerSettings) -> Server` in
`memmachine_server.composition`; `memmachine serve` runs it and
`--compose module:function` runs another. Kinds register by import for
built-ins and through `importlib.metadata` entry points
(`memmachine.<family>`) for out-of-tree implementations.

## Changes to existing code

Replaces `server/app.py`, `server/api_v2/` (router, config router,
service, exceptions, mcp), `common/configuration/`,
`common/resource_manager/`, `main/memmachine.py`, `installation/`, and
`memmachine_common/api/spec.py`. The engine construction and
`enable_sqlite_foreign_keys` in
`common/resource_manager/database_manager.py:51` become the `postgres`
and `sqlite` kinds. Nothing else is carried over.
