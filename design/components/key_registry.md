# Key registry

New component. Per-key bookkeeping for stores whose data is not in a SQL
database, written once and shared as code and as a database, never as
rows. A store receives only a view scoped to its own name.

## Constructed with

- `SqlKeyRegistry(engine: AsyncEngine)`.
- A store receives `registry.scoped(scope: str) -> KeyRegistry`, produced
  by the composition; the registry raises at the second `scoped` call
  with the same scope in one process, and `scope` is the store's name.

## Storage

`key_registry`:

| column | type | note |
| --- | --- | --- |
| `scope` | `TEXT` | the store's name |
| `key` | `UUID` | the tenant id |
| `state` | enum | `creating`, `live`, `dropping` |
| `address` | `JSON` | what the store needs beyond the key: container, backend-assigned id |
| `created_at`, `updated_at` | timestamps | |

Primary key `(scope, key)`. A row exists before any record can carry the
key, and goes when the store's purge finds nothing under it.

## API of the scoped view

```python
class KeyRegistry(Protocol):
    async def create(self, key: UUID, address: Mapping, *,
                     state: KeyState = KeyState.LIVE) -> None
    async def get(self, key: UUID) -> KeyRow | None
    async def set_state(self, key: UUID, state: KeyState) -> None
    async def set_address(self, key: UUID, address: Mapping) -> None
    async def remove(self, key: UUID) -> None
    async def count_by_address(self, field: str) -> Mapping[str, Mapping[KeyState, int]]
```

- `create` is strict: any existing row under the key, in any state,
  raises `KeyExistsError`.
- Every statement the view issues carries `scope = <its scope>`; there
  is no method that names another scope.
- `count_by_address` serves `memmachine schema status` (rows per
  container by state) and container retirement.

## Fencing, as used by a store

- Before a remote write or read: `get(key)`; no row or a row not `live`
  raises `KeyNotLiveError`; the address tells the store where to go.
- After the remote operation: `get(key)` again; not `live` raises
  `KeyNotLiveError`. A write already sent is then garbage under a
  `dropping` key, purged by the store's `purge`.
- Logical delete: `set_state(key, DROPPING)`; waits for nothing.
- No lock is held across the remote operation, and no clock is read.

## Concurrency scope

`cluster` on PostgreSQL; `host` on a SQLite file; `process` on in-memory
SQLite. A store's scope is the minimum of its backend's and its
registry's.

## Changes to existing code

Replaces the registries the vector stores keep inside their backends:
the `{namespace}__registry` collection on Qdrant
(`common/vector_store/qdrant_vector_store.py:609`), the
`memmachine_{namespace}__registry` collection on Milvus
(`milvus_vector_store.py:477`), and the `vector_store_sqlite_vec_cl` and
`vector_store_sqlite_cl` tables in the SQLite stores. Nothing is carried
over.

## Schema

`key_registry`:

| column | type | constraint |
| --- | --- | --- |
| `scope` | `String(64)` | primary key part; check matches `[a-z0-9_-]+` |
| `key` | `Uuid` | primary key part |
| `state` | `String(16)` | not null; check in (`creating`, `live`, `dropping`) |
| `address` | `JSON` (`JSONB` on PostgreSQL) | not null |
| `created_at` | `DateTime(timezone=True)` | not null, `func.now()` |
| `updated_at` | `DateTime(timezone=True)` | not null, `func.now()`, updated on every write |

Indexes: `key_registry__scope_state (scope, state)` for the sweep's
"rows in `dropping`" and for `count_by_address`, which additionally
reads `address ->> 'container'` (a JSONB expression index on
PostgreSQL where a deployment has many keys per store). The primary key
serves every per-key read.

## Contract

`KeyRegistry` is an ABC; `SqlKeyRegistry.scoped(scope)` returns its one
implementation, `ScopedKeyRegistry`, and no other class implements it.
