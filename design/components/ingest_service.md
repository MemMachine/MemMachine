# Ingest service

New component. The write path: records additions and deletions in the
event store, whose log is what memory subsystems replay, and resets the
subsystems' `replay` jobs so the log is processed.

## Constructed with

`IngestService(event_store: EventStore, tenants: TenantService,
settings: IngestSettings)`; settings: `inline`.

## API

```python
class IngestService:
    async def ingest(self, tenant_id: UUID, events: Sequence[Event]) -> IngestResult
    async def delete_events(self, tenant_id: UUID, uuids: Iterable[UUID]) -> int
    async def wait_processed(self, tenant_id: UUID, position: int,
                             timeout: timedelta) -> Mapping[str, int]
```

`IngestResult` is the event store's (`event_store.md`): stored ids,
skipped ids, and the key's head position after the write, which is what
a caller waits on and is the previous head when every event was
skipped. `delete_events` returns the head the same way.

- `ingest`: `event_store.partition(key).add_events(events)`, one
  transaction that writes the events and their `added` log entries,
  then `tenants.reset_replay` (inside that transaction where the
  engines are shared, after its commit otherwise), then the response.
  The client is acknowledged with 202 only after the reset, so an
  acknowledged batch always has a pending `replay` job; a crash between
  the commit and the reset leaves the client without a response, and
  its retry is idempotent by event id. Where the process runs the
  reconciler role, or `ingest.inline` is set, `tenants.reconcile_tenant`
  then executes the tenant's `replay` jobs through the ordinary claim,
  after the transaction has committed, before the response.
- `delete_events`: `event_store.partition(key).delete_events(uuids)`,
  which removes the rows and appends `deleted` entries, then the same
  reset; 202 with the head position.
- `wait_processed`: poll `tenants.watermarks(tenant_id)` until every
  subsystem's watermark has reached `position` or the timeout elapses;
  `?wait=` on both requests, which then respond 200, or 202 with the
  watermarks so far when the wait elapses.
- An unknown or deleted tenant raises `KeyNotLiveError` from the event
  store, which the router maps by asking the tenant service.

The subsystems see nothing on the request path: `IngestService` holds
the tenant service for the reset and the event store, and no
`MemorySubsystem` reference at all. Processing is the `replay` job's,
always.

## Changes to existing code

Replaces `MemMachine.add_episodes` and `delete_episodes`
(`main/memmachine.py:735`, `:1186`) and `server/api_v2/service.py`.
Nothing is carried over.
