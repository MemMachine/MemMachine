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
    async def delete_events(self, tenant_id: UUID, uuids: Iterable[UUID]) -> DeleteResult
    async def wait_processed(self, tenant_id: UUID, position: int,
                             timeout: timedelta) -> Mapping[str, int]

class IngestResult(BaseModel):
    stored: list[UUID]
    skipped: list[UUID]
    position: int            # the batch's last log position
```

- `ingest`: `event_store.partition(key).add_events(events)`, one transaction
  that writes the events and their `added` log entries and, where the engines
  are shared, resets every subsystem's `replay` job to pending
  (`tenants.reset_replay`); otherwise the reset follows the commit. Where the
  process runs the reconciler role, or `ingest.inline` is set, the tenant's
  `replay` jobs are executed now through the ordinary claim. The client is
  acknowledged with 202 when the events are durable.
- `delete_events`: `event_store.partition(key).delete_events(uuids)`, which
  removes the rows and appends `deleted` entries, then the same reset;
  202.
- `wait_processed`: poll each subsystem's watermark until every one has
  reached `position` or the timeout elapses; `?wait=` on both requests.
- An unknown tenant raises `KeyNotLiveError` from the event store,
  which the router maps by asking the tenant service.

The subsystems see nothing on the request path: `IngestService` holds
the tenant service for the reset and the event store, and no
`MemorySubsystem` reference at all. Processing is the `replay` job's,
always.

## Changes to existing code

Replaces `MemMachine.add_episodes` and `delete_episodes`
(`main/memmachine.py:735`, `:1186`) and `server/api_v2/service.py`.
Nothing is carried over.
