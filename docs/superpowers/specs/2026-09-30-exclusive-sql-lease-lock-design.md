# Exclusive SQL lease lock for distributed tasks

## Intent

Provide one exclusive, SQL coordinated lease per resource key so tasks running
on different server instances can synchronize work that outlives a database
transaction. The service is a general coordination primitive; this change does
not wire it into `EpisodicMemoryManager` or another consumer.

## Public contract

- Rename the public service to `SQLLeaseLockService` and the store to
  `SQLLeaseStore`. Keep `ResourceManager.get_sql_lock_service()` as its accessor.
- Expose `try_acquire(key, *, lease_duration) -> Lease | None`,
  `acquire(key, *, lease_duration, wait_timeout=None) -> Lease`, and
  `lock(key, *, lease_duration, wait_timeout=None)` as an async context manager.
  Remove read/write methods and the `Lease.mode` property; no production caller
  uses them today.
- A key has at most one live lease. Different keys do not block each other as
  held locks. New grants briefly serialize through the global token counter
  on PostgreSQL; SQLite's write transaction already serializes attempts.
- A waiting caller polls with the existing bounded jitter and may time out or
  be cancelled. Acquisition order is not FIFO; the service makes no fairness
  guarantee among waiters.
- `Lease` retains its ID, key, expiry, fencing token, `renew()`, and `release()`.
  The context manager renews while the task body runs and releases on exit.
  A failed renewal interrupts the body with `LeaseLostError` or propagates the
  underlying database error. Callers must stop protected work when that occurs.
- Fencing tokens increase on every grant for the same key, including grants
  after expiry. A token only fences external side effects if that consumer
  checks or persists it at the side effect boundary.

## Storage and lifecycle

Replace the resource and holder tables with a `lease_lock` table containing
`key` (primary key), `fencing_token` (non-null integer), `lease_id` (nullable),
and `expires_at_ms` (nullable). A singleton `lease_lock_counter` row holds the
last issued token. Successful release deletes the key row; reacquiring the key
receives a higher token from the durable counter. The shared counter briefly
serializes grants across distinct keys on PostgreSQL. Expired rows whose
workers never return remain until that key is reacquired; workloads with
abandoned one-time keys need periodic expiry cleanup.

Each acquire runs in a short transaction. It inserts the key if absent, locks
its row (`FOR UPDATE` on PostgreSQL; `BEGIN IMMEDIATE` on SQLite), and reads
database time. A live `lease_id` with future expiry denies acquisition. An
empty or expired row grants a fresh UUID lease ID, atomically increments the
global counter, and sets the expiry and token. Renewal and release lock the
same row and succeed only if the ID still matches and the lease has not
expired. Release deletes the row. Renewal and release never insert a missing
key. No SQL transaction remains open while the caller runs its task.

The current branch has no production users of the tables and has not landed
on `main`. This proposal replaces its schema in place; it does not migrate data
from the unpublished two-table schema.

## Cancellation and errors

Preserve the current grant cleanup when acquisition is cancelled after SQL
has granted a lease. Fix the context cleanup race: once exit has signalled the
renewal loop to stop, an in-flight renewal failure must not cancel the owner.
The context must await the renewal task, attempt release, and then report the
renewal failure. When the renewal loop cancels the owner during the body and
the context converts that cancellation into a renewal error, it must remove
exactly its own pending cancellation request, leaving unrelated cancellation
requests intact. Body exceptions retain precedence over release failures.

## Verification

- SQLite tests with two independent engines: simultaneous acquisition of one
  key grants only one lease; distinct keys are independent; release and expiry
  allow a new grant with a larger token; stale handles cannot renew or release
  the replacement.
- Public API tests: key and duration validation, wait timeout, cancellation
  during acquisition, renewal during a context, and release after exit.
- Deterministic regression tests for renewal failure while exit awaits an
  in-flight renewal, and for cancellation state after converting a renewal
  failure to `LeaseLostError`.
- PostgreSQL contention tests for the same exclusive behavior.
- Run focused pytest, Ruff checks, and format checks for touched Python files.
