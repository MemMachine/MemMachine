# Exclusive SQL Lease Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide one exclusive SQL lease per key for distributed tasks that outlive a database transaction.

**Architecture:** A row per held key stores its lease and token; successful release deletes it. A durable singleton counter issues monotonic tokens. Short PostgreSQL row-lock or SQLite immediate transactions arbitrate changes. The public service polls for acquisition and renews a context-managed lease.

**Tech Stack:** Python 3.12+, SQLAlchemy async, SQLite, PostgreSQL, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-30-exclusive-sql-lease-lock-design.md`

**Revision:** After the original implementation, the lock now uses a durable
global counter and deletes key rows on successful release. See the updated
spec and tests for the final storage contract.

## Global Constraints

- Expose only exclusive `try_acquire`, `acquire`, and `lock` operations; remove read/write methods and `Lease.mode`.
- Keep the `ResourceManager.get_sql_lock_service()` accessor.
- Use database time for expiry decisions and preserve per-key fencing token monotonicity across release and expiry.
- Keep transactions short; no transaction spans protected task work.
- Keep cancellation cleanup and preserve unrelated cancellation requests.
- No migration from the unpublished two-table schema is required.
- A key has at most one live holder. Acquisition is not FIFO.

## Review Focus

- Empty keys and nonpositive lease durations must raise before creating a row (Task 2).
- A simultaneous first acquisition on a missing key must grant exactly one holder (Task 1).
- A stale handle after expiry and replacement must not renew or release the replacement (Task 1).
- Cancellation after a SQL grant must release that grant (Task 2).
- An outside cancellation concurrent with renewal failure must remain pending after the lock translates its own cancellation (Task 3).

---

### Task 1: Single-row exclusive store

**Files:**
- Modify: `packages/server/src/memmachine_server/common/sql_lease_lock/_store.py`
- Test: `packages/server/server_tests/memmachine_server/common/sql_lease_lock/test_service.py`
- Test: `packages/server/server_tests/memmachine_server/common/sql_lease_lock/test_postgres.py`

**Interfaces:**
- Produces: `SQLLeaseStore.startup()`, `try_acquire(key: str, duration_ms: int) -> StoredLease | None`, `renew(key: str, lease_id: str, duration_ms: int) -> int | None`, and `release(key: str, lease_id: str) -> bool`.
- Produces: `StoredLease(key: str, lease_id: str, expires_at_ms: int, fencing_token: int)`.

- [ ] **Step 1: Rewrite store tests to fail against the old API.** Test two independent engines contesting a new key, distinct keys, release/reacquire, expiry/replacement, stale renew/release, monotonic tokens, and a SQL failure that propagates rather than appearing as contention. Update PostgreSQL contention tests for exclusive grants.
- [ ] **Step 2: Run the store tests.** `uv run pytest packages/server/server_tests/memmachine_server/common/sql_lease_lock/test_service.py -k 'store or simultaneous or expired or database_error' -q`; expect API or assertion failures. Run PostgreSQL tests only when the configured test database is available.
- [ ] **Step 3: Implement the store.** Replace both tables with `lease_lock(key, generation, lease_id, expires_at_ms)`; use a nullable holder and expiry, keyword `StoredLease` construction, row locking on PostgreSQL, and the existing SQLite `BEGIN IMMEDIATE` path. Delete `LockMode` and all holder-table logic. Clear the holder on release while retaining generation.
- [ ] **Step 4: Verify the store.** Run the focused SQLite store tests and PostgreSQL tests when available; expect all selected tests to pass.
- [ ] **Step 5: Commit.** Stage only `_store.py` and its store tests; commit `refactor: use one row for exclusive SQL leases`.

### Task 2: Exclusive public API and resource manager

**Files:**
- Modify: `packages/server/src/memmachine_server/common/sql_lease_lock/service.py`
- Modify: `packages/server/src/memmachine_server/common/sql_lease_lock/__init__.py`
- Modify: `packages/server/src/memmachine_server/common/resource_manager/__init__.py`
- Modify: `packages/server/src/memmachine_server/common/resource_manager/resource_manager.py`
- Test: `packages/server/server_tests/memmachine_server/common/sql_lease_lock/test_service.py`
- Test: `packages/server/server_tests/memmachine_server/common/resource_manager/test_resource_manager.py`

**Interfaces:**
- Consumes: Task 1's `SQLLeaseStore` and `StoredLease`.
- Produces: `SQLLeaseLockService.try_acquire(key: str, *, lease_duration: timedelta) -> Lease | None`, `acquire(key: str, *, lease_duration: timedelta, wait_timeout: timedelta | None = None) -> Lease`, and `lock(key: str, *, lease_duration: timedelta, wait_timeout: timedelta | None = None) -> AbstractAsyncContextManager[Lease]`.
- Produces: `ResourceManager.get_sql_lock_service() -> SQLLeaseLockService`.

- [ ] **Step 1: Update public tests to the exclusive API.** Assert empty-key and duration validation occurs before row creation, timeout and waiting behavior, independent keys, renewal and release, cancellation after a SQL grant, and that the resource manager returns a shared service backed by the session database.
- [ ] **Step 2: Run public tests.** `uv run pytest packages/server/server_tests/memmachine_server/common/sql_lease_lock/test_service.py packages/server/server_tests/memmachine_server/common/resource_manager/test_resource_manager.py -q`; expect failures from old names and signatures.
- [ ] **Step 3: Implement API and wiring.** Rename the service and store imports, remove `Lease.mode` and read/write methods, add the three exclusive methods, adapt acquisition calls to Task 1, and update public exports and resource manager annotations. Document distributed-task intent, non-FIFO contention, persistent per-key rows, and the need for consumers to check fencing tokens at external side effects.
- [ ] **Step 4: Verify public tests.** Re-run the two files above; expect pass.
- [ ] **Step 5: Commit.** Stage Task 2 files and tests; commit `refactor: expose exclusive SQL lease lock API`.

### Task 3: Renewal and cancellation correctness

**Files:**
- Modify: `packages/server/src/memmachine_server/common/sql_lease_lock/service.py`
- Test: `packages/server/server_tests/memmachine_server/common/sql_lease_lock/test_service.py`

**Interfaces:**
- Consumes: Task 2's `SQLLeaseLockService.lock(...)` and `Lease.renew()` / `release()`.
- Produces: context exit that always attempts release after an in-flight renewal failure, converts only its own cancellation to a renewal error, and preserves body exceptions.

- [ ] **Step 1: Add deterministic failing tests.** Block a renewal until the body has exited, then fail it; assert the context raises the renewal error and releases the lease. After a renewal failure during the body, assert the task's cancellation count is restored. Add a concurrent outside-cancellation case that asserts its request is preserved.
- [ ] **Step 2: Run those tests.** `uv run pytest packages/server/server_tests/memmachine_server/common/sql_lease_lock/test_service.py -k 'renewal_failure or cancellation_state' -q`; expect failures.
- [ ] **Step 3: Fix cancellation accounting.** After failed renew, do not cancel the owner if stop was already signalled. When translating a cancellation issued by the renewal task, remove exactly that request via `Task.uncancel()`. Ensure the context awaits renewal and attempts release before reporting the renewal error; keep body-error precedence.
- [ ] **Step 4: Verify context tests.** Run the focused file; expect pass.
- [ ] **Step 5: Commit.** Stage service and context tests; commit `fix: clean up SQL leases after renewal failure`.

### Task 4: Final verification

**Files:** All files changed in Tasks 1–3.

**Interfaces:** No new API.

- [ ] **Step 1: Run focused tests.** `uv run pytest packages/server/server_tests/memmachine_server/common/sql_lease_lock packages/server/server_tests/memmachine_server/common/resource_manager/test_resource_manager.py -q`; expect pass or report unavailable PostgreSQL infrastructure separately.
- [ ] **Step 2: Run style checks.** `uv run ruff check` on changed Python files and `uv run ruff format --check` on changed Python files; expect pass.
- [ ] **Step 3: Review diff and workspace.** Run `git diff --check`, `git status --short`, and inspect the branch diff against `main` for remaining read/write API references or old table names.
