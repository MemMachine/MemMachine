# Episode UUID Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give episodes and semantic history one durable sequence order, then close the remaining valid review gaps on PR #1707.

**Architecture:** The episode database atomically reserves a sequence range before the existing concurrent write fan-out. The same numbers flow to the episode table, episodic memory, and every semantic-history backend. Semantic storage supplies its own clock for registration and expiry decisions; the remaining review items are focused cleanup and documentation.

**Tech Stack:** Python 3.12+, SQLAlchemy async, SQLite, PostgreSQL, Neo4j, Pydantic, pytest, Ruff.

**Spec:** [Durable episode sequence ordering](../specs/2026-10-02-episode-sequence-ordering-design.md). Other changes answer the [October 1 review](https://github.com/MemMachine/MemMachine/pull/1707#pullrequestreview-5386761757).

## Global Constraints

- UUID remains the episode identifier; sequence numbers are server-owned and start above zero.
- A failed sequence reservation starts no episode or memory write. Successful reservations can have gaps after later failures.
- Episode storage, episodic memory, and semantic registration still write concurrently after reservation.
- Equal episode timestamps sort by sequence number before registration time or UUID.
- Registration time controls ingestion debounce and must use the semantic storage backend's clock in production.
- Existing integer-key episode databases remain outside this PR's upgrade support.

## Review Focus

- Two concurrent requests reserve disjoint sequence ranges: Task 1 test.
- A failed reservation starts no backend writes: Task 2 test.
- Equal timestamps across separate batches retain insertion order, including pagination: Task 1 and Task 3 tests.
- A missing episode still registers and follows the retry path: Task 3 test.
- API and ingestion hosts with skewed clocks do not shorten or extend the grace period: Task 4 test.

---

### Task 1: Reserve and persist episode sequence numbers

**Files:** `packages/server/src/memmachine_server/common/episode_store/{episode_storage.py,episode_sqlalchemy_store.py,count_caching_episode_storage.py}`, `packages/server/server_tests/memmachine_server/common/episode_store/{test_episode_storage.py,test_count_caching_episode_storage.py}`.

**Interfaces:** Add `EpisodeStorage.reserve_sequence_numbers(count: int) -> list[int]`. Extend `add_episodes(session_key: str, episodes: list[EpisodeEntry], *, sequence_nums: Sequence[int] | None = None) -> list[Episode]`; omitted numbers are reserved by the store. The count-cache wrapper delegates both operations.

- [ ] Write `test_reserve_sequence_numbers_concurrently`, `test_add_episodes_persists_sequence_numbers`, `test_add_episodes_rejects_invalid_sequence_numbers`, and `test_equal_timestamp_batches_paginate_in_sequence_order` in `test_episode_storage.py`. Use batches with the same timestamp and UUIDs in reverse lexical order.
- [ ] Run those tests and confirm they fail because the allocator and persisted column do not exist.
- [ ] Add a single-row counter table and a non-null, unique `BigInteger` sequence column. Initialize the counter idempotently at startup; reserve ranges with atomic `UPDATE ... RETURNING`. Update the SQL-store test fixture to call `startup()`. Do not reset the counter on delete. Validate supplied numbers, persist and hydrate them, and order list/ID queries by `(created_at, sequence_num)` even without pagination.
- [ ] Run `uv run pytest packages/server/server_tests/memmachine_server/common/episode_store/test_episode_storage.py packages/server/server_tests/memmachine_server/common/episode_store/test_count_caching_episode_storage.py` and Ruff on touched files.
- [ ] Commit this task.

### Task 2: Assign sequence numbers before concurrent dispatch

**Files:** `packages/server/src/memmachine_server/main/memmachine.py`, `packages/server/server_tests/memmachine_server/main/test_memmachine_mock.py`, test fakes implementing `EpisodeStorage`.

**Interfaces:** Consume `reserve_sequence_numbers` and pass its result to the store's `add_episodes(..., sequence_nums=...)`. Construct each in-memory `Episode` with the corresponding `sequence_num`.

- [ ] Add `test_add_episodes_dispatches_reserved_sequences` and `test_add_episodes_reservation_failure_starts_no_writes` to `test_memmachine_mock.py`. Keep its existing concurrent-write test.
- [ ] Run the new tests and confirm they fail because the main path does not reserve numbers.
- [ ] Reserve before task creation, assign numbers to episodes, pass them to the store, and update storage fakes for the new interface.
- [ ] Run `uv run pytest packages/server/server_tests/memmachine_server/main/test_memmachine_mock.py` and Ruff on touched files.
- [ ] Commit this task.

### Task 3: Replace semantic batch position with episode sequence

**Files:** `packages/server/src/memmachine_server/semantic_memory/{semantic_memory.py,semantic_session_manager.py,storage/storage_base.py,storage/sqlalchemy_pgvector_semantic.py,storage/vector_store_semantic_storage.py,storage/neo4j_semantic_storage.py,storage/alembic_pg/versions/}`, `packages/server/server_tests/memmachine_server/semantic_memory/{storage/in_memory_semantic_storage.py,storage/test_history_ordering.py,test_semantic_session_manager.py,test_semantic_memory.py,test_semantic_ingestion.py}`.

**Interfaces:** Rename semantic registration argument and stored field from `batch_position` to `sequence_num: int = 0`. For registration by existing ID, use the stored episode sequence when found; preserve the zero fallback for missing IDs. Order history by episode time, sequence, registration time, UUID.

- [ ] Add `test_history_equal_times_preserve_global_sequence` in `test_history_ordering.py` and `test_session_manager_registers_episode_sequence` in `test_semantic_session_manager.py`; route `test_ingestion_keeps_latest_value_across_uuid_ordered_batches` through `SemanticSessionManager.add_message` instead of `SemanticService.add_messages`. Keep the missing-ID retry case with a zero fallback.
- [ ] Run the changed tests and confirm they fail against `batch_position`.
- [ ] Pass `Episode.sequence_num` through the session manager and service. Replace the field and order in SQL, vector, Neo4j, and in-memory storage; add a semantic SQL migration that renames the existing history column. Update the storage contract docstring to the four-key order.
- [ ] Run `uv run pytest packages/server/server_tests/memmachine_server/semantic_memory/storage/test_history_ordering.py packages/server/server_tests/memmachine_server/semantic_memory/test_semantic_session_manager.py packages/server/server_tests/memmachine_server/semantic_memory/test_semantic_ingestion.py` and Ruff on touched files.
- [ ] Commit this task.

### Task 4: Compare registration and expiry with one storage clock

**Files:** `packages/server/src/memmachine_server/semantic_memory/{semantic_memory.py,semantic_ingestion.py,semantic_session_manager.py,storage/storage_base.py,storage/sqlalchemy_pgvector_semantic.py,storage/vector_store_semantic_storage.py,storage/neo4j_semantic_storage.py}`, semantic storage fakes and timing tests.

**Interfaces:** Add `SemanticStorage.get_storage_time() -> datetime`. PostgreSQL/SQLite read database current time; Neo4j reads `datetime()`; in-memory storage uses its own clock. Production history registration leaves `registered_at` unset so storage creates it; explicit test injection remains available.

- [ ] Add `test_missing_episode_grace_uses_storage_clock` to `test_semantic_ingestion.py` and `test_background_age_trigger_uses_storage_clock` to `test_semantic_memory_background.py` with a deliberately skewed application clock.
- [ ] Run the new tests and confirm they fail against the application-clock comparisons.
- [ ] Use storage time for `_process_single_set` expiry and `_background_ingestion_task` age cutoff. Let each backend generate production registration time rather than accepting a host timestamp. Keep episode-created time and sequence unchanged.
- [ ] Run semantic ingestion, background, session-manager, and storage timing tests; run Ruff on touched files.
- [ ] Commit this task.

### Task 5: Remove unreachable UUID guards and stale test setup

**Files:** `packages/server/src/memmachine_server/episodic_memory/long_term_memory/long_term_memory.py`, `packages/server/src/memmachine_server/semantic_memory/{cluster_splitter.py,semantic_ingestion.py}`, `packages/server/server_tests/memmachine_server/main/test_memmachine_mock.py`.

**Interfaces:** Required `Episode.uid: UUID` is used directly. Keep `SemanticService.add_messages` as a supported entry point; Task 3 exercises the production manager path separately.

- [ ] Remove `or uuid4()`, the two `m.uid is not None` filters, and the `message.uid is None` branch. Remove only the five unused `get_episodes` mock setups identified in the review; keep the assertion that the missing-row delete does not read the store.
- [ ] Run targeted long-term, cluster-splitter, semantic-ingestion, and main mock tests; run Ruff on touched files; commit this task.

### Task 6: Align PR documentation with the resulting behavior

**Files:** `docs/open_source/configuration.mdx`, PR #1707 body.

- [ ] Add `missing_episode_grace_period_sec` as numeric seconds to both semantic YAML examples and the parameter table. Keep `ingestion_trigger_age` as the existing duration key; `ingestion_trigger_age_seconds` belongs to the API update spec.
- [ ] Expand the PR compatibility section to name the episode table, semantic history, and citations. State that old integer IDs are not upgraded by this PR. Keep the partial-write caveat.
- [ ] Check the Markdown diff and PR body for accuracy; commit the repository documentation change. Do not treat the already-fixed Python client batch-delete comment as new work.

## Final verification

- [ ] Run `uv run ruff check` and `uv run ruff format --check` on touched Python files, then `uv run ty check packages`.
- [ ] Run the focused episode-store, main, semantic-history, ingestion, and episodic-memory tests. Run PostgreSQL and Neo4j integration tests when those services are available; otherwise report that limit explicitly.
- [ ] Check `git diff --check` and verify that unrelated untracked evaluation files remain untouched.
