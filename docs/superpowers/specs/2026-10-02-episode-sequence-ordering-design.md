# Durable episode sequence ordering

## Intent

Give every episode a database-reserved, increasing `sequence_num` before
episode storage, episodic memory, and semantic history start their concurrent
writes. Use that one number to break ties between equal episode timestamps in
both episode-store reads and semantic ingestion. The sequence replaces the
existing per-batch `batch_position` in semantic history.

## Sequence allocation

The episode database owns a single-row counter table. Reserving a batch of
`n` episodes atomically increments the counter by `n` and returns the new
value. The caller assigns the contiguous range in input order. PostgreSQL
serializes updates to the counter row; SQLite serializes writers. The
reservation transaction commits before the three episode writes fan out, so
all backends receive the same numbers without waiting for the episode insert.

The counter row is initialized idempotently at episode-store startup. Direct
episode-store callers reserve a range inside `add_episodes` if they have not
supplied numbers. The store rejects mismatched lengths, duplicate numbers, or
nonpositive numbers. Failed writes may leave gaps; numbers must never be
reused. Deleting episodes, including `delete_all`, does not reset the counter.

## Episode data and ordering

`Episode.sequence_num` remains the shared typed field. The SQL episode table
stores it as a non-null, unique 64-bit integer and hydrates it on reads. The
main `add_episodes` path reserves numbers first, assigns them to the in-memory
episodes, and passes the same numbers to the episode-store insert. API clients
cannot choose them. Existing UUIDs remain the episode identifiers.

Episode-store list and ID queries order by `(created_at, sequence_num)` for
both paged and unpaged reads. Single-ID lookups do not need sorting. Batch
lookups by an arbitrary set of UUIDs retain their current unordered contract.
Short-term and long-term memory receive the assigned sequence through the
existing `Episode` model.

## Semantic history

Replace `batch_position` with `sequence_num` in the semantic storage
interface, SQL history tables, Neo4j nodes, and in-memory test storage.
Production registration passes `Episode.sequence_num` through the session
manager; it does not wait for the episode insert. Paths that register existing
IDs read their stored sequence when available. Missing episode IDs retain a
defined fallback sequence of zero, allowing the existing ingestion retry path
to operate.

History reads order by episode time, sequence number, registration time, then
UUID, so clocks on different API hosts cannot override the reserved order.
Registration time remains the ingestion debounce clock. A new semantic
SQL migration renames the history column; the new episode-store column and
counter table follow this PR's existing no-upgrade policy for old episode
databases. The vector and Neo4j implementations use the same field name and
ordering contract.

## Verification

Tests cover disjoint range reservations from concurrent callers, input-order
assignment, round-trip persistence, equal-timestamp episodes across separate
batches, stable pagination, and semantic ingestion order through the production
session-manager path. Shared semantic storage tests cover SQL, vector, Neo4j,
and the in-memory implementation. Existing concurrent-write and missing-row
retry tests continue to pass.

The reservation adds one episode-database round trip before fan-out. If that
reservation fails, no backend write starts. The existing non-atomic behavior
after a successful reservation remains outside this change.
