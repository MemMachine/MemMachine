# Schema migration

Every store's durable resources (SQL tables, native vector collections and
their indexes, graph schemas) are created and evolved by one setup step,
`memmachine-server migrate`, run before servers start; a serving process only
checks them. This document records how that step is built, what it covers,
what a server checks, and how schema changes will be carried by it.

## The setup step

`memmachine-server migrate` loads the configuration the way the server does,
calls `migrate()` on every store the configuration can select, and exits. It
is the same executable as the server and runs from the same image, so the
setup step and the servers share one code version and one configuration.

The setup step relies on assumptions a serving process cannot make:

- **One process runs it at a time.** Nothing arbitrates between concurrent
  runs; compose runs it as a one-shot service and Helm as an init container
  of a single-replica deployment.
- **No server is running.** A schema change never runs beside code that
  expects the previous schema.
- **It may hold privileges servers do not need**, such as creating a
  PostgreSQL extension.

The operator runs it after every upgrade and after every configuration change
that names a store or embedder no earlier run covered, and never changes an
embedder's model under an existing name.

## The store contract

Every storage interface has `migrate()`: it creates the store's durable
resources or brings them up to date, runs before `startup`, while no other
process uses the store, and changes nothing when run again.

`startup()` opens the store on what `migrate` created and never creates
anything. It raises `SchemaMismatchError`, naming `memmachine-server migrate`,
when that is missing or behind:

- a SQL store compares its lineage's version table with the newest revision
  this code has, and refuses a missing, older, or unknown revision;
- a Qdrant or Milvus store checks that its native collection exists with its
  dimensions and metric (and Milvus loads it);
- a semantic store checks that its tables, or Neo4j constraints, exist.

So an old server refuses a database a newer release migrated, and a new
server refuses one not yet migrated. A server never runs beside a schema
change it was not written for.

## What migrate covers

Everything a server or a project can select under the configuration:

- the session data manager and SQL lease lock on the session database, and
  the episode store;
- the segment store on every configured relational database, and the event
  backend's store for every configured vector store and embedder, because a
  project can name any of them;
- every NebulaGraph graph store (a Neo4j graph store creates its indexes as
  its labels grow);
- semantic memory, when enabled.

Selecting among configured resources therefore never needs schema at
request time. The cost is storage for combinations no project uses: one
vector collection per configured vector store and embedder.

Some creation stays on the request path, as data-plane work rather than
schema: a SQLite vector store's tables for each partition, Neo4j and
NebulaGraph indexes and types for each session, and Neo4j semantic memory's
index for each set.

## SQL: one Alembic lineage per store

Each SQL store owns a lineage: the Alembic revisions that create and evolve
its tables, declared beside the store with the store's table metadata. A
lineage records its revision in a version table of its own,
`alembic_version_<lineage>`.

- **Why one version table per lineage.** The configuration places each store
  on a database of its own choosing, so one database holds the tables of
  several stores and another holds a different set. With separate version
  tables, upgrading one store's tables never creates or touches another's,
  and a revision this code does not know is attributed to its lineage: the
  database is newer than the code, not unprovisioned. One shared table with
  a branch per store would lose that attribution.
- **One environment.** A single Alembic environment serves every lineage;
  the lineage selects the revision directory, the version table, and the
  tables autogenerate compares.
- **One transaction per lineage.** A revision that fails leaves the tables as
  they were; on SQLite the runner opens the transaction explicitly, since the
  driver otherwise commits each schema change on its own.
- **First revisions start from nothing.** A database whose tables were created
  some other way is not adopted: its first upgrade fails on the existing
  tables, and the database is recreated.

Semantic memory keeps its own mechanisms, since it is expected to be
replaced: pgvector storage runs its existing Alembic chain, and the other
semantic stores create their tables directly.

## Carrying schema changes

### Upgrade procedure

1. Stop the servers.
2. Back up the databases.
3. Run `memmachine-server migrate` from the new release.
4. Start the servers.

### SQL, including EventMemory's segment store

- A change to a store's tables is a new revision in its lineage, generated
  with `alembic -n <lineage> -x url=<database URL> revision --autogenerate`
  against a database upgraded to the previous revision, then reviewed.
- Revisions spell out their operations and never import the store's models,
  which describe only the newest revision.
- A data change belongs in the revision that needs it.
- SQLite revisions alter tables in batch mode, which recreates the table.
  Recreating a table other tables reference needs foreign keys off for the
  transaction and `foreign_key_check` after it.
- Revisions run inside the runner's transaction, so an operation PostgreSQL
  refuses in a transaction, such as `CREATE INDEX CONCURRENTLY`, runs in an
  `autocommit_block`.
- A test upgrades every lineage on empty SQLite and PostgreSQL databases and
  compares the result with the store's models, so a model changed without a
  revision fails it.
- Until a release ships a lineage, a change to it replaces its first
  revision under a new revision id instead of adding one. A development
  database built from the replaced revision then reports a revision the code
  does not know, instead of drifting unnoticed.

### Vector stores

A vector store's schema is a small declaration: dimensions, metric, and
declared properties, and each backend reports its collection's live schema.
So vector stores reconcile instead of keeping versioned revisions:
`migrate()` compares the live collection with the declaration.

- **A key added to the declaration** is applied in place; records written
  before lack the key, which a declared property allows.
  - Qdrant: a payload index.
  - Milvus: a nullable field and its index.
  - SQLite: a column and index on each partition's tables, found through the
    catalog.
- **Any other change** (dimensions, metric, a key's type, a removed key) is a
  rebuild into a new store, then a switch to it; Qdrant and Milvus aliases
  can make the switch atomic.

Versioned revisions calling vector clients are possible, since the registry's
database is SQL, but the two would not commit together, so every such
revision would have to be an idempotent reconciliation anyway.

### EventMemory data changes

A change of embedder, deriver, or segmenter, or a vector store change that is
not additive, rebuilds the vector side from the segment store: re-derive each
partition's derivatives from its segments, embed them, and write the vector
records and derivative links into the new store, partition by partition.

### Semantic memory, short-term memory, Neo4j, and NebulaGraph

No migration plan: semantic memory is expected to be replaced and
short-term memory removed. A replacement starts with lineages of its own.

### Rolling upgrades

Upgrades stop every server. Running two versions side by side needs:

- revisions tagged expand (additive, readable by the previous release) or
  contract;
- a server check that accepts a database ahead of its code only by expand
  revisions;
- `migrate` refusing a contract revision unless forced;
- online DDL on PostgreSQL;
- every schema change split into expand, a release that reads and writes both
  shapes, and contract, with the previous release tested against the new
  schema.
