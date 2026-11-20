# Schema migration

Every store's durable resources (SQL tables, native vector collections and
their indexes, graph schemas) are created by one setup step,
`memmachine-server migrate`, run before servers start. A serving process
creates none of them. This document records how that step is built and what
it covers.

## The setup step

`memmachine-server migrate` loads the configuration the way the server does,
calls `migrate()` on every store the configuration names, and exits. It is
the same executable as the server and runs from the same image, so the setup
step and the servers share one code version and one configuration.

The setup step relies on assumptions a serving process cannot make:

- **One process runs it at a time.** Nothing arbitrates between concurrent
  runs; compose runs it as a one-shot service and Helm as an init container
  of a single-replica deployment.
- **No server is running.** Creating schema never races a server's reads
  and writes of the same database.
- **It may hold privileges servers do not need**, such as creating a
  PostgreSQL extension.

The operator runs it after every upgrade and after every configuration change
that names a store or embedder no earlier run covered, and never changes an
embedder's model under an existing name.

## The store contract

Every storage interface has `migrate()`: it creates the store's durable
resources or brings them up to date, runs before `startup`, while no other
process uses the store, and changes nothing when run again. It runs the
creation the store's startup used to run, so it adopts a database an earlier
release created. `startup()` creates nothing: it opens the store on what
`migrate` created, and does what serving needs beyond schema, such as
loading a Milvus collection or replaying a SQLite vector store's pending
operations.

## What migrate covers

Every store a server can open under the configuration:

- the session data manager and SQL lease lock on the session database, and
  the episode store;
- event memory's segment store, vector store, and partition registry, the
  deployment's, when it configures event memory;
- every NebulaGraph graph store (a Neo4j graph store creates its indexes as
  its labels grow);
- semantic memory's storage and config store, each when configured, whether
  or not semantic memory is enabled; for the vector store backend, that is
  its vector store, its partition registry, and its one partition.

Some creation stays on the request path, as data-plane work rather than
schema: a SQLite vector store's tables for each partition, Neo4j and
NebulaGraph indexes and types for each session, and Neo4j semantic memory's
index for each set.
