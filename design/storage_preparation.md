# Storage preparation

Every store's durable resources (SQL tables, native vector collections and
their indexes, graph schemas) are created by one setup step,
`memmachine-server prepare`, run before servers start. A serving process
creates none of them. This document records how that step is built and what
it covers.

## The setup step

`memmachine-server prepare` loads the configuration the way the server does,
calls `prepare` on every store the configuration names, and exits. It is
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

The operator runs it before servers first start and after every configuration
change that names a store or embedder no earlier run covered. Such changes
are made in the configuration file with servers stopped, never through a
running server's configuration API, and an embedder's model never changes
under an existing name.

## The store contract

Every storage interface extends `Preparable` and `Lifecycle`
(`common/lifecycle.py`), whose docstrings state the contract:

- `prepare` runs in the setup step. On a store with none of its durable
  resources, it creates what the store needs; run again on what it created, it
  changes nothing. It promises nothing for any other state.
- Its owner calls `start` once before use and `stop` once after. Neither
  creates, changes, or deletes anything durable, so any number of processes
  may start stores over the same resources at once.

Some stores' creation code does more than `prepare` promises, such as
converting session tables and Neo4j semantic memory data of an older shape or
upgrading pgvector semantic storage's tables, but nothing relies on that, and a
database an earlier release created is recreated. What `start` does beyond
reading is in process: loading a Milvus collection into memory, or replaying a
SQLite vector store's pending operations into its search engines.

## What `prepare` covers

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
