# MemMachine Core

The core memory engine for MemMachine: episodic and semantic memory systems, their
abstractions, data models, and provider/storage implementations.

## Installation

```bash
pip install memmachine-core
```

## What this package provides

`memmachine-core` is the reusable memory library underpinning MemMachine. It contains:

- **Memory systems** — episodic memory (long-term and short-term) and semantic memory.
- **Abstractions** — the ABCs and Protocols for embedders, language models, rerankers,
  vector stores, graph stores, episode storage, and segment stores.
- **Data models** — the shared event, episode, segment, derivative, and semantic types.
- **Provider/storage implementations** — concrete embedders, language models, rerankers,
  and vector/graph/SQL backends.

It does **not** include the resource manager, configuration loading, or the HTTP/MCP
server — those live in `memmachine-server`, which depends on this package.

## Development

For local development inside the MemMachine monorepo:

```bash
pip install -e packages/core
```

## Source

- Repository: <https://github.com/MemMachine/MemMachine>
- Issues: <https://github.com/MemMachine/MemMachine/issues>
