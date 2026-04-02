# MemMachine Client

A Python REST client and CLI for MemMachine.

## Features

- **Project-scoped API**: Work with MemMachine through
  `MemMachineClient -> Project -> Memory`.
- **Unified search**: Query episodic and semantic memory through one
  `memory.search()` call.
- **Metadata scoping**: Attach default metadata filters to a `Memory` instance.
- **Retrieval-agent support**: Turn on multi-hop retrieval with `agent_mode=True`.
- **CLI for terminal agents**: Use the `memmachine` command for `search` and
  `ingest`.

## Installation

```bash
pip install memmachine-client
```

From this repo:

```bash
uv pip install -e packages/client
```

## Quick Start

```python
from memmachine_client import MemMachineClient

client = MemMachineClient(base_url="http://localhost:8080")
project = client.get_or_create_project(
    org_id="my_org",
    project_id="my_project",
)

memory = project.memory(
    metadata={
        "user_id": "alice",
        "agent_id": "travel_agent",
        "session_id": "session_001",
    }
)

memory.add(
    "I prefer aisle seats on flights.",
    role="user",
    metadata={"topic": "travel"},
)

results = memory.search(
    "What are my flight preferences?",
    agent_mode=True,
)

episodes = results.content.episodic_memory.long_term_memory.episodes
print(episodes[0].content)
print(results.content.retrieval_trace)
```

## API Reference

### MemMachineClient

The main HTTP client for MemMachine.

Common methods:

- `get_or_create_project(org_id, project_id, ...)`
- `get_project(org_id, project_id)`
- `list_projects()`
- `health_check()`
- `get_metrics()`

### Project

A project is the boundary for memory operations.

Common methods:

- `memory(metadata=...)`
- `delete()`
- `refresh()`
- `get_episode_count()`

### Memory

The `Memory` object is where you ingest, search, list, and manage memory.

Common methods:

- `add(content, role="user", metadata=..., memory_types=...)`
- `search(query, limit=..., filter_dict=..., set_metadata=..., agent_mode=...)`
- `list(memory_type=..., filter_dict=..., set_metadata=...)`
- `get_context()`
- `get_current_metadata()`

## Metadata and Filtering

`Project.memory(metadata=...)` lets you attach default metadata to a memory
view. Those values become built-in filters for `search()` and `list()`.

Example:

```python
memory = project.memory(metadata={"user_id": "alice", "session_id": "session_001"})

# Searches automatically include the metadata-based filters above.
results = memory.search(
    "What did Alice say about travel?",
    filter_dict={"topic": "travel"},
)
```

If a key exists in both the instance metadata and `filter_dict`, the explicit
`filter_dict` value wins.

## Retrieval-Agent Search

Set `agent_mode=True` to enable the top-level retrieval agent:

```python
results = memory.search(
    "Where was the father of Rembrandt's wife born?",
    agent_mode=True,
)

trace = results.content.retrieval_trace
print(trace["selected_agent_name"])
```

The search response still contains the normal episodic and semantic payloads.
When the retrieval agent runs, `results.content.retrieval_trace` includes
diagnostic data about the orchestration path.

## CLI

```bash
memmachine search "<query>"
memmachine ingest --content "<text>" --role user
```

Required environment variables:

- `MEMMACHINE_URL`
- `MEMMACHINE_ORG_ID`
- `MEMMACHINE_PROJECT_ID`

Optional:

- `MEMMACHINE_API_KEY`

Examples:

```bash
memmachine search "What are Alice's flight preferences?" --limit 5 --json

memmachine ingest \
  --session-id "agent-session-20260326010101" \
  --producer-id "assistant" \
  --role user \
  --content "User asked for the latest benchmark numbers"
```

The CLI is useful for terminal workflows, scripting, and custom agent
integrations.

## Shared Skill Helpers

Provider-native skill primitives now live in `memmachine-common` and are shared
by the client and server packages:

```python
from memmachine_common import Skill, SkillRunner, install_skill
```

These helpers are used by the retrieval-agent runtime and by evaluation
harnesses that install markdown skill bundles into provider Files APIs.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the same license as MemMachine.
