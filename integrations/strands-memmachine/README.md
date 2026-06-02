<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>
  <h1>strands-memmachine</h1>
  <h2>MemMachine memory tool for Strands Agents</h2>
  <p>
    <a href="https://strandsagents.com/">Strands Docs</a> ◆
    <a href="https://memmachine.ai">MemMachine</a> ◆
    <a href="https://api.memmachine.ai/docs">API Docs</a> ◆
    <a href="https://strandsagents.com/latest/community/community-packages/">Community Packages</a>
  </p>
</div>

A Strands Agents extension that integrates MemMachine as a persistent memory backend, enabling AI agents to store, search, list, and delete episodic and semantic memories across sessions. It also supports full project management — create, retrieve, list, and delete isolated memory namespaces.

## What is MemMachine?

[MemMachine](https://memmachine.ai) is an AI memory platform by [MemVerge](https://memverge.com) that provides persistent, structured memory for AI agents. It enables agents to remember information across sessions, learn from past interactions, and deliver personalized, context-aware responses.

MemMachine supports two memory types:

- **Episodic memory** — conversational, event-based memories tied to interactions
- **Semantic memory** — factual, structured knowledge extracted from conversations

Memory is organized into **Projects** — isolated namespaces that keep memories separated by use case, user, or application.

## Installation

```bash
pip install strands-memmachine
```

## Configuration

| Environment Variable | Required | Description |
|---|---|---|
| `MEMMACHINE_API_KEY` | Yes | API key from [console.memmachine.ai](https://console.memmachine.ai) |

```bash
export MEMMACHINE_API_KEY=your_api_key
```

## Usage

```python
from strands import Agent
from strands_memmachine import memmachine_memory

agent = Agent(tools=[memmachine_memory])

# Store a memory
agent.tool.memmachine_memory(
    action="store",
    content="User prefers aisle seats on flights",
    metadata={"user_id": "alice", "category": "travel"},
)

# Search memories
agent.tool.memmachine_memory(
    action="search",
    query="What are the flight preferences?",
    top_k=5,
)

# List memories with filter
agent.tool.memmachine_memory(
    action="list",
    filter="metadata.user_id=alice AND metadata.category=travel",
    page_size=20,
)

# Delete a single episodic memory
agent.tool.memmachine_memory(
    action="delete",
    memory_type="episodic",
    memory_id="mem-123",
)

# Bulk delete semantic memories
agent.tool.memmachine_memory(
    action="delete",
    memory_type="semantic",
    memory_ids=["sem-1", "sem-2", "sem-3"],
)

# Create a project
agent.tool.memmachine_memory(
    action="create_project",
    project_id="my-project",
    description="Project for storing travel preferences",
)

# Get a project
agent.tool.memmachine_memory(
    action="get_project",
    project_id="my-project",
)

# List all projects
agent.tool.memmachine_memory(
    action="list_projects",
)

# Get episode count for a project
agent.tool.memmachine_memory(
    action="get_episode_count",
    project_id="my-project",
)

# Delete a project
agent.tool.memmachine_memory(
    action="delete_project",
    project_id="my-project",
)
```

## Actions

### Memory Actions

| Action | Required Parameters | Optional Parameters |
|---|---|---|
| `store` | `content` | `types`, `producer`, `produced_for`, `metadata` |
| `search` | `query` | `top_k`, `types`, `filter` |
| `list` | — | `memory_type`, `filter`, `page_size`, `page_num` |
| `delete` | `memory_type`, `memory_id` or `memory_ids` | — |

### Project Actions

| Action | Required Parameters | Optional Parameters |
|---|---|---|
| `create_project` | `project_id` | `description` |
| `delete_project` | `project_id` | — |
| `get_project` | `project_id` | — |
| `list_projects` | — | — |
| `get_episode_count` | `project_id` | — |

## Memory Types

- **Episodic** — conversational, event-based memories
- **Semantic** — factual, structured knowledge

Use the `types` parameter to target one or both:

```python
agent.tool.memmachine_memory(
    action="store",
    content="User is a software engineer",
    types=["semantic"],
)
```

## Metadata Filter Syntax

Used with `search` and `list` actions:

```
# Single condition
metadata.user_id=alice

# Multiple conditions
metadata.user_id=alice AND metadata.category=travel
```

## Parameters

### Memory Parameters

| Parameter | Type | Description |
|---|---|---|
| `action` | `str` | Operation to perform |
| `content` | `str` | Memory text. Required for `store` |
| `query` | `str` | Search query. Required for `search` |
| `memory_id` | `str` | Single UID to delete. Required for `delete` when `memory_ids` not provided |
| `memory_ids` | `list[str]` | List of UIDs for bulk delete. Required for `delete` when `memory_id` not provided |
| `memory_type` | `str` | `episodic` or `semantic`. Required for `delete`, optional for `list` |
| `types` | `list[str]` | Memory types to write to or search. Defaults to both when omitted |
| `top_k` | `int` | Max search results. Default: `10` |
| `page_size` | `int` | Results per page for `list`. Default: `100` |
| `page_num` | `int` | Zero-based page index for `list`. Default: `0` |
| `filter` | `str` | Metadata filter expression for `search` and `list` |
| `producer` | `str` | Producer identity label for `store`. Default: `user` |
| `produced_for` | `str` | Intended recipient for `store` |
| `metadata` | `dict` | Key-value pairs attached to the stored memory. All values coerced to strings |

### Project Parameters

| Parameter | Type | Description |
|---|---|---|
| `project_id` | `str` | Unique project identifier. Allowed characters: letters, numbers, underscores, hyphens, colons, and Unicode |
| `description` | `str` | Optional human-readable description of the project |

## Resources

- [MemMachine Platform](https://memmachine.ai)
- [MemMachine API Docs](https://api.memmachine.ai/docs)
- [MemVerge](https://memverge.com)
- [Strands Agents Documentation](https://strandsagents.com)
- [Strands Community Packages](https://strandsagents.com/latest/community/community-packages/)
- [Get Featured in Strands Docs](https://strandsagents.com/latest/community/get-featured/)