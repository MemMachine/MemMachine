
# OpenAI Agents SDK + MemMachine (Python) Example

This example shows how to integrate **MemMachine** as an external memory store for an agent built with the **OpenAI Agents SDK** (the Python package that provides the `agents` module).

The agent is configured with two tools:

- `add_memory(memory: str)` — persists a memory string to MemMachine
- `search_memory(query: str)` — queries MemMachine and returns a simplified list of relevant memories

The implementation is in `example.py`.

## What this demonstrates

- Creating (or reusing) a MemMachine **Project** via the Python SDK
- Writing “explicit memory” items via `memory.add(...)`
- Retrieving context via `memory.search(...)` and formatting results for the agent
- Using an OpenAI-compatible chat model endpoint (this example uses **Qwen** via DashScope’s OpenAI-compatible API)

## Prerequisites

1. **Python 3.12+**
2. A running **MemMachine server** (local Docker is recommended)
3. An API key for an OpenAI-compatible LLM endpoint

## Start MemMachine

From the repository root, the quickest way is:

```bash
./memmachine-compose.sh
```

Then verify it is up:

```bash
curl -s http://localhost:8080/health
```

## Install dependencies

This example depends on:

- `memmachine` (this repo)
- `openai` (Async client used for OpenAI-compatible endpoints)
- the **OpenAI Agents SDK** (the package that provides the `agents` module)

If you are running from this repo checkout, you can typically install the repo plus the extra agent dependencies like this:

```bash
pip install -e .
pip install openai-agents
```

Notes:

- The exact pip name for the OpenAI Agents SDK may differ depending on your environment. If `pip install openai-agents` does not provide the `agents` module, install the package recommended by the Agents SDK documentation.

## Configuration

### MemMachine environment variables

`example.py` reads the following variables (all optional):

| Variable | Purpose | Default |
|---|---|---|
| `MEMMACHINE_BASE_URL` | MemMachine server base URL | `http://localhost:8080` |
| `MEMMACHINE_API_KEY` | MemMachine API key (if enabled on your server) | empty |
| `MEMMACHINE_ORG_ID` | Organization ID (API v2 scoping) | `default_org` |
| `MEMMACHINE_PROJECT_ID` | Project ID used for this demo | `openai_agent_demo` |

Example:

```bash
export MEMMACHINE_BASE_URL="http://localhost:8080"
export MEMMACHINE_ORG_ID="default_org"
export MEMMACHINE_PROJECT_ID="openai_agent_demo"
```

### LLM environment variables (Qwen / DashScope)

This example uses Qwen via an OpenAI-compatible endpoint using `openai.AsyncOpenAI(base_url=..., api_key=...)`.

| Variable | Purpose | Default |
|---|---|---|
| `QWEN_BASE_URL` | OpenAI-compatible base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `QWEN_API_KEY` | API key for the endpoint | (required if not using `DASHSCOPE_API_KEY`) |
| `DASHSCOPE_API_KEY` | Alternate key name supported by DashScope | (optional) |

Example:

```bash
export QWEN_API_KEY="your-dashscope-api-key"
```

## Run

From this directory:

```bash
python example.py
```

You should see two turns printed:

- Turn 1 stores a memory (e.g. “My name is Alice.”)
- Turn 2 asks the agent to recall the stored memory via `search_memory`

## How the integration works

### 1) Create or reuse a Project

`_get_memmachine_project()` builds a `MemMachineClient` and calls:

- `get_or_create_project(org_id=..., project_id=...)`

This ensures the example can run repeatedly without manual setup.

### 2) Write memory

`add_memory()` calls:

```python
mem.add(content=memory, role="user", metadata={"type": "explicit_memory"})
```

### 3) Search memory

`search_memory()` calls:

```python
result = mem.search(query=query, limit=10)
```

and then formats key parts of the response:

- episodic memory (short/long-term buckets)
- episode summaries
- semantic memory items (if present)

## Important notes / customization

- **Memory scope**: this example uses `project.memory()` with no `user_id`, `agent_id`, or `session_id`, so all runs share a single default context. In a real app, pass identifiers to isolate memory per user/session:

	- `project.memory(user_id=..., agent_id=..., session_id=...)`

- **Tracing**: `set_tracing_disabled(True)` is called so you don’t need extra tracing configuration.

- **Model choice**: the example uses `model="qwen3-max"` via an OpenAI-compatible endpoint. You can swap to other providers by changing `_get_qwen_client()` and the `OpenAIChatCompletionsModel` configuration.

## Troubleshooting

- `Connection error to MemMachine`: ensure `MEMMACHINE_BASE_URL` points to a running server and `curl http://localhost:8080/health` returns OK.
- `Missing API key`: set `QWEN_API_KEY` (or `DASHSCOPE_API_KEY`).
- `ModuleNotFoundError: agents`: install the OpenAI Agents SDK package that provides the `agents` module (see “Install dependencies”).

