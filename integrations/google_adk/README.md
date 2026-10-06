# Google ADK Integration with MemMachine

This directory contains tools for integrating MemMachine with [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/) agents.

## Overview

MemMachine provides memory tools that can be registered with a Google ADK `LlmAgent` to enable persistent memory capabilities. This allows agents to remember past interactions, user preferences, and context across multiple sessions.

The tools are plain Python callables (`add_memory` and `search_memory`) with type hints and docstrings, so ADK builds the function declarations the model sees directly from their signatures.

## API keys

This integration uses **two independent LLM providers**, and you need a key for each:

| Key | Used by | For what |
|-----|---------|----------|
| `GOOGLE_API_KEY` | The **ADK agent** in your process (`google-genai` / Gemini) | Running the agent that decides when to call the memory tools |
| `OPENAI_API_KEY` | The **MemMachine server** (`http://localhost:8080`) | Generating embeddings and extracting/ingesting the memories you store |

`GOOGLE_API_KEY` is set in this integration's environment (see below). `OPENAI_API_KEY` is read by the **server** — set it where the server runs (for the Docker Compose stack, in the repo-root `.env`), not here. If the server's OpenAI key is missing or invalid, `add_memory` fails with a backend `500` even though the Gemini agent itself works.

## Installation

**1. Install the client libraries** (in your agent's environment):

```bash
# Install Google ADK (and python-dotenv for the example's .env loading)
pip install google-adk python-dotenv

# Install MemMachine client
pip install memmachine-client
```

**2. Install and configure the MemMachine server.** Follow the main
[MemMachine setup](../../README.md) to run the server (e.g. via `docker compose`).
**This is where the `OPENAI_API_KEY` goes** — the server needs it for embeddings
and memory ingestion. Set it in **one** of these places before starting the server:

- Repo-root `.env`: `OPENAI_API_KEY=sk-...` (recommended — the `configuration.yml`
  `openai_embedder` and `openai_model` blocks already reference `$OPENAI_API_KEY`), **or**
- Directly in `configuration.yml`, under both `openai_embedder` and `openai_model`
  (look for the `>>> ADD YOUR OPENAI API KEY HERE <<<` markers).

Then (re)start the server so it loads the key:

```bash
docker compose up -d --force-recreate memmachine
```

## Configuration

The example can be configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | API key for Gemini models (required) | — |
| `GOOGLE_MODEL` | Gemini model to use | `gemini-3.5-flash` |
| `MEMORY_BACKEND_URL` | URL of the MemMachine backend service | `http://localhost:8080` |
| `GOOGLE_ADK_ORG_ID` | Organization identifier | `google_adk_org` |
| `GOOGLE_ADK_PROJECT_ID` | Project identifier | `google_adk_project` |
| `GOOGLE_ADK_GROUP_ID` | Group identifier (optional) | `None` |
| `GOOGLE_ADK_AGENT_ID` | Agent identifier (optional) | `None` |
| `GOOGLE_ADK_USER_ID` | User identifier (optional) | `demo_user` |
| `GOOGLE_ADK_SESSION_ID` | Session identifier (optional) | `None` |

## Quick Start

### 1. Basic Usage

```python
from google.adk.agents import LlmAgent

from integrations.google_adk import MemMachineTools

# Create MemMachine tools scoped to a project and user
memmachine_tools = MemMachineTools(
    base_url="http://localhost:8080",
    org_id="my_org",
    project_id="my_project",
    user_id="user123",
)

# Register the tools with an ADK agent
agent = LlmAgent(
    name="memmachine_assistant",
    model="gemini-3.5-flash",
    instruction=(
        "You are a helpful assistant with persistent memory. "
        "Use search_memory to recall relevant facts before answering, "
        "and use add_memory to store new facts or preferences the user shares."
    ),
    tools=memmachine_tools.tools,
)
```

### 2. Running the Example

Copy the template into a `.env` **in this integration directory** and fill in your key:

```bash
cp integrations/google_adk/.env.example integrations/google_adk/.env
# then edit integrations/google_adk/.env and set GOOGLE_API_KEY
```

The example auto-loads that `.env` (via `python-dotenv`). Run from the repo root:

```bash
# pip install python-dotenv   # if not already installed

python -m integrations.google_adk.example \
  "Remember that my name is Tarun and my preferred language is Python."

python -m integrations.google_adk.example \
  "Search your memory: what is my name and preferred programming language?"
```

### 3. Using MemMachineTools Directly

```python
from integrations.google_adk import MemMachineTools

tools = MemMachineTools(
    base_url="http://localhost:8080",
    org_id="my_org",
    project_id="my_project",
    user_id="user123",
)

# Add memory
result = tools.add_memory(
    content="User prefers Python over JavaScript",
    role="user",
)
print(result)  # {"status": "success", "memory_ids": [...]}

# Search memory
results = tools.search_memory(
    query="What programming languages does the user prefer?",
    limit=5,
)
print(results["results"])

# Close the client when done (only closes clients this toolkit created)
tools.close()
```

## Tool Descriptions

### `add_memory`

Stores facts, preferences, or conversation context in MemMachine memory. Use this whenever the user shares new information that should be remembered.

**Parameters:**
- `content`: The content to store in memory (required)
- `role`: Message role — `"user"`, `"assistant"`, or `"system"` (default: `"user"`)

**Returns:** a dict with `status` and, on success, the stored `memory_ids`.

### `search_memory`

Recalls relevant facts and past conversations from MemMachine memory. Use this before answering when you need information from previous interactions.

**Parameters:**
- `query`: Natural-language description of what to recall (required)
- `limit`: Maximum number of results; must be positive (default: 5)

**Returns:** a dict with `status` and, on success, the episodic and semantic `results`.

## Requirements

- MemMachine server running (default: http://localhost:8080), configured with a valid `OPENAI_API_KEY` for embeddings and memory ingestion
- Python 3.10+
- `google-adk`
- A Gemini API key (`GOOGLE_API_KEY`) for the ADK agent

## Notes

- Scope memories with `user_id`, `group_id`, `agent_id`, and `session_id`; non-empty values are passed as metadata and used as search filters.
- Memories persist across runs, enabling long-term context.
- The tools return plain dicts (not JSON strings), matching ADK's function-tool convention.
