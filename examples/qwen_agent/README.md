# Qwen Agent + MemMachine (Tool Integration Example)

This example shows how to integrate **MemMachine** into a **Qwen Agent** app by exposing MemMachine operations as Qwen Agent tools:

- `save_memory`: writes a message to MemMachine
- `search_memory`: queries MemMachine and formats results back to the agent

The runnable script is in [qwen_agent_example.py](qwen_agent_example.py).

## Overview

The integration pattern is:

1. Your Qwen Agent calls a tool (`save_memory` / `search_memory`) based on the user request.
2. The tool uses the MemMachine Python SDK to connect to a MemMachine project.
3. The tool stores or retrieves memory, then returns a plain-text summary to the agent.

## Architecture

```
qwen_agent/
├── qwen_agent_example.py   # Qwen Agent script + MemMachine tools
└── README.md               # This file
```

## Prerequisites

- Python 3.10+ (recommended: the same version you use for MemMachine)
- A running MemMachine server (start one locally or connect to an existing deployment)
- Qwen Agent credentials/configuration (follow the `qwen-agent` docs for how to authenticate)

## Start MemMachine

From the repo root, you can start MemMachine using the provided compose helper:

```bash
./memmachine-compose.sh
```

Once it’s running, verify:

```bash
curl -s http://localhost:8080/health
```

## Install Dependencies

This example only needs the MemMachine **Python client** package plus Qwen Agent:

```bash
pip install memmachine-client qwen-agent
```

If you are developing MemMachine from source in this repo, you can still install the local package instead:

```bash
pip install -e .
```

## Configuration

This example reads the following environment variables:

- `MEMMACHINE_BASE_URL` (default: `http://localhost:8080`)
- `MEMMACHINE_API_KEY` (default: empty)
- `MEMMACHINE_ORG_ID` (default: `default_org`)
- `MEMMACHINE_PROJECT_ID` (default: `qwen_agent_demo`)

Example:

```bash
export MEMMACHINE_BASE_URL="http://localhost:8080"  # or your existing MemMachine deployment URL
export MEMMACHINE_ORG_ID="default_org"
export MEMMACHINE_PROJECT_ID="qwen_agent_demo"
```

Qwen Agent authentication is configured via the `DASHSCOPE_API_KEY` environment variable:

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

## Run the Example

```bash
cd examples/qwen_agent
python qwen_agent_example.py
```

The script runs two short conversations:

1. A user message that asks the agent to remember something (it should call `save_memory`).
2. A user message that asks the agent to recall it (it should call `search_memory`).

## Notes

- This script uses `project.memory()` with no explicit `user_id` / `session_id`. For real applications, you typically want per-user/per-session isolation, e.g.:

  ```python
  mem = project.memory(user_id="user123", session_id="session456")
  ```

- If you see connection errors, confirm the server is running and `MEMMACHINE_BASE_URL` is reachable.

## Troubleshooting

- **MemMachine connection refused**: ensure the server is running and `MEMMACHINE_BASE_URL` is correct.
- **Qwen model/auth errors**: verify your Qwen Agent credentials/config follow `qwen-agent` requirements.
