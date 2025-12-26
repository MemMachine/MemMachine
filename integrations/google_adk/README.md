
# MemMachine x Google ADK Integration

This integration provides a MemMachine-backed implementation of Google ADK’s `BaseMemoryService`.

It lets ADK agents:

- Store ADK session events into MemMachine (as episodic memory)
- Retrieve relevant memories using MemMachine search (episodic + semantic)

## What’s included

The adapter lives in the Python client package as:

- `memmachine.integrations.google_adk.memmachine_memory_service.MemmachineMemoryService`

Key behaviors:

- Session boundary fields are stored in MemMachine `metadata` (e.g. `app_name`, `user_id`, `session_id`, `event_id`, …)
- Each ingested memory message contains only: `content`, `timestamp`, `metadata`
- Authentication uses `Authorization: Bearer <api_key>`

## Installation

Install the MemMachine client with the Google ADK extra:

```bash
pip install "memmachine-client[google-adk]"
```

This extra installs the additional dependencies required by the adapter (Google ADK + Google GenAI types).

## Configuration

You configure the adapter via constructor arguments:

- `api_key` (required)
- `endpoint` (optional)
- `org_id` / `project_id` (optional)
- `timeout_s` (optional)

### Endpoint

The adapter expects a MemMachine REST v2 base URL.

- MemMachine Cloud (default in the adapter): `https://api.memmachine.ai/v2`
- Self-hosted example: `http://localhost:8080/api/v2`

## Usage

The recommended way to use memory in ADK is through a `Runner` + the `load_memory` tool.

### 1) Configure MemMachine as the ADK `memory_service`

```python
from memmachine.integrations.google_adk.memmachine_memory_service import (
    MemmachineMemoryService,
)

memory_service = MemmachineMemoryService(
    api_key="YOUR_MEMMACHINE_API_KEY",
    endpoint="http://localhost:8080/api/v2",  # or use the default cloud endpoint
    org_id="my-org",
    project_id="my-project",
)
```

### 2) Give your agent the `load_memory` tool

```python
from google.adk.agents import LlmAgent
from google.adk.tools import load_memory

memory_recall_agent = LlmAgent(
    model="gemini-2.0-flash",  # use a valid model
    name="MemoryRecallAgent",
    instruction=(
        "Answer the user's question. Use the 'load_memory' tool "
        "if the answer might be in past conversations."
    ),
    tools=[load_memory],
)
```

### 3) Wire the `memory_service` into the `Runner`

```python
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

APP_NAME = "memory_example_app"
USER_ID = "mem_user"

session_service = InMemorySessionService()

runner = Runner(
    agent=memory_recall_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)
```

### 4) Add a finished session to MemMachine (so it can be recalled later)

```python
# completed_session: google.adk.sessions.session.Session
await memory_service.add_session_to_memory(completed_session)
```

### 5) Ask a question; the agent will call `load_memory` when needed

```python
from google.genai.types import Content, Part

await runner.session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id="session_recall",
)

question = Content(parts=[Part(text="What is my favorite project?")], role="user")

async for event in runner.run_async(
    user_id=USER_ID,
    session_id="session_recall",
    new_message=question,
):
    if event.is_final_response() and event.content and event.content.parts:
        print(event.content.parts[0].text)
```

Notes:

- `load_memory` queries the configured `memory_service`.
- Searches are automatically scoped to `(app_name, user_id)` via MemMachine metadata.

## Notes

- If you do not install the extra, importing the ADK integration module will fail because the Google ADK / GenAI packages won’t be present.
- This adapter is intentionally minimal and keeps the MemMachine payloads simple (text + timestamp + metadata).

