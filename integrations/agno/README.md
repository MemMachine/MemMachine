# Agno integration with MemMachine

Use `MemMachineTools` to give an Agno agent tools for storing and searching
persistent memories. The integration follows the other framework integrations:
it uses `MemMachineClient`, a project, and metadata for the memory context.
It implements Agno's [custom toolkit interface](https://docs.agno.com/tools/creating-tools/toolkits).

## Setup

Requirements:

- Python 3.12 or newer.
- A running MemMachine server. Follow the
  [server quick start](https://docs.memmachine.ai/getting_started/quickstart)
  before running this example.
- An OpenAI API key for the example agent.

Run these commands from the MemMachine repository root. Install the local client
and common packages so the integration uses the APIs in this checkout:

```bash
python3 -m venv .venv-agno
source .venv-agno/bin/activate
python -m pip install -e ./packages/common -e ./packages/client agno openai

export OPENAI_API_KEY="your-openai-api-key"
export MEMORY_BACKEND_URL="http://localhost:8080"
export AGNO_ORG_ID="agno_org"
export AGNO_PROJECT_ID="agno_project"
export AGNO_USER_ID="demo_user"
```


## Run the example

From the repository root, with the virtual environment activated:

```bash
python -m integrations.agno.example \
  "Remember that my name is Tarun and I prefer Python for backend development and love travelling. Italy is my favourite country"

python -m integrations.agno.example \
  "Search your memory: what is my name and preferred programming language?"
```

Each command starts a new agent. The first asks the agent to save a fact; the
second asks it to retrieve that fact from MemMachine. Keep the same organization,
project, and metadata settings between runs. Allow the server to finish memory
processing before trying recall. Tool selection is performed by the model;
these commands are a manual usage example, not an automated test.

## Configuration

These environment variables configure `example.py`. When using the toolkit
directly, pass its constructor arguments instead.

| Variable | Default | Purpose |
|----------|---------|---------|
| `MEMORY_BACKEND_URL` | `http://localhost:8080` | MemMachine server URL |
| `AGNO_ORG_ID` | `agno_org` | Organization identifier |
| `AGNO_PROJECT_ID` | `agno_project` | Project identifier |
| `AGNO_USER_ID` | `demo_user` | User metadata and search filter |
| `AGNO_AGENT_ID` | Unset | Optional agent metadata and search filter |
| `AGNO_GROUP_ID` | Unset | Optional group metadata and search filter |
| `AGNO_SESSION_ID` | Unset | Optional session metadata and search filter |
| `OPENAI_API_KEY` | Required | OpenAI credentials for the example |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model for the example |

The toolkit defaults `user_id` to `None`; the example supplies `demo_user`.
Configured context fields are stored as metadata and included by the client in
search filters. Leave `session_id` unset to recall across sessions. Setting it
restricts the context to that session. These filters are not authentication or
authorization controls.

## Use in your agent

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from memmachine_client import MemMachineClient

from integrations.agno import MemMachineTools

with MemMachineClient(base_url="http://localhost:8080") as client:
    memory_tools = MemMachineTools(
        client=client,
        org_id="my_org",
        project_id="my_project",
        user_id="alice",
    )
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[memory_tools],
        instructions=[
            "Search memory before answering questions about the user.",
            "Save new user facts and preferences using add_memory.",
            "Only confirm a save after the tool succeeds.",
        ],
    )
    agent.print_response("Remember that I prefer aisle seats on flights.")
```

The client gets or creates the project when a memory tool is first called.
`add_memory(content, role="user")` returns JSON with stored memory identifiers.
`search_memory(query, limit=5)` returns the client's serialized search result,
preserving episodic and semantic memory data. Client errors propagate to Agno's
tool execution handling; the toolkit does not return fabricated success results.

Only these two methods are exposed to the model. Organization, project, and
context identifiers are configured by application code. The integration provides
synchronous tools for `Agent.run()` and `Agent.print_response()`; it does not
implement Agno session storage or automatically save every conversation turn.

If the toolkit creates its own client, call `memory_tools.close()` when finished.
When you supply a client, manage its lifetime yourself, as in the example above.
