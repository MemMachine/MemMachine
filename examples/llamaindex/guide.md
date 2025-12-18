# LlamaIndex Integration with MemMachine

This directory contains examples and tools for integrating MemMachine with LlamaIndex.

## Overview

MemMachine provides persistent memory for LlamaIndex chat agents. It enables agents to remember past interactions, user preferences, and context across sessions by storing episodic and profile/semantic memories, and surfacing them back into the prompt at inference time.

## Configuration

The demo can be configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_BACKEND_URL` | URL of the MemMachine backend service | `http://localhost:8080` |
| `LLAMAINDEX_ORG_ID` | Organization identifier | `llamaindex_org` |
| `LLAMAINDEX_PROJECT_ID` | Project identifier | `llamaindex_project` |
| `LLAMAINDEX_GROUP_ID` | Group identifier for the demo | `llamaindex_demo` |
| `LLAMAINDEX_AGENT_ID` | Agent identifier for the demo | `demo_agent` |
| `LLAMAINDEX_USER_ID` | User identifier for the demo | `demo_user` |
| `LLAMAINDEX_SESSION_ID` | Session identifier for the demo | `demo_session_001` |

## Usage

### Setting Environment Variables

```bash
# Set environment variables (optional)
export MEMORY_BACKEND_URL="http://localhost:8080"
export LLAMAINDEX_ORG_ID="llamaindex_org"
export LLAMAINDEX_PROJECT_ID="llamaindex_project"
export LLAMAINDEX_GROUP_ID="llamaindex_demo"
export LLAMAINDEX_AGENT_ID="demo_agent"
export LLAMAINDEX_USER_ID="demo_user"
export LLAMAINDEX_SESSION_ID="demo_session_001"

# Run the demo
python examples/llamaindex/example.py
```

### Running the Demo

```bash
cd examples/llamaindex
python example.py
```

## Integration Guide

```python
import os

from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI
from mem_machine_memory import MemMachineMemory

# 1) Read configuration (or rely on defaults)
base_url = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
org_id = os.getenv("LLAMAINDEX_ORG_ID", "llamaindex_org")
project_id = os.getenv("LLAMAINDEX_PROJECT_ID", "llamaindex_project")
group_id = os.getenv("LLAMAINDEX_GROUP_ID", "llamaindex_demo")
agent_id = os.getenv("LLAMAINDEX_AGENT_ID", "demo_agent")
user_id = os.getenv("LLAMAINDEX_USER_ID", "demo_user")
session_id = os.getenv("LLAMAINDEX_SESSION_ID", "demo_session_001")

# 2) Initialize MemMachine-backed memory
memory = MemMachineMemory(
    base_url=base_url,
    org_id=org_id,
    project_id=project_id,
    group_id=group_id,
    agent_id=agent_id,
    user_id=user_id,
    session_id=session_id,
)

# 3) Build the chat engine and start chatting
llm = OpenAI(api_key="<your-openai-api-key>")
agent = SimpleChatEngine.from_defaults(llm=llm, memory=memory)

print(agent.chat("I am Alice, I like Python programming."))
print(agent.chat("What do you know about me?"))
```

## Notes

- MemMachine injects a SYSTEM message containing filtered user facts and a short summary retrieved from memory; the LLM answers with awareness of prior interactions.
- Tune `search_msg_limit` to balance recall and latency/noise.
- Scope memories per user/agent/session using the IDs above.

## Requirements

- MemMachine server running (default: http://localhost:8080)
- Python 3.12+
- LlamaIndex
- OpenAI-compatible LLM provider
