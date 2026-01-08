"""
Example: Using MemMachine with LlamaIndex.

This example demonstrates how to integrate MemMachine memory with LlamaIndex
to provide persistent memory capabilities for conversational AI applications.
"""
# ruff: noqa: T201

import os

from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI
from mem_machine_memory import MemMachineMemory

# Configuration from environment
MEMORY_BACKEND_URL = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
LLAMAINDEX_ORG_ID = os.getenv("LLAMAINDEX_ORG_ID", "llamaindex_org")
LLAMAINDEX_PROJECT_ID = os.getenv("LLAMAINDEX_PROJECT_ID", "llamaindex_project")
LLAMAINDEX_GROUP_ID = os.getenv("LLAMAINDEX_GROUP_ID", "llamaindex_demo")
LLAMAINDEX_AGENT_ID = os.getenv("LLAMAINDEX_AGENT_ID", "demo_agent")
LLAMAINDEX_USER_ID = os.getenv("LLAMAINDEX_USER_ID", "demo_user")
LLAMAINDEX_SESSION_ID = os.getenv("LLAMAINDEX_SESSION_ID", "demo_session_001")

# Initialize MemMachine-backed memory
memory = MemMachineMemory(
    base_url=MEMORY_BACKEND_URL,
    org_id=LLAMAINDEX_ORG_ID,
    project_id=LLAMAINDEX_PROJECT_ID,
    group_id=LLAMAINDEX_GROUP_ID,
    agent_id=LLAMAINDEX_AGENT_ID,
    user_id=LLAMAINDEX_USER_ID,
    session_id=LLAMAINDEX_SESSION_ID,
)

# Initialize an LLM (OpenAI shown here as in the example)
llm = OpenAI(api_key="<your-openai-api-key>")

# Build a simple chat agent with memory
agent = SimpleChatEngine.from_defaults(llm=llm, memory=memory)

# Interact with the agent
print(agent.chat("I am Alice, I like Python programming."))
print(agent.chat("What do you know about me?"))
