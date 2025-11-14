# AWS Strands Agent SDK Integration with MemMachine

This directory contains tools for integrating MemMachine memory operations into AWS Strands Agent SDK workflows.

## Overview

MemMachine provides memory tools that can be integrated into AWS Strands Agent SDK to enable AI agents with persistent memory capabilities. This allows agents to remember past interactions, user preferences, and context across multiple sessions.

## Installation

Ensure you have the required dependencies:

```bash
pip install memmachine
# If using AWS Strands Agent SDK
pip install strands-agents bedrock-agentcore
```

## Configuration

### AWS Credentials

AWS Strands Agent SDK requires AWS credentials to access AWS Bedrock services. Configure credentials using one of these methods:

1. **AWS CLI** (recommended):
   ```bash
   aws configure
   ```

2. **Environment variables**:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_REGION=us-east-1
   ```

3. **IAM role** (if running on EC2/ECS/Lambda)

4. **AWS credentials file**: `~/.aws/credentials`

### MemMachine Configuration

The tools can be configured via environment variables or directly in code:

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_BACKEND_URL` | URL of the MemMachine backend service | `http://localhost:8080` |
| `MEMORY_API_KEY` | API key for MemMachine authentication | `None` |
| `STRANDS_GROUP_ID` | Group identifier for the demo | `strands_group` |
| `STRANDS_AGENT_ID` | Agent identifier for the demo | `strands_agent` |
| `STRANDS_USER_ID` | User identifier for the demo | `default_user` |
| `STRANDS_SESSION_ID` | Session identifier for the demo | Auto-generated |

## Usage

### Basic Usage

```python
from tool import get_memmachine_tools, create_tool_handler

# Initialize tools
tools, tool_schemas = get_memmachine_tools(
    base_url="http://localhost:8080",
    group_id="my_group",
    agent_id="my_agent",
    user_id="user123",
)

# Get tool schemas for Strands Agent SDK
# tool_schemas can be passed to your Strands Agent configuration

# Create tool handler for executing tool calls
tool_handler = create_tool_handler(tools)

# Use tools directly
result = tools.add_memory(
    content="User prefers Python for backend development",
    metadata={"category": "preference"}
)

search_result = tools.search_memory(
    query="What does the user prefer for development?",
    limit=5
)
```

### Integration with Strands Agent SDK

```python
from strands import Agent
from tool import get_memmachine_tools, create_tool_handler
from strands_tools import set_tools_instance
import sys
import os

# Initialize MemMachine tools
tools, tool_schemas = get_memmachine_tools(
    base_url="http://localhost:8080",
    group_id="my_group",
    agent_id="my_agent",
    user_id="user123",
)

# Set up tools for Strands SDK
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from strands_tools import add_memory, search_memory, get_context
set_tools_instance(tools)

# Create tool handler for manual execution
tool_handler = create_tool_handler(tools)

# Create Strands Agent with function objects
agent = Agent(
    tools=[add_memory, search_memory, get_context],  # Pass function objects
    system_prompt="You are a helpful assistant with memory capabilities."
)

# The agent can now use the memory tools automatically
response = agent("Remember that I like Python programming")
print(response)

# Later, the agent can recall memories
response = agent("What programming language do I prefer?")
print(response)

# Cleanup
tools.close()
```

**Note:** AWS credentials must be configured for the Strands Agent to work. See [AWS Credentials Configuration](#aws-credentials) below.

### Direct Tool Usage

```python
from tool import MemMachineTools

# Initialize tools
tools = MemMachineTools(
    base_url="http://localhost:8080",
    group_id="my_group",
    agent_id="my_agent",
    user_id="user123",
)

# Add a memory
result = tools.add_memory(
    content="User mentioned they have a meeting tomorrow at 10 AM",
    metadata={"type": "reminder", "time": "10:00 AM"}
)

# Search memories
search_result = tools.search_memory(
    query="What meetings does the user have?",
    limit=5
)

# Get context
context = tools.get_context()
print(f"Current context: {context}")

# Cleanup
tools.close()
```

## Available Tools

### 1. add_memory

Store important information about the user or conversation into memory.

**Parameters:**
- `content` (required): The content to store in memory
- `user_id` (optional): User ID override
- `episode_type` (optional): Type of episode (default: "text")
- `metadata` (optional): Additional metadata dictionary

**Returns:**
- Dictionary with `status`, `message`, and `content` fields

### 2. search_memory

Retrieve relevant context, memories, or profile for a user.

**Parameters:**
- `query` (required): Search query string
- `user_id` (optional): User ID override
- `limit` (optional): Maximum number of results (default: 5, max: 20)

**Returns:**
- Dictionary with `status`, `results`, and `summary` fields

### 3. get_context

Get the current memory context configuration.

**Parameters:**
- `user_id` (optional): User ID override

**Returns:**
- Dictionary containing `group_id`, `agent_id`, `user_id`, and `session_id`

## Requirements

- MemMachine server running (default: http://localhost:8080)
- Python 3.8+
- AWS Strands Agent SDK (for full integration)

## Example Workflow

1. **Configure AWS Credentials**: Set up AWS credentials (see above)
2. **Start MemMachine Server**: Ensure MemMachine server is running
3. **Initialize Tools**: Create MemMachine tools with your configuration
4. **Set Up Tool Functions**: Import and configure `strands_tools` functions
5. **Create Agent**: Pass function objects to Strands Agent
6. **Use Agent**: The agent automatically calls tools when needed
7. **Cleanup**: Close the client when done

## Error Handling

All tool methods return dictionaries with a `status` field:
- `"success"`: Operation completed successfully
- `"error"`: Operation failed (check `message` field for details)

## Notes

- The tools automatically manage memory sessions
- Memories are stored in both episodic (short-term) and profile (long-term) memory
- Search queries are embedded and compared semantically
- Tool schemas follow AWS Bedrock tool specification format

