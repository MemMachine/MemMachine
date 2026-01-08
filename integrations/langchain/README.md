# LangChain Integration with MemMachine

This directory contains the integration between MemMachine and LangChain, providing persistent memory capabilities for LangChain applications.

## Overview

The MemMachine integration for LangChain implements the `BaseMemory` interface, allowing you to use MemMachine as a memory backend for LangChain chains and agents. This enables:

- **Persistent Memory**: Conversations and context persist across sessions
- **Semantic Search**: Retrieve relevant memories based on semantic similarity
- **User Context**: Automatic filtering by user_id, agent_id, session_id
- **Episodic & Semantic Memory**: Access to both conversation history and extracted knowledge

## Installation

### Prerequisites

1. MemMachine server running (default: `http://localhost:8080`)
2. Python 3.10+
3. Required packages:
   ```bash
   pip install langchain memmachine
   ```

### Optional Dependencies

For using with OpenAI LLMs:
```bash
pip install openai
```

## Quick Start

### Basic Usage

```python
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from integrations.langchain.memory import MemMachineMemory

# Initialize MemMachine memory
memory = MemMachineMemory(
    base_url="http://localhost:8080",
    org_id="my_org",
    project_id="my_project",
    user_id="user123",
    session_id="session456",
)

# Create LLM
llm = OpenAI(temperature=0)

# Create conversation chain with MemMachine memory
chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
)

# Use the chain
response = chain.run("Hello, my name is Alice")
response = chain.run("What's my name?")  # Will remember from previous interaction
```

### Advanced Configuration

```python
from integrations.langchain.memory import MemMachineMemory

memory = MemMachineMemory(
    base_url="http://localhost:8080",
    org_id="my_org",
    project_id="my_project",
    user_id="user123",
    agent_id="agent456",
    session_id="session789",
    group_id="group1",
    search_limit=10,  # Number of memories to retrieve
    return_messages=False,  # Set to True to return LangChain message objects
)
```

## Configuration

The integration can be configured via environment variables or constructor parameters:

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| `base_url` | `MEMORY_BACKEND_URL` | `http://localhost:8080` | MemMachine server URL |
| `org_id` | `LANGCHAIN_ORG_ID` | `langchain_org` | Organization ID |
| `project_id` | `LANGCHAIN_PROJECT_ID` | `langchain_project` | Project ID |
| `user_id` | `LANGCHAIN_USER_ID` | `None` | User identifier |
| `agent_id` | `LANGCHAIN_AGENT_ID` | `None` | Agent identifier |
| `session_id` | `LANGCHAIN_SESSION_ID` | `None` | Session identifier |
| `group_id` | `LANGCHAIN_GROUP_ID` | `None` | Group identifier |
| `search_limit` | - | `10` | Max memories to retrieve |
| `return_messages` | - | `False` | Return LangChain message objects |

## How It Works

### Memory Storage

When `save_context()` is called:

1. Extracts user input and AI output from inputs/outputs dictionaries
2. Stores user message to MemMachine with `role="user"`
3. Stores AI response to MemMachine with `role="assistant"`
4. Messages are automatically filtered by the Memory instance's context (user_id, agent_id, session_id)

### Memory Retrieval

When `load_memory_variables()` is called:

1. Builds a search query from the input (or uses a default query)
2. Searches MemMachine for relevant episodic and semantic memories
3. Formats episodic memories as conversation history
4. Formats semantic memories as context facts
5. Returns both as memory variables

### Memory Variables

The memory provides two variables:

- **`history`**: Conversation history from episodic memory (formatted as "Human: ..." / "AI: ...")
- **`memmachine_context`**: Extracted facts from semantic memory (formatted as "feature: value")

## Examples

### Example 1: Simple Conversation

```python
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from integrations.langchain.memory import MemMachineMemory

memory = MemMachineMemory(
    base_url="http://localhost:8080",
    org_id="demo_org",
    project_id="demo_project",
    user_id="user123",
)

llm = OpenAI(temperature=0)
chain = ConversationChain(llm=llm, memory=memory)

# First interaction
chain.run("My name is Bob and I like Python")

# Second interaction - will remember the name
chain.run("What's my name?")
```

### Example 2: Custom Prompt with Memory Context

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from integrations.langchain.memory import MemMachineMemory

memory = MemMachineMemory(
    base_url="http://localhost:8080",
    org_id="demo_org",
    project_id="demo_project",
    user_id="user123",
)

prompt = PromptTemplate(
    input_variables=["history", "memmachine_context", "input"],
    template="""You are a helpful assistant with access to the user's memory.

Relevant context from memory:
{memmachine_context}

Conversation history:
{history}

User: {input}
Assistant:""",
)

llm = OpenAI(temperature=0)
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

response = chain.run("What do I like?")
```

### Example 3: Direct Memory Operations

```python
from integrations.langchain.memory import MemMachineMemory

memory = MemMachineMemory(
    base_url="http://localhost:8080",
    org_id="demo_org",
    project_id="demo_project",
    user_id="user123",
)

# Add memory directly
memory._memory.add(
    content="I prefer working in the morning",
    role="user",
)

# Search memories
results = memory.load_memory_variables({"input": "What are my preferences?"})
print(results["history"])
print(results["memmachine_context"])
```

## Running the Example

```bash
# Set environment variables (optional)
export MEMORY_BACKEND_URL="http://localhost:8080"
export LANGCHAIN_ORG_ID="my_org"
export LANGCHAIN_PROJECT_ID="my_project"
export LANGCHAIN_USER_ID="user123"

# Run the example
cd integrations/langchain
python example.py
```

## API Reference

### MemMachineMemory

#### `__init__(...)`

Initialize MemMachine memory for LangChain.

**Parameters:**
- `base_url` (str): Base URL for MemMachine server
- `org_id` (str): Organization ID
- `project_id` (str): Project ID
- `user_id` (str, optional): User identifier
- `agent_id` (str, optional): Agent identifier
- `session_id` (str, optional): Session identifier
- `group_id` (str, optional): Group identifier
- `search_limit` (int): Maximum number of memories to retrieve (default: 10)
- `client` (MemMachineClient, optional): Pre-initialized client
- `return_messages` (bool): Return LangChain message objects (default: False)

#### `load_memory_variables(inputs: Dict[str, Any]) -> Dict[str, Any]`

Load memory variables from MemMachine.

**Parameters:**
- `inputs`: Input dictionary (may contain "input", "question", "query", or "messages")

**Returns:**
- Dictionary with keys:
  - `history`: Conversation history string or list of messages
  - `memmachine_context`: Semantic memory context string

#### `save_context(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None`

Save conversation context to MemMachine.

**Parameters:**
- `inputs`: Input dictionary (typically contains user message)
- `outputs`: Output dictionary (typically contains AI response)

#### `clear() -> None`

Clear memory (note: doesn't delete from MemMachine, only resets local state).

## Integration with LangChain Components

### ConversationChain

```python
from langchain.chains import ConversationChain
from integrations.langchain.memory import MemMachineMemory

memory = MemMachineMemory(...)
chain = ConversationChain(llm=llm, memory=memory)
```

### LLMChain

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from integrations.langchain.memory import MemMachineMemory

memory = MemMachineMemory(...)
prompt = PromptTemplate(
    input_variables=["history", "memmachine_context", "input"],
    template="...",
)
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
```

### Agents

```python
from langchain.agents import initialize_agent
from integrations.langchain.memory import MemMachineMemory

memory = MemMachineMemory(...)
agent = initialize_agent(
    tools=[],
    llm=llm,
    agent="conversational-react-description",
    memory=memory,
    verbose=True,
)
```

## Troubleshooting

### Common Issues

1. **Connection Error**: Ensure MemMachine server is running at the specified URL
   ```bash
   curl http://localhost:8080/health
   ```

2. **Import Error**: Install required dependencies
   ```bash
   pip install langchain memmachine
   ```

3. **Memory Not Persisting**: Check that user_id, session_id are set correctly

4. **No Search Results**: 
   - Ensure memories have been added first
   - Check that search_limit is appropriate
   - Verify context filters (user_id, agent_id, session_id) match stored memories

## See Also

- [MemMachine Documentation](https://docs.memmachine.ai)
- [LangChain Memory Documentation](https://python.langchain.com/docs/modules/memory/)
- [LangGraph Integration](../langgraph/README.md) - Similar integration for LangGraph

## License

This integration follows the same license as MemMachine.

