# MemMachine Client

A Python client library for the MemMachine memory system.

## Features

- **Simple API**: Easy-to-use interface
- **Memory Management**: Add and search episodic and profile memories
- **Context Awareness**: Automatic context retrieval for better responses
- **Error Handling**: Robust error handling with retry mechanisms
- **Type Safety**: Full type hints for better development experience

## Installation

```bash
# Install from source (for development)
pip install -e .

# Or install dependencies
pip install requests urllib3
```

## Quick Start

### Basic Usage

```python
from memmachine import MemMachineClient

# Initialize client
client = MemMachineClient(
    base_url="http://localhost:8080",
    timeout=30
)

# Create a memory instance
memory = client.memory(
    group_id="my_group",
    agent_id="my_agent",
    user_id="user123",
    session_id="session456"
)

# Add memories
memory.add("I like pizza", metadata={"type": "preference"})
memory.add("I work as a software engineer", metadata={"type": "fact"})

# Search memories
results = memory.search("What do I like to eat?")
print(results)
```

## API Reference

### MemMachineClient

The main client class for interacting with MemMachine.

#### Constructor

```python
MemMachineClient(
    api_key: Optional[str] = None,
    base_url: str = "http://localhost:8080",
    timeout: int = 30,
    max_retries: int = 3,
    **kwargs
)
```

#### Methods

- `memory(group_id, agent_id, user_id, session_id)` - Create a Memory instance
- `health_check()` - Check server health

### Memory

Interface for managing episodic and profile memory.

#### Methods

- `add(content, producer, produced_for, episode_type, metadata)` - Add a memory
- `search(query, limit, filter_dict)` - Search memories
- `get_context()` - Get current context

## Examples

### Basic Memory Operations

```python
from memmachine import MemMachineClient

client = MemMachineClient(base_url="http://localhost:8080")

# Create memory instance
memory = client.memory(
    group_id="demo_group",
    agent_id="demo_agent",
    user_id="user123",
    session_id="demo_session"
)

# Add memories with metadata
memory.add("I like pizza", metadata={"type": "preference", "category": "food"})
memory.add("I work as a software engineer", metadata={"type": "fact", "category": "work"})

# Search memories
results = memory.search("What do I like to eat?")
print(f"Episodic memory: {results.get('episodic_memory', [])}")
print(f"Profile memory: {results.get('profile_memory', [])}")

# Search with filters
work_results = memory.search("Tell me about work", filter_dict={"category": "work"})
print(f"Work results: {work_results}")
```

### Multiple Users

```python
from memmachine import MemMachineClient

client = MemMachineClient(base_url="http://localhost:8080")

# Create memory instances for multiple users
users = ["alice", "bob", "charlie"]
memories = {}

for user in users:
    memories[user] = client.memory(
        group_id="team_group",
        agent_id="team_agent",
        user_id=user
    )

# Add user-specific memories
memories["alice"].add("I'm a frontend developer", metadata={"role": "frontend"})
memories["bob"].add("I'm a backend developer", metadata={"role": "backend"})
memories["charlie"].add("I'm a DevOps engineer", metadata={"role": "devops"})

# Search across users
for user, memory in memories.items():
    results = memory.search("What is your role?")
    print(f"{user}: {results}")
```

### Error Handling

```python
from memmachine import MemMachineClient

try:
    client = MemMachineClient(base_url="http://localhost:8080")
    
    # Check server health
    health = client.health_check()
    print(f"Server health: {health}")
    
    # Create memory instance
    memory = client.memory(
        group_id="demo_group",
        agent_id="demo_agent",
        user_id="user123"
    )
    
    # Add memory
    memory.add("Test memory")
    
except Exception as e:
    print(f"Error: {e}")
```

### Context Manager Usage

```python
from memmachine import MemMachineClient

# Use client as context manager
with MemMachineClient(base_url="http://localhost:8080") as client:
    memory = client.memory(
        group_id="demo_group",
        agent_id="demo_agent",
        user_id="user123"
    )
    
    memory.add("This is a test memory")
    results = memory.search("test")
    print(f"Results: {results}")

# Client is automatically closed
```

## Configuration

### Environment Variables

- `MEMORY_BACKEND_URL`: Base URL for MemMachine server (default: http://localhost:8080)
- `MEMORY_API_KEY`: API key for authentication (optional for local development)

### Client Configuration

```python
client = MemMachineClient(
    api_key="your_api_key",  # Optional
    base_url="http://localhost:8080",
    timeout=30,  # Request timeout in seconds
    max_retries=3  # Maximum retries for failed requests
)
```

<!-- Comparison section removed: no external product references -->

## Running Examples

```bash
# Start MemMachine server first
python -m memmachine.server.app

# Run examples
python examples/memmachine_client_example.py
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the same license as MemMachine.
