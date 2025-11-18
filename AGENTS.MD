# AGENTS.MD - Guide for AI Assistants

This document provides instructions for AI agents (like Cursor, Codex, etc.) on how to help developers build applications that use MemMachine. Use this guide to understand MemMachine's APIs, memory types, and integration patterns.

## Project Overview

MemMachine is an open-source memory layer for AI agents. It enables AI-powered applications to learn, store, and recall data and preferences from past sessions. The system provides three types of memory:

- **Short-term (Episodic) Memory**: Recent conversational context stored in a graph database (Neo4j)
- **Long-term (Episodic) Memory**: Summarized and deduplicated episodes from short-term memory
- **Profile (Semantic) Memory**: User-specific facts and preferences stored in PostgreSQL with vector embeddings

## Architecture Overview

### How MemMachine Works

1. **Your Application** → Interacts with MemMachine via REST API, Python SDK, or MCP Server
2. **MemMachine** → Processes interactions and stores them in appropriate memory types
3. **Storage** → Episodic Memory → Neo4j, Profile Memory → PostgreSQL

### Memory Types

- **Episodic Memory**: Stores conversational episodes (messages, interactions) in a graph database. Supports both short-term (recent context) and long-term (summarized) storage.
- **Profile Memory**: Stores user-specific facts, preferences, and knowledge extracted from conversations. Uses vector embeddings for semantic search.

## Getting Started

### Installation

MemMachine is **not a hosted service**. You must run your own MemMachine server instance. 

- **Docker (Recommended)**: Use the `memmachine-compose.sh` script for a complete setup
- **Python Package**: Install via `pip install memmachine` or `uv`
- **Configuration**: The compose script will help you set up `configuration.yml` with your database connections and model providers
- See the [Installation Guide](https://docs.memmachine.ai) for detailed setup instructions

### Running the Server

```bash
# Using the MemMachine compose script (recommended)
./memmachine-compose.sh

# The script will:
# - Check Docker installation
# - Create .env file if needed
# - Generate configuration.yml with your provider choice (OpenAI, Bedrock, or Ollama)
# - Set up API keys
# - Start all services (MemMachine, Neo4j, PostgreSQL)

# Other useful commands:
./memmachine-compose.sh stop      # Stop services
./memmachine-compose.sh restart   # Restart services
./memmachine-compose.sh logs      # View logs
./memmachine-compose.sh clean     # Remove all data and volumes

# Or using the Python package directly
memmachine-server --config configuration.yml
```

## API Structure

### REST Endpoints

The FastAPI server provides these endpoints:

- `POST /v1/memories`: Store episodes to both episodic and profile memory
- `POST /v1/memories/episodic`: Store episodes to episodic memory only
- `POST /v1/memories/profile`: Store episodes to profile memory only
- `POST /v1/memories/search`: Search across both episodic and profile memory
- `POST /v1/memories/episodic/search`: Search episodic memory only
- `POST /v1/memories/profile/search`: Search profile memory only
- `DELETE /v1/memories`: Delete memories
- `GET /v1/sessions`: Get session information
- `GET /health`: Health check endpoint
- `GET /metrics`: Prometheus metrics endpoint

### Headers and Session Context

MemMachine uses HTTP headers to manage session context and isolate memories. All endpoints that interact with memory accept these headers:

#### Header Names

- **`group-id`** (or `group_id`): Unique identifier for a group or shared context
- **`session-id`** (or `session_id`): Unique identifier for a specific conversation thread
- **`user-id`** (or `user_id`): Comma-separated list of user identifiers
- **`agent-id`** (or `agent_id`): Comma-separated list of agent identifiers

**Note**: Header names are case-insensitive and support both hyphenated (`group-id`) and underscore (`group_id`) formats.

#### Default Values and Logic

**Important**: Always set explicit `user-id` headers in your requests. Using defaults can cause memory isolation issues and data overlap between different users.

The system applies defaults when headers are not provided, but these should only be used for testing:

1. **`user-id`** (Required in production):
   - **Always set this explicitly** with a unique identifier for each user
   - Can be a comma-separated list: `"user1,user2,user3"`
   - Values are automatically sorted and deduplicated
   - Defaults to `["default"]` only if both `user-id` and `agent-id` are empty (not recommended)

2. **`group-id`**:
   - Defaults to the first `user-id` if `user-id` is provided
   - Defaults to `"default"` if not provided and `user-id` is empty
   - **Recommended**: Set explicitly for group conversations or shared contexts

3. **`session-id`**:
   - Defaults to the first `user-id` if `user-id` is provided
   - Defaults to `"default"` if not provided and `user-id` is empty
   - **Recommended**: Set explicitly with a unique identifier for each conversation thread

4. **`agent-id`**:
   - Defaults to `[]` (empty list) if not provided
   - Can be a comma-separated list: `"agent1,agent2"`
   - Set this to identify which AI agent(s) are participating in the conversation

#### Response Headers

The server echoes session context in response headers:
- Response includes `group-id`, `session-id`, `user-id`, and `agent-id` headers
- List values are joined with commas in response headers
- This allows clients to track the session context used for the request

#### Example Usage

```python
import requests

# Single user request (recommended)
# Always set user-id explicitly for proper memory isolation
response = requests.post(
    "http://localhost:8000/v1/memories",
    headers={"user-id": "alice@example.com"},
    json={"content": "I prefer dark mode"}
)
# Results in: group-id="alice@example.com", session-id="alice@example.com", 
#             user-id=["alice@example.com"], agent-id=[]

# Single user with explicit session
response = requests.post(
    "http://localhost:8000/v1/memories",
    headers={
        "user-id": "alice@example.com",
        "session-id": "chat-2024-01-15"
    },
    json={"content": "Hello, I'm Alice"}
)

# Multi-user group conversation
response = requests.post(
    "http://localhost:8000/v1/memories",
    headers={
        "group-id": "project-alpha",
        "session-id": "chat-thread-42",
        "user-id": "alice@example.com,bob@example.com",
        "agent-id": "crm-agent"
    },
    json={"content": "Let's discuss the project"}
)
```

## Python SDK

MemMachine provides a Python SDK for easier integration:

```python
from memmachine import MemMachineClient, Memory

# Initialize client
client = MemMachineClient(base_url="http://localhost:8000")

# Create a memory context
memory = Memory(
    group_id="project-alpha",
    session_id="chat-1",
    user_id=["alice@example.com"]
)

# Store a memory
await memory.add_episode(
    content="I prefer Python over JavaScript",
    episode_type="preference"
)

# Search memories
results = await memory.search(query="What are Alice's preferences?")
```

See the [Python SDK documentation](https://docs.memmachine.ai/api_reference/python/client) for more details.

## Data Types

### Episode

An `Episode` represents a single event or piece of data in the memory system:

- `uuid`: Unique identifier (UUID)
- `episode_type`: Type of episode (e.g., "message", "preference", "action")
- `content_type`: Format of content (currently `ContentType.STRING`)
- `content`: The actual data/content
- `timestamp`: When the episode occurred
- `group_id`: Group identifier
- `session_id`: Session identifier
- `producer_id`: Who created this episode
- `produced_for_id`: Intended recipient (optional)
- `user_metadata`: Additional JSON-compatible metadata (optional)

### MemoryContext

A `MemoryContext` isolates memories for different conversations:

- `group_id`: Group identifier
- `session_id`: Session identifier
- `user_id`: Set of user identifiers
- `agent_id`: Set of agent identifiers

## Configuration

MemMachine is configured via a YAML file (`configuration.yml`). Key sections:

- **`model`**: Language model configurations (OpenAI, Ollama, AWS Bedrock, etc.)
- **`embedder`**: Embedding model configurations
- **`storage`**: Database connections (Neo4j for episodic, PostgreSQL for profile)
- **`profile_memory`**: Profile memory settings
- **`long_term_memory`**: Long-term memory settings

See `sample_configs/` directory for example configurations.

## Common Integration Patterns

### Storing User Messages

```python
# Store a user message
response = requests.post(
    "http://localhost:8000/v1/memories",
    headers={"user-id": "user123"},
    json={
        "content": "I'm working on a Python project",
        "episode_type": "message"
    }
)
```

### Searching for Context

```python
# Search for relevant context before responding
response = requests.post(
    "http://localhost:8000/v1/memories/search",
    headers={"user-id": "user123"},
    json={"query": "What projects is the user working on?"}
)

context = response.json()
# Use context['episodic'] and context['profile'] in your LLM prompt
```

### Multi-User Conversations

```python
# Group chat scenario
response = requests.post(
    "http://localhost:8000/v1/memories",
    headers={
        "group-id": "team-alpha",
        "session-id": "slack-channel-123",
        "user-id": "alice,bob,charlie",
        "agent-id": "team-assistant"
    },
    json={"content": "Meeting scheduled for tomorrow"}
)
```

## Error Handling

- **400 Bad Request**: Invalid request data or missing required fields
- **404 Not Found**: Memory instance not found for the given session context
- **500 Internal Server Error**: Server-side error
- **503 Service Unavailable**: Service not ready (check `/health` endpoint)

Always check the response status and handle errors appropriately in your application.

## Best Practices

1. **Always set user-id headers**: **Required** - Always provide explicit `user-id` headers for proper memory isolation. Never rely on defaults in production as this can cause data overlap between different users.

2. **Use unique session IDs**: Use unique session IDs for different conversations to avoid data overlap

3. **Set group-id for shared contexts**: Explicitly set `group-id` for group conversations or shared contexts

4. **Search before responding**: Query memory for relevant context before generating responses

5. **Store important information**: Store user preferences, facts, and important conversation points

6. **Monitor health**: Use the `/health` endpoint to check server status

7. **Handle async operations**: Most operations are async - use `async/await` or async HTTP clients

## Examples

Check the `examples/` directory for complete integration examples:

- **`examples/crm/`**: CRM agent with memory
- **`examples/financial_analyst/`**: Financial analysis agent
- **`examples/health_assistant/`**: Healthcare assistant
- **`examples/writing_assistant/`**: Writing assistant
- **`examples/frontend/`**: Web frontend integration

## Getting Help

- **Documentation**: [https://docs.memmachine.ai](https://docs.memmachine.ai)
- **Examples**: See `examples/` directory
- **API Reference**: [https://docs.memmachine.ai/api_reference](https://docs.memmachine.ai/api_reference)
- **Community**: Join the [Discord Server](https://discord.gg/usydANvKqD)

## Summary

When building applications with MemMachine:

1. **Set up your server**: Run MemMachine locally or in your infrastructure
2. **Use headers**: Set appropriate session headers (`group-id`, `session-id`, `user-id`, `agent-id`)
3. **Store memories**: Store important user information and conversation context
4. **Search context**: Query memory before generating responses to provide personalized, context-aware answers
5. **Handle errors**: Implement proper error handling for API calls

MemMachine enables your AI applications to remember past interactions, learn user preferences, and provide personalized, context-aware responses across sessions.
