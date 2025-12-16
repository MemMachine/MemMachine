# Guide for AI Assistants - Using MemMachine

This guide helps AI coding assistants (like Cursor, Claude Code, Codex, etc.) understand how to help developers integrate MemMachine into their applications. MemMachine is an open-source memory layer for AI agents that enables applications to learn, store, and recall data and preferences from past sessions.

## Project Overview

MemMachine provides two types of memory:

- **Episodic Memory**: Stores conversational episodes (messages, interactions) in a graph database (Neo4j). Supports both short-term (recent context) and long-term (summarized) storage.
- **Semantic Memory**: User-specific facts, preferences, and knowledge extracted from conversations. Uses vector embeddings for semantic search and is stored in PostgreSQL.

## Installation

MemMachine is **not a hosted service**. Users must run their own MemMachine server instance.

### Quick Installation (Docker - Recommended)

The easiest way to get started is using Docker with the provided compose script:

```bash
# Download and run the setup script
./memmachine-compose.sh

# The script will:
# - Check Docker installation
# - Create .env file if needed
# - Generate cfg.yml (or configuration.yml) with provider choice (OpenAI, Bedrock, or Ollama)
# - Set up API keys
# - Start all services (MemMachine, Neo4j, PostgreSQL)

# Other useful commands:
./memmachine-compose.sh stop      # Stop services
./memmachine-compose.sh restart   # Restart services
./memmachine-compose.sh logs      # View logs
./memmachine-compose.sh clean     # Remove all data and volumes
```

### Python Package Installation

Users can install MemMachine via pip:

```bash
# Install the full suite (client + server)
pip install memmachine

# Or install components separately:
pip install memmachine-client    # Just the Python client library
pip install memmachine-server   # Just the server

# For GPU support (if using local embedding models):
pip install "memmachine-server[gpu]"
```

### Running the Server

After installation, users can run the server:

```bash
# Using the Python package
# Note: The config file can be named cfg.yml or configuration.yml
memmachine-server --config cfg.yml
# or
memmachine-server --config configuration.yml

# Or if using Docker, the compose script handles this automatically
```

## API Structure

### REST API v2

The v2 API uses a project-based model where all operations are scoped to an organization and project:

**Base URL**: `http://localhost:8080/api/v2`

**Key Endpoints**:

- `POST /api/v2/projects` - Create a new project
- `POST /api/v2/projects/get` - Get project information
- `POST /api/v2/projects/list` - List all projects
- `POST /api/v2/projects/delete` - Delete a project
- `POST /api/v2/memories` - Add memories (episodes)
- `POST /api/v2/memories/search` - Search memories
- `POST /api/v2/memories/episodic/delete` - Delete episodic memories
- `POST /api/v2/memories/semantic/delete` - Delete semantic memories
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics endpoint

## Python SDK Usage

The Python SDK provides a convenient interface for interacting with MemMachine. Here's how users typically use it:

### Basic Setup

```python
from memmachine import MemMachineClient

# Initialize client
client = MemMachineClient(base_url="http://localhost:8080")

# Create or get a project
# Note: description, embedder, and reranker are optional
# - description defaults to "" (empty string)
# - embedder defaults to "" (uses server's configured default)
# - reranker defaults to "" (uses server's configured default)
project = client.create_project(
    org_id="my-org",
    project_id="my-project",
    description="My project description"  # Optional, defaults to ""
)

# Or get an existing project
project = client.get_project(org_id="my-org", project_id="my-project")
```

### Working with Memory

```python
# Create a memory interface for a specific context
memory = project.memory(
    user_id="user123",        # User identifier (stored in metadata)
    agent_id="agent456",      # Agent identifier (stored in metadata)
    session_id="session789",  # Session identifier (stored in metadata)
    group_id="group1"         # Optional: Group identifier (stored in metadata)
)

# Add memories
# Note: Default values when not specified:
# - role defaults to "user"
# - producer defaults to user_id (if set) or "user"
# - produced_for defaults to agent_id (if set) or "agent"
# - episode_type defaults to "text" (stored in metadata)
# - metadata defaults to {} (empty dict, but context fields like user_id are auto-added)
# - timestamp is automatically set to current time
memory.add(
    content="I prefer Python over JavaScript",
    role="user",              # Optional, defaults to "user"
    metadata={"type": "preference"}  # Optional, defaults to {}
)

memory.add(
    content="I understand you prefer Python",
    role="assistant"  # Optional, defaults to "user"
)

# Search memories
# Note: Default values when not specified:
# - limit defaults to 10 (used as top_k)
# - filter_dict defaults to None (but context filters from user_id/agent_id/session_id are auto-applied)
# - types defaults to both episodic and semantic memory
results = memory.search(
    query="What are the user's preferences?",
    limit=10  # Optional, defaults to 10
)

# Results structure:
# {
#     "episodic_memory": [...],      # List of episodic memory results
#     "episode_summary": [...],      # Summaries of episodes
#     "semantic_memory": [...]       # List of semantic memory results
# }
```

### Memory Operations

```python
# Delete episodic memories
memory.delete_episodic(episodic_id="episode_123")
# Or delete multiple
memory.delete_episodic(episodic_ids=["ep1", "ep2"])

# Delete semantic memories
memory.delete_semantic(semantic_id="semantic_123")
# Or delete multiple
memory.delete_semantic(semantic_ids=["sem1", "sem2"])

# Get current context
context = memory.get_context()
# Returns: {"org_id": "...", "project_id": "...", "user_id": "...", ...}
```

## REST API Usage Examples

The following examples demonstrate how to use the MemMachine REST API v2 with `curl` commands. All examples assume the server is running at `http://localhost:8080`.

### Project Management

#### List Projects

List all projects in an organization:

```bash
curl -X POST "http://localhost:8080/api/v2/projects/list" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Response**: Returns a list of all projects with their configurations.

#### Create a Project

Create a new project. Required fields: `org_id`, `project_id`. Optional fields: `description` (defaults to `""`), `config.embedder` (defaults to `""`, uses server default), `config.reranker` (defaults to `""`, uses server default).

```bash
curl -X POST "http://localhost:8080/api/v2/projects" \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "my-org",
    "project_id": "my-project",
    "description": "My project description",
    "config": {
      "embedder": "my-openai-embedder",
      "reranker": "my-rrf-reranker"
    }
  }'
```

**Response**: Returns the created project with its configuration.

#### Get Project Information

Retrieve information about an existing project:

```bash
curl -X POST "http://localhost:8080/api/v2/projects/get" \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "my-org",
    "project_id": "my-project"
  }'
```

**Response**: Returns the project details including description and configuration.

#### Delete a Project

Delete a project and all its associated memories:

```bash
curl -X POST "http://localhost:8080/api/v2/projects/delete" \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "my-org",
    "project_id": "my-project"
  }'
```

**Response**: Returns 204 No Content on success.

### Memory Operations

#### Add Memories

```bash
# Required fields: org_id, project_id, messages[].content
# Optional message fields:
# - producer (defaults to "user")
# - produced_for (defaults to "")
# - role (defaults to "")
# - timestamp (defaults to current time in UTC)
# - metadata (defaults to {})
# Optional request fields:
# - types (defaults to [] which means all memory types)
curl -X POST "http://localhost:8080/api/v2/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "my-org",
    "project_id": "my-project",
    "messages": [
      {
        "content": "I like pizza",
        "producer": "user123",
        "produced_for": "agent456",
        "role": "user",
        "timestamp": "2024-01-15T10:00:00Z",
        "metadata": {
          "type": "preference",
          "topic": "food"
        }
      }
    ]
  }'
```

#### Search Memories

Search for memories across episodic and semantic memory. Required fields: `org_id`, `project_id`, `query`. Optional fields: `top_k` (defaults to `10`), `filter` (defaults to `""`), `types` (defaults to `[]` which means all memory types).

```bash
curl -X POST "http://localhost:8080/api/v2/memories/search" \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "my-org",
    "project_id": "my-project",
    "query": "What does the user like?",
    "top_k": 5,
    "types": ["episodic", "semantic"]
  }'
```

**Response**: Returns search results with `episodic_memory` and `semantic_memory` arrays.

**Example with filter**:

```bash
curl -X POST "http://localhost:8080/api/v2/memories/search" \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "my-org",
    "project_id": "my-project",
    "query": "user preferences",
    "top_k": 10,
    "filter": "metadata.user_id='user123'",
    "types": ["semantic"]
  }'
```

#### Delete Episodic Memories

Delete specific episodic memories by ID:

```bash
# Delete a single episodic memory
curl -X POST "http://localhost:8080/api/v2/memories/episodic/delete" \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "my-org",
    "project_id": "my-project",
    "episodic_id": "episode_123"
  }'
```

**Delete multiple episodic memories**:

```bash
curl -X POST "http://localhost:8080/api/v2/memories/episodic/delete" \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "my-org",
    "project_id": "my-project",
    "episodic_ids": ["episode_123", "episode_456", "episode_789"]
  }'
```

**Response**: Returns 204 No Content on success.

#### Delete Semantic Memories

Delete specific semantic memories by ID:

```bash
# Delete a single semantic memory
curl -X POST "http://localhost:8080/api/v2/memories/semantic/delete" \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "my-org",
    "project_id": "my-project",
    "semantic_id": "semantic_123"
  }'
```

**Delete multiple semantic memories**:

```bash
curl -X POST "http://localhost:8080/api/v2/memories/semantic/delete" \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "my-org",
    "project_id": "my-project",
    "semantic_ids": ["semantic_123", "semantic_456"]
  }'
```

**Response**: Returns 204 No Content on success.

### Utility Endpoints

#### Health Check

Check if the MemMachine server is running and healthy:

```bash
curl -X GET "http://localhost:8080/health"
```

**Response**: Returns server health status.

#### Metrics

Get Prometheus metrics (useful for monitoring):

```bash
curl -X GET "http://localhost:8080/metrics"
```

**Response**: Returns Prometheus-formatted metrics.

## Common Integration Patterns

### Pattern 1: Chatbot with Memory

```python
from memmachine import MemMachineClient

client = MemMachineClient(base_url="http://localhost:8080")
project = client.get_project(org_id="my-org", project_id="chatbot-project")

def handle_user_message(user_id: str, session_id: str, message: str):
    # Create memory context for this user/session
    memory = project.memory(user_id=user_id, session_id=session_id)
    
    # Store user message
    memory.add(content=message, role="user")
    
    # Search for relevant context
    context = memory.search(query=message, limit=5)
    
    # Build prompt with context
    prompt = build_prompt_with_context(message, context)
    
    # Get AI response (using your LLM)
    response = get_llm_response(prompt)
    
    # Store assistant response
    memory.add(content=response, role="assistant")
    
    return response
```

### Pattern 2: Multi-User Group Chat

```python
def handle_group_message(group_id: str, user_id: str, message: str):
    memory = project.memory(
        group_id=group_id,
        user_id=user_id,
        session_id=f"group-{group_id}"
    )
    
    # Store message
    memory.add(content=message, role="user")
    
    # Search for group context
    context = memory.search(query=message, limit=10)
    
    # Process with context...
    return process_group_message(message, context)
```

### Pattern 3: User Preference Learning

```python
def learn_user_preference(user_id: str, preference: str, category: str):
    memory = project.memory(user_id=user_id)
    
    # Store preference as semantic memory
    memory.add(
        content=preference,
        role="user",
        episode_type="semantic",  # Store in semantic memory
        metadata={"category": category, "type": "preference"}
    )

def get_user_preferences(user_id: str, category: str = None):
    memory = project.memory(user_id=user_id)
    
    query = f"user preferences"
    if category:
        query += f" about {category}"
    
    results = memory.search(query=query, limit=20)
    return results["semantic_memory"]
```

## Data Types and Structures

### Message Structure

```python
{
    "content": str,              # Required: The message content
    "producer": str,             # Optional: Who created this (defaults to "user")
    "produced_for": str,         # Optional: Who this is for (defaults to "")
    "role": str,                 # Optional: "user", "assistant", or "system" (defaults to "")
    "timestamp": str,            # Optional: ISO 8601 timestamp (defaults to current time in UTC)
    "metadata": dict             # Optional: Additional metadata (defaults to {})
}
```

### Search Results Structure

```python
{
    "episodic_memory": [         # List of episodic memory results
        {
            "content": str,
            "producer": str,
            "produced_for": str,
            "role": str,
            "timestamp": str,
            "metadata": dict,
            # ... other fields
        }
    ],
    "episode_summary": [         # Summaries of episodes
        {
            "content": str,
            # ... other fields
        }
    ],
    "semantic_memory": [         # List of semantic memory results
        {
            "content": str,
            "metadata": dict,
            # ... other fields
        }
    ]
}
```

## Error Handling

Common HTTP status codes:

- **200 OK**: Request successful
- **201 Created**: Resource created successfully
- **400 Bad Request**: Invalid request data or missing required fields
- **404 Not Found**: Resource not found (e.g., project doesn't exist)
- **422 Unprocessable Entity**: Validation error
- **500 Internal Server Error**: Server-side error
- **503 Service Unavailable**: Service not ready (check `/health` endpoint)

Always check response status and handle errors appropriately:

```python
import requests

try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raises exception for 4xx/5xx status codes
    return response.json()
except requests.HTTPError as e:
    if e.response.status_code == 404:
        # Handle not found
        pass
    elif e.response.status_code == 400:
        # Handle bad request
        pass
    else:
        # Handle other errors
        pass
```

## Default Values Summary

### Project Creation
- `description`: Defaults to `""` (empty string)
- `config.embedder`: Defaults to `""` (uses server's configured default embedder)
- `config.reranker`: Defaults to `""` (uses server's configured default reranker)

### Memory.add() (Python SDK)
- `role`: Defaults to `"user"`
- `producer`: Defaults to `user_id` (if set on Memory instance) or `"user"`
- `produced_for`: Defaults to `agent_id` (if set on Memory instance) or `"agent"`
- `episode_type`: Defaults to `"text"` (stored in metadata)
- `metadata`: Defaults to `{}` (empty dict, but context fields like `user_id`, `agent_id`, `session_id` are automatically added)
- `timestamp`: Automatically set to current time (not a parameter)

### Memory.search() (Python SDK)
- `limit`: Defaults to `10` (used as `top_k` in API)
- `filter_dict`: Defaults to `None`, but context filters from `user_id`, `agent_id`, and `session_id` are automatically applied
- `types`: Defaults to both `[MemoryType.Episodic, MemoryType.Semantic]`

### REST API v2 Defaults
- **Add Memories**: `producer` defaults to `"user"`, `produced_for` defaults to `""`, `role` defaults to `""`, `timestamp` defaults to current time, `metadata` defaults to `{}`, `types` defaults to `[]` (all memory types)
- **Search Memories**: `top_k` defaults to `10`, `filter` defaults to `""`, `types` defaults to `[]` (all memory types)

## Best Practices

1. **Always use projects**: The v2 API requires `org_id` and `project_id`. Use projects to isolate different applications or environments.

2. **Set explicit user IDs**: Always provide explicit `user_id` values for proper memory isolation. Never rely on defaults in production.

3. **Use appropriate roles**: In the Python SDK, `role` defaults to `"user"`. Use `role="user"` for user messages, `role="assistant"` for AI responses, and `role="system"` for system messages. Note: In the REST API, `role` defaults to an empty string, so it's recommended to always specify it.

4. **Search before responding**: Query memory for relevant context before generating responses to provide personalized, context-aware answers.

5. **Store important information**: Store user preferences, facts, and important conversation points to build up a knowledge base over time.

6. **Use metadata effectively**: Add relevant metadata to memories for better filtering and organization:
   ```python
   memory.add(
       content="I work at Acme Corp",
       role="user",
       metadata={"type": "fact", "category": "employment", "verified": True}
   )
   ```

7. **Monitor health**: Use the `/health` endpoint to check server status before making requests.

8. **Handle timeouts**: Set appropriate timeouts for API calls, especially for search operations which may take longer.

## Configuration

MemMachine is configured via a YAML file. The file can be named `cfg.yml` (recommended in documentation) or `configuration.yml` (used in Docker setups). Key sections include:

- **`resources.databases`**: Database connections (Neo4j for episodic, PostgreSQL for semantic)
- **`resources.embedders`**: Embedding model configurations (OpenAI, AWS Bedrock, Ollama, etc.)
- **`resources.language_models`**: LLM configurations for summarization and extraction
- **`episodic_memory`**: Episodic memory settings (short-term and long-term)
- **`semantic_memory`**: Semantic memory settings

Users typically configure this file when setting up their server. The Docker compose script can help generate it.

## Examples and Resources

- **Documentation**: [https://docs.memmachine.ai](https://docs.memmachine.ai)
- **API Reference**: [https://docs.memmachine.ai/api_reference](https://docs.memmachine.ai/api_reference)
- **Examples Directory**: Check the `examples/` directory in the repository for complete integration examples:
  - `examples/crm/` - CRM agent with memory
  - `examples/financial_analyst/` - Financial analysis agent
  - `examples/health_assistant/` - Healthcare assistant
  - `examples/writing_assistant/` - Writing assistant
  - `examples/frontend/` - Web frontend integration

## Summary

When helping users integrate MemMachine:

1. **Installation**: Guide users to install via `pip install memmachine` or use Docker with `./memmachine-compose.sh`
2. **Server Setup**: Ensure the MemMachine server is running (default: `http://localhost:8080`)
3. **Project Creation**: Help users create projects using `client.create_project(org_id, project_id)`
4. **Memory Operations**: Show how to use `memory.add()` to store and `memory.search()` to retrieve context
5. **Context Management**: Explain how to use `user_id`, `agent_id`, `session_id` for proper memory isolation
6. **Error Handling**: Implement proper error handling for API calls
7. **Best Practices**: Encourage searching before responding and storing important information

MemMachine enables AI applications to remember past interactions, learn user preferences, and provide personalized, context-aware responses across sessions.

