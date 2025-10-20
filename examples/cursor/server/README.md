# Cursor MCP Server for MemMachine

This MCP (Model Context Protocol) server provides memory CRUD operations to Cursor, enabling seamless integration with MemMachine's memory system.

## Features

- **Memory Management**: Add, search, and delete memories
- **Session Management**: Automatic session ID generation and user session tracking
- **HTTP Communication**: Communicates with MemMachine service via HTTP requests

## Prerequisites

1. **MemMachine Service**: Ensure MemMachine is running on `http://localhost:8080` (or configure `MEMORY_BACKEND_URL`)
2. **Python Dependencies**: The project includes a virtual environment setup


## File Structure

The original monolithic `cursor_mcp_server.py` (1309 lines) has been refactored into the following structure:

```
cursor_mcp/
├── scripts/                      
|   ├── start_server.sh             # Startup script
|   ├── activate_venv.sh            # Virtual environment setup
|-- tests/                          
|   |-- test_mcp_server.py
|   |-- test_server.py
├── server/                         # Server package
│   ├── __init__.py                 # Package initialization
│   ├── main.py                     # Entry point (run with python -m server.main)
│   ├── config.py                   # Configuration and constants
│   ├── models.py                   # Pydantic data models
│   ├── client.py                   # MemMachine HTTP client
│   ├── auth.py                     # Authentication and token management
│   ├── handlers.py                 # Memory operation handlers and utilities
│   ├── middleware.py               # FastAPI middleware
│   ├── mcp_tools.py                # MCP tool functions
│   └── endpoints.py                # REST API endpoints
└── cursor_mcp_server.py            # Original file (preserved)
```

## Setup

### Option 1: Using the activation script (Recommended)
```bash
cd server
./scripts/activate_venv.sh
```

### Option 2: Manual setup
```bash
cd server
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Configuration

Set environment variables (optional):

```bash
export MEMORY_BACKEND_URL="http://localhost:8080"  # MemMachine service URL
export CURSOR_MCP_PORT="8001"                      # MCP server port
```

## Running the Server

Make sure you have activated the virtual environment first:

```bash
cd server
source .venv/bin/activate  # or use ./activate_venv.sh
python -m server.main
```

The server will start on `http://localhost:8001` by default.

## Available MCP Tools

All tools include detailed parameter descriptions and use header-based session management following the MCP specification.

### Request Headers
All requests must include the following headers:
- `User-ID`: The user ID for the session
- `Mcp-Session-Id`: The session ID (optional for initialization, required for subsequent requests)

### 1. `add_memory`
Add a new memory episode to MemMachine.

**Parameters:**
- `content` (string, required): The memory content to store
- `metadata` (dict, optional): Additional metadata for the memory

**Headers:**
- `User-ID`: The user ID for the session
- `Mcp-Session-Id`: The session ID (optional, will be generated if not provided)

**Example:**
```bash
curl -X POST "http://localhost:8001/mcp/add_memory" \
  -H "User-ID: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "I prefer working in the morning",
    "metadata": {"category": "preference"}
  }'
```

### 2. `search_memory`
Search for memories in MemMachine.

**Parameters:**
- `query` (string, required): The search query
- `limit` (int, optional): Maximum number of results (default: 5)

**Headers:**
- `User-ID`: The user ID for the session
- `Mcp-Session-Id`: The session ID (optional, will be generated if not provided)

**Example:**
```bash
curl -X POST "http://localhost:8001/mcp/search_memory" \
  -H "User-ID: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "work preferences",
    "limit": 10
  }'
```

### 3. `delete_session_memory`
Delete all memories for the current session.

**Parameters:**
- No parameters required (session info extracted from headers)

**Headers:**
- `User-ID`: The user ID for the session
- `Mcp-Session-Id`: The session ID to delete

**Example:**
```bash
curl -X POST "http://localhost:8001/mcp/delete_session_memory" \
  -H "User-ID: user123" \
  -H "Mcp-Session-Id: cursor-session-user123-1703123456789-abc123def" \
  -H "Content-Type: application/json"
```

## Development

To extend the server:

1. **Add New Tools**: Use the `@mcp.tool()` decorator
2. **Modify Constants**: Update the constant values at the top of the file
3. **Add Validation**: Extend the Pydantic models for additional validation
4. **Custom Error Handling**: Add specific error handling for new operations

## License

This MCP server follows the same license as the MemMachine project.


