# MemMachine Extension Server

A FastAPI-based MCP (Model Context Protocol) server that provides memory management capabilities for MemMachine integration with Cursor editor, web applications, and AI models.

## ✨ Features

- **Memory Management**: Add, search, and delete memories with full CRUD operations
- **User Management**: User authentication and token-based access control
- **Authentication**: Token-based authentication with secure token management
- **Multi-Protocol Support**: HTTP REST API and MCP protocol for AI model integration
- **Database Integration**: SQLite database for user and token management
- **Modular Architecture**: Clean separation of concerns with reusable components

## 🏗️ Architecture

The server follows a modular architecture with clear separation of concerns:

### Core Components
- **FastAPI Application**: Main web framework with automatic OpenAPI documentation
- **MCP Integration**: Model Context Protocol support for AI model communication
- **Authentication System**: Token-based authentication with password hashing
- **Database Layer**: SQLAlchemy ORM with SQLite backend
- **Memory Client**: HTTP client for MemMachine service communication

## 📁 Project Structure

```
server/
├── main.py                         # Application entry point
├── app.py                          # FastAPI application factory
├── settings.py                     # Configuration and environment variables
├── api/                            # REST API endpoints
│   ├── auth.py                     # Authentication endpoints
│   ├── memory.py                   # Memory operation endpoints
│   └── misc.py                     # Miscellaneous endpoints
├── auth/                           # Authentication modules
│   ├── jwt.py                      # JWT token utilities
│   ├── password.py                 # Password hashing utilities
│   └── token.py                    # Token management
├── core/                           # Core business logic
│   ├── constants.py                # Application constants
│   ├── database.py                 # Database configuration
│   ├── formatter.py                # Memory formatting utilities
│   ├── handlers.py                 # Memory operation handlers
│   └── mm_client.py               # MemMachine HTTP client
├── mcp/                            # MCP protocol implementation
│   ├── app.py                      # MCP application setup
│   └── memory.py                  # MCP memory tools
├── middleware/                     # FastAPI middleware
│   ├── authentication.py          # Authentication middleware
│   └── logging.py                 # Logging middleware
├── models/                         # Database models
│   ├── token.py                   # Token database model
│   └── user.py                    # User database model
├── schemas/                        # Pydantic schemas
│   ├── auth.py                    # Authentication schemas
│   ├── base.py                    # Base schemas
│   └── memory.py                 # Memory schemas
└── services/                       # Business services
    └── user.py                    # User service
```

## 🚀 Prerequisites

1. **MemMachine Service**: Ensure MemMachine is running (default: `http://localhost:8080`)
2. **Python 3.10+**: Required for modern Python features
3. **Virtual Environment**: Recommended for dependency isolation

## 🛠️ Setup

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

### Option 3: Docker setup
```bash
# From project root
docker-compose up -d cursor-mcp-server
```

## ⚙️ Configuration

The server uses Pydantic Settings for configuration management. All settings can be configured via environment variables:

### Core Settings
```bash
# MemMachine Backend
MM_BACKEND_URL=http://localhost:8080        # MemMachine service URL
SERVER_PORT=8001                           # Server port
OAUTH_DOMAIN=https://api.example.com       # OAuth resource metadata domain

# Database
DATABASE_URL=sqlite:///./bigmemory.db       # Database connection URL

# SSL & Security
VERIFY_SSL=false                           # SSL certificate verification

# Debug & Development
DEBUG=false                                # Enable debug mode
REQUEST_TIMEOUT=30                         # HTTP request timeout
```

### Memory Configuration
```bash
TOKEN_TTL_SECONDS=604800                  # Token TTL (7 days)
```

**Note**: `DEFAULT_PRODUCED_FOR`, `DEFAULT_EPISODE_TYPE`, and `DEFAULT_SESSION_ID` are now constants defined in `core/constants.py` and are not configurable.

### Authentication Configuration
```bash
# Token settings
TOKEN_TTL_SECONDS=604800                  # Token time-to-live (7 days)
```

## 🚀 Running the Server

### Development Mode
```bash
cd server
source .venv/bin/activate  # or use ./activate_venv.sh
python -m server.main
```

### Production Mode
```bash
# Using uvicorn directly
uvicorn server.main:app --host 0.0.0.0 --port 8001

# Using Docker
docker-compose up -d cursor-mcp-server
```

The server will start on `http://localhost:8001` by default.

### Health Check
```bash
curl http://localhost:8001/api/health
```

## 📚 API Documentation

### REST API Endpoints

#### Authentication Endpoints (`/api/auth/`)
- `POST /api/auth/login` - User login with email/password
- `POST /api/auth/register` - User registration

#### Memory Endpoints (`/api/memory/`)
- `GET /api/memory/search` - Search memories with query parameters
- `POST /api/memory/add` - Add new memory episode
- `DELETE /api/memory/delete` - Delete memories
- `GET /api/memory/profile` - Get profile memories
- `GET /api/memory/episodic` - Get episodic memories

#### Miscellaneous Endpoints (`/api/`)
- `GET /api/health` - Health check endpoint

### OpenAPI Documentation
Once the server is running, visit:
- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`
- **OpenAPI JSON**: `http://localhost:8001/openapi.json`

## 🔧 MCP Tools

The server provides MCP (Model Context Protocol) tools for AI model integration. All tools require authentication and follow the MCP specification.

### Authentication
All MCP tool requests require authentication via Bearer token:
```bash
Authorization: Bearer <your-token>
```

**Getting a Token:**
```bash
# Register a new user
curl -X POST "http://localhost:8001/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "your-password"}'

# Login (returns a token)
curl -X POST "http://localhost:8001/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "your-password"}'
```

The login endpoint returns a token that can be used for subsequent API calls. Tokens have a configurable TTL (time-to-live) and are stored securely in the database.

### 1. `mcp_add_memory`
Add a new memory episode to MemMachine. This method is called whenever the user informs anything about themselves, their preferences, or anything that has relevant information which can be useful in future conversations.

**Parameters:**
- `content` (string, required): The content to store in memory

**Returns:**
Simple confirmation message string

**Example:**
```bash
curl -X POST "http://localhost:8001/mcp" \
  -H "Authorization: Bearer your-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "mcp_add_memory",
    "params": {
      "content": "I prefer working in the morning"
    }
  }'
```

### 2. `mcp_search_memory`
Search for memories in MemMachine. This function should be invoked to find relevant context, previous conversations, user preferences, or important information stored in memory.

**Parameters:**
- `query` (string, required): The raw content that user input
- `limit` (int, required): Maximum number of results to return (recommended: 10)

**Returns:**
Formatted text string in simple markdown format with episodic and profile memories

**Example:**
```bash
curl -X POST "http://localhost:8001/mcp" \
  -H "Authorization: Bearer your-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "mcp_search_memory",
    "params": {
      "query": "work preferences",
      "limit": 10
    }
  }'
```

### 3. `mcp_delete_episodic_memory`
Delete all episodic memories for the current user.

**Parameters:**
None

**Returns:**
Simple confirmation message string

**Example:**
```bash
curl -X POST "http://localhost:8001/mcp" \
  -H "Authorization: Bearer your-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "mcp_delete_episodic_memory",
    "params": {}
  }'
```

### 4. `mcp_get_profile_memory`
Get the profile memory for the current user. This function retrieves user profile information stored in MemMachine, including user preferences, facts, and personalized data that persists across multiple interactions.

**Parameters:**
- `limit` (int, required): Maximum number of profile memories to return

**Returns:**
Formatted text string with profile memories in simple markdown format

**Example:**
```bash
curl -X POST "http://localhost:8001/mcp" \
  -H "Authorization: Bearer your-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "mcp_get_profile_memory",
    "params": {
      "limit": 10
    }
  }'
```

### 5. `mcp_get_episodic_memory`
Get the episodic memory for the current user. This function retrieves conversation episodes and contextual memories stored in MemMachine, including recent interactions and user-specific context.

**Parameters:**
- `limit` (int, required): Maximum number of episodic memories to return

**Returns:**
Formatted text string with episodic memories in simple markdown format

**Example:**
```bash
curl -X POST "http://localhost:8001/mcp" \
  -H "Authorization: Bearer your-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "mcp_get_episodic_memory",
    "params": {
      "limit": 10
    }
  }'
```

## 🧪 Development & Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=server --cov-report=html

# Run specific test file
python -m pytest tests/test_server.py -v
```

### Development Setup
```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run in development mode with auto-reload
uvicorn server.main:app --reload --host 0.0.0.0 --port 8001
```

### Code Quality
```bash
# Format code
black server/

# Lint code
flake8 server/

# Type checking
mypy server/
```

## 📝 License

This MCP server follows the same license as the MemMachine project.
