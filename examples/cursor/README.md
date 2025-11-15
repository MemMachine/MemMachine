# MemMachine Cursor Integration

A comprehensive integration suite for MemMachine with Cursor editor, providing memory management capabilities through multiple interfaces: VS Code/Cursor extension, web application, and MCP (Model Context Protocol) server.

## 🏗️ Architecture Overview

This project consists of three main components:

### 1. **Cursor/VS Code Extension** (`extension/`)
- Native Cursor/VS Code extension for seamless memory management
- Tree view panels for episodic and profile memories
- Real-time memory synchronization
- Authentication token management

### 2. **MCP Server** (`server/`)
- FastAPI-based MCP server for AI model integration
- RESTful API endpoints for memory operations
- SQLite database for session management
- Modular architecture with authentication

### 3. **Web Application** (`web/`)
- React-based web interface for memory management
- Modern UI with Tailwind CSS and Radix UI components
- User authentication and session management
- Real-time memory visualization

## ✨ Features

- **Memory Management**: Add, search, and delete memories across all interfaces
- **Session Management**: Automatic session ID generation and user session tracking
- **Multi-Interface Support**: Extension, web app, and MCP server
- **Authentication**: JWT-based authentication with secure token management
- **Real-time Sync**: Memory updates sync across all interfaces
- **Docker Support**: Complete containerized deployment
- **Modular Architecture**: Clean separation of concerns with reusable components

## 🚀 Quick Start

### Prerequisites
- **MemMachine Service**: Ensure MemMachine is running (default: `http://localhost:8080`)
- **Node.js**: v18+ for extension and web app development
- **Python**: 3.10+ for MCP server
- **Docker**: For containerized deployment (optional)

### Option 1: Docker Deployment (Recommended)
```bash
# Clone and start all services
git clone <repository-url>
cd cursor
docker-compose up -d

# Services will be available at:
# - MCP Server: http://localhost:8001
# - Web App: http://localhost:3000
```

### Option 2: Development Setup
```bash
# 1. Start MCP Server
cd server
./scripts/activate_venv.sh
python -m server.main

# 2. Build and install extension
cd extension
pnpm install
pnpm run package
# Install the generated .vsix file in Cursor/VS Code

# 3. Start web application
cd web
pnpm install
pnpm run dev
```

## 📁 Project Structure

```
cursor/
├── extension/                 # Cursor/VS Code Extension
│   ├── src/                   # TypeScript source code
│   ├── dist/                  # Compiled extension
│   └── package.json           # Extension manifest
├── server/                    # MCP Server (Python/FastAPI)
│   ├── api/                   # REST API endpoints
│   ├── auth/                  # Authentication modules
│   ├── core/                  # Core business logic
│   ├── mcp/                   # MCP protocol handlers
│   ├── models/                # Database models
│   ├── schemas/               # Pydantic schemas
│   └── services/              # Business services
├── web/                       # React Web Application
│   ├── src/                   # React source code
│   ├── public/                # Static assets
│   └── dist/                  # Built web app
├── tests/                     # Test suites
├── docker-compose.yml         # Docker orchestration
└── Dockerfile                 # Server containerization
```

## 🔧 Configuration

### Environment Variables
```bash
# MCP Server Configuration
MEMORY_BACKEND_URL=http://localhost:8080  # MemMachine service URL
CURSOR_MCP_PORT=8001                      # MCP server port
DATABASE_URL=sqlite:///data/bigmemory.db  # Database URL
VERIFY_SSL=false                          # SSL verification
DEBUG=false                               # Debug mode
```

### Extension Configuration
Configure in Cursor/VS Code settings:
- `memmachine.authToken`: Your authentication token
- `memmachine.apiBaseUrl`: API base URL (default: `http://localhost:8001/api`)

## 🔌 API Reference

### REST API Endpoints

#### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/register` - User registration
- `POST /api/auth/refresh` - Refresh authentication token

#### Memory Operations
- `GET /api/memory/search` - Search memories
- `POST /api/memory/add` - Add new memory
- `DELETE /api/memory/session/{session_id}` - Delete session memories
- `GET /api/memory/profile` - Get profile memories
- `GET /api/memory/episodic` - Get episodic memories

### MCP Tools

The MCP server provides the following tools for AI model integration:

#### 1. `mcp_add_memory`
Add a new memory episode to MemMachine.

**Parameters:**
- `content` (string, required): The content to store in memory
- `session_id` (string, required): Memory session identifier

**Example:**
```json
{
  "method": "mcp_add_memory",
  "params": {
    "content": "User prefers working in the morning",
    "session_id": "PROJECT-my-workspace"
  }
}
```

#### 2. `mcp_search_memory`
Search for memories in MemMachine.

**Parameters:**
- `query` (string, required): Search query
- `limit` (int, required): Maximum results to return
- `session_id` (string, required): Session identifier

#### 3. `mcp_delete_session_memory`
Delete all memories for a specific session.

**Parameters:**
- `session_id` (string, required): Session to delete

#### 4. `mcp_get_profile_memory`
Get profile memories for the current session.

**Parameters:**
- `limit` (int, required): Maximum results to return
- `session_id` (string, required): Session identifier

#### 5. `mcp_get_episodic_memory`
Get episodic memories for the current session.

**Parameters:**
- `limit` (int, required): Maximum results to return
- `session_id` (string, required): Session identifier

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Individual Services
```bash
# Build and run MCP server
docker build -t cursor-mcp-server .
docker run -p 8001:8001 cursor-mcp-server

# Build and run web app
cd web
docker build -t cursor-web-app .
docker run -p 3000:80 cursor-web-app
```

## 🧪 Development

### Running Tests
```bash
# Server tests
cd server
python -m pytest tests/

# Web app tests
cd web
pnpm test

# Extension tests
cd extension
pnpm test
```

### Building for Production
```bash
# Extension
cd extension
pnpm run package

# Web app
cd web
pnpm run build

# Server (already production-ready)
cd server
python -m server.main
```

## 📝 License

This project follows the same license as the MemMachine project.
