# Docker Setup for Cursor MCP Server and Web Application

This document explains how to run both the Cursor MCP Server and Web Application using Docker.

## Quick Start

1. **Build and run with Docker Compose (Recommended):**
   ```bash
   docker-compose up --build
   ```
   
   This will start both services:
   - **MCP Server**: Available at `http://localhost:8001` (internal only)
   - **Web Application**: Available at `http://localhost:3000`

2. **Or build and run manually:**
   ```bash
   # Build the image
   docker build -t cursor-mcp-server .
   
   # Run the container with named volume
   docker run -p 8001:8001 \
     -v mcp_data:/app/data \
     -e MEMORY_BACKEND_URL=http://host.docker.internal:8080 \
     cursor-mcp-server
   ```

## Services Architecture

The Docker setup includes two independent services:

### 🔧 **cursor-mcp-server** (Backend)
- **Port**: 8001 (internal only)
- **Purpose**: MCP server for Cursor integration with MemMachine
- **Database**: SQLite stored in `./data/bigmemory.db`
- **Health Check**: `http://localhost:8001/api/health`

### 🌐 **web** (Frontend)
- **Port**: 3000 (external access)
- **Purpose**: React web application with nginx
- **Proxy**: Automatically proxies `/api/*` and `/mcp` requests to the MCP server
- **Health Check**: `http://localhost:3000/`

### 🔗 **Service Communication**
- Services communicate via Docker network (`cursor-network`)
- Web app proxies API calls to the MCP server automatically
- No manual configuration needed for service discovery

## Environment Variables

The following environment variables can be configured:

### Required
- `MEMORY_BACKEND_URL`: URL of the MemMachine backend service
  - For Docker: `http://host.docker.internal:8080` (to access host services)
  - For production: `http://your-memmachine-service:8080`

### Optional
- `CURSOR_MCP_PORT`: Port for the MCP server (default: 8001)
- `DATABASE_URL`: Database connection string (default: `sqlite:////app/data/bigmemory.db`)
- `VERIFY_SSL`: SSL verification for backend requests (default: false)
- `DEBUG`: Enable debug mode (default: false)
- `TOKEN_TTL_SECONDS`: Token expiration time (default: 2419200)

## Docker Compose Usage

The `docker-compose.yml` file includes:
- Automatic volume mounting for data persistence
- Health checks
- Restart policy
- Proper environment variable handling

### Custom Environment File
Create a `.env` file with your configuration:
```bash
MEMORY_BACKEND_URL=http://host.docker.internal:8080
CURSOR_MCP_PORT=8001
DEBUG=false
```

Then run:
```bash
docker-compose up --build
```

## Production Deployment

For production deployment:

1. **Update the MemMachine backend URL** in `docker-compose.yml` or environment variables
2. **Consider using a managed database** instead of SQLite for better reliability
3. **Set up proper logging** and monitoring
4. **Configure authentication** if needed

### Example Production docker-compose.yml
```yaml
services:
  cursor-mcp-server:
    build: .
    environment:
      - MEMORY_BACKEND_URL=http://your-memmachine-service:8080
      - DATABASE_URL=postgresql://user:pass@db:5432/mcpdb
      - MCP_AUTH_TOKEN=${MCP_AUTH_TOKEN}
    volumes:
      - mcp_data:/app/data
    restart: unless-stopped
```

## Health Checks

The container includes health checks that verify the service is responding:
- Endpoint: `http://localhost:8001/api/health`
- Check interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3

## Data Persistence

The SQLite database is stored in `/app/data/bigmemory.db` inside the container and persisted using Docker named volumes (`mcp_data`). This provides:
- **Automatic persistence** between container restarts
- **Docker-managed storage** for reliability
- **Easy backup** using Docker volume commands

To backup the database:
```bash
# Create a backup
docker run --rm -v cursor_cursor-mcp-server_mcp_data:/data -v $(pwd):/backup alpine tar czf /backup/mcp-data-backup.tar.gz -C /data .

# Restore from backup
docker run --rm -v cursor_cursor-mcp-server_mcp_data:/data -v $(pwd):/backup alpine tar xzf /backup/mcp-data-backup.tar.gz -C /data
```

## Troubleshooting

1. **Container won't start:**
   - Check if the MemMachine backend is accessible
   - Verify environment variables are set correctly
   - Check Docker logs: `docker logs cursor-mcp-server`

2. **Health check fails:**
   - Ensure the service is starting properly
   - Check if the port is being exposed correctly
   - Verify the health endpoint is responding

3. **Cannot connect to MemMachine:**
   - For Docker Desktop: use `host.docker.internal` instead of `localhost`
   - For Linux: use `172.17.0.1` or the actual host IP
   - Ensure MemMachine service is running and accessible
