# MemMachine Docker Setup Guide

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- OpenAI API key configured

### 1. Configure Environment
Copy the example environment file and add your OpenAI API key:
```bash
cp env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Start Services
Run the startup script:
```bash
./start-docker.sh
```

This will:
- ✅ Check Docker and Docker Compose availability
- ✅ Verify .env file and OpenAI API key
- ✅ Build and start all services (PostgreSQL, Neo4j 5.23, MemMachine)
- ✅ Wait for all services to be healthy
- ✅ Display service URLs and connection info

### 3. Access Services
Once started, you can access:

- **MemMachine API**: http://localhost:8080
- **Neo4j Browser**: http://localhost:7474
- **Health Check**: http://localhost:8080/health
- **Metrics**: http://localhost:8080/metrics

### 4. Test the Setup
```bash
# Test health endpoint
curl http://localhost:8080/health

# Test memory storage
curl -X POST "http://localhost:8080/v1/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "session": {
      "group_id": "test-group",
      "agent_id": ["test-agent"],
      "user_id": ["test-user"],
      "session_id": "test-session-123"
    },
    "producer": "test-user",
    "produced_for": "test-user",
    "episode_content": "Hello, this is a test message",
    "episode_type": "text",
    "metadata": {"test": true}
  }'
```

## Useful Commands

### View Logs
```bash
docker-compose logs -f
```

### Stop Services
```bash
docker-compose down
```

### Restart Services
```bash
docker-compose restart
```

### Clean Up (Remove All Data)
```bash
docker-compose down -v
```

## Services

- **PostgreSQL** (port 5432): Profile memory storage with pgvector
- **Neo4j** (ports 7474, 7687): Episodic memory with vector similarity
- **MemMachine** (port 8080): Main API server

## Configuration

Key files:
- `.env` - Environment variables
- `configuration.yml` - MemMachine configuration
- `docker-compose.yml` - Service definitions
- `start-docker.sh` - Startup script

### ⚠️ Important Configuration Note
**Make sure the database configuration details in `configuration.yml` match the database configuration details in `.env`**

Both files must have consistent:
- Database hostnames (use service names: `postgres`, `neo4j`)
- Database ports (5432 for PostgreSQL, 7687 for Neo4j)
- Database credentials (usernames and passwords)
- Database names

This ensures MemMachine can properly connect to the Docker services.
