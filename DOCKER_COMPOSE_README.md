# MemMachine Docker Setup Guide

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- OpenAI API key configured

### 1. Configure Environment
Copy the example environment file and add your OpenAI API key:
```bash
cp sample_configs/env.dockercompose .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Configure MemMachine
Copy the sample configuration file and update it with your settings:
```bash
cp sample_configs/episodic_memory_config.sample configuration.yml
# Edit configuration.yml and update:
# - Replace <YOUR_API_KEY> with your OpenAI API key
# - Replace <YOUR_PASSWORD_HERE> with your Neo4j password
# - Update host from 'localhost' to 'neo4j' for Docker environment
```

### 3. Start Services

#### Option A: Using the MemMachine Compose Script (Recommended)
Run the startup script:
```bash
./memmachine-compose.sh
```

This will:
- ✅ Check Docker and Docker Compose availability
- ✅ Verify .env file and OpenAI API key
- ✅ Check and create configuration.yml if needed
- ✅ Validate configuration settings
- ✅ Pull and start all services (PostgreSQL, Neo4j 5.23, MemMachine)
- ✅ Wait for all services to be healthy
- ✅ Display service URLs and connection info

#### Option B: Using Docker Compose Directly
```bash
docker-compose up -d
```

### 4. Access Services
Once started, you can access:

- **MemMachine API**: http://localhost:8080
- **Neo4j Browser**: http://localhost:7474
- **Health Check**: http://localhost:8080/health
- **Metrics**: http://localhost:8080/metrics

### 5. Test the Setup
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

### Using the MemMachine Compose Script (Recommended)

#### View Logs
```bash
./memmachine-compose.sh logs
```

#### Stop Services
```bash
./memmachine-compose.sh stop
```

#### Restart Services
```bash
./memmachine-compose.sh restart
```

#### Clean Up (Remove All Data)
```bash
./memmachine-compose.sh clean
```

#### Show Help
```bash
./memmachine-compose.sh help
```

### Using Docker Compose Directly

#### View Logs
```bash
docker-compose logs -f
```

#### Stop Services
```bash
docker-compose down
```

#### Restart Services
```bash
docker-compose restart
```

#### Clean Up (Remove All Data)
```bash
docker-compose down -v
```

## Services

- **PostgreSQL** (port 5432): Profile memory storage with pgvector
- **Neo4j** (ports 7474, 7687): Episodic memory with vector similarity
- **MemMachine** (port 8080): Main API server (uses pre-built `memmachine/memmachine` image)

### Apache AGE: single-Postgres alternative to Neo4j

MemMachine can run against Apache AGE (openCypher over PostgreSQL, Apache-2.0) instead of Neo4j. AGE's image ships both the ``age`` and ``vector`` extensions, so one Postgres instance backs both the graph store and semantic memory — the same way the helm chart's ``episodicBackend=age`` mode does.

**Option 1 (recommended): full AGE stack via `docker-compose.age.yml`**

Mirrors the helm single-stack experience. Use this file instead of ``docker-compose.yml`` (not alongside it — both define a ``memmachine`` service).

```bash
# 1. Copy the AGE sample config and edit it (set API key; change
#    host from ``localhost`` to ``postgres-age`` so MemMachine resolves
#    the service name over the compose network).
cp sample_configs/episodic_memory_config.age.sample configuration.yml

# 2. Bring the stack up — one database, one command.
docker compose -f docker-compose.age.yml up
```

**Option 2: AGE container + locally-installed server**

If you'd rather run the server outside Docker, start just the AGE service from the base file and point a pip-installed MemMachine at it.

```bash
# 1. Start just the AGE-enabled Postgres from the base compose file.
docker compose --profile age up postgres-age

# 2. Install the server and point it at postgres-age on localhost.
pip install memmachine
cp sample_configs/episodic_memory_config.age.sample configuration.yml
# Edit configuration.yml: keep host: localhost, set port to ${AGE_PORT:-5433},
# and replace placeholders with your API key / password.
memmachine-server --config configuration.yml
```

Both options use the same config shape:

```yaml
resources:
  databases:
    my_storage_id:
      provider: age
      config:
        host: postgres-age     # or localhost for Option 2
        port: 5432             # or ${AGE_PORT:-5433} for Option 2
        user: ${AGE_USER:-memmachine}
        password: ${AGE_PASSWORD:-memmachine_password}
        db_name: ${AGE_DB:-memmachine}
        graph_name: mem_graph
```

See ``sample_configs/episodic_memory_config.age.sample`` for a complete example including the pgvector-backed semantic-memory configuration that rides on the same Postgres.

## Configuration

Key files:
- `.env` - Environment variables
- `configuration.yml` - MemMachine configuration
- `docker-compose.yml` - Service definitions
- `memmachine-compose.sh` - Startup script with validation and health checks

### ⚠️ Important Configuration Notes

**1. Database Configuration Consistency**
Make sure the database configuration details in `configuration.yml` match the database configuration details in `.env`

Both files must have consistent:
- Database hostnames (use service names: `postgres`, `neo4j`)
- Database ports (5432 for PostgreSQL, 7687 for Neo4j)
- Database credentials (usernames and passwords)
- Database names

**2. Configuration.yml Setup**
The `configuration.yml` file contains MemMachine-specific settings:
- **Model configuration**: OpenAI API settings for LLM and embeddings
- **Storage configuration**: Neo4j connection details
- **Memory settings**: Session memory capacity and limits
- **Reranker configuration**: Search and ranking algorithms

**Key settings to update in configuration.yml:**
- Replace `<YOUR_API_KEY>` with your OpenAI API key (appears in both Model and embedder sections)
- Replace `<YOUR_PASSWORD_HERE>` with your Neo4j password
- Ensure the Neo4j host is set to `neo4j` (not `localhost`) for Docker environment

This ensures MemMachine can properly connect to the Docker services and use your OpenAI API key for embeddings and LLM operations.
