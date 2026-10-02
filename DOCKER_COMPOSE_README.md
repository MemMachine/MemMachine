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

`COMPOSE_PROFILES` in `.env` selects the long-term memory backend, and with it
which vector store container starts. Exactly one runs:

| `COMPOSE_PROFILES` | Long-term memory backend | Store started |
|---|---|---|
| `event` (default) | event memory (`vector_store` + `segment_store`) | Qdrant |
| `declarative` | declarative memory (`vector_graph_store`) | Neo4j |

The choice must match `episodic_memory.long_term_memory` in `configuration.yml`.

### 2. Configure MemMachine
`./memmachine-compose.sh` generates a `configuration.yml` wired for the selected
backend, so with Option A below you can skip this step.

To write it by hand for the default `event` backend, start from the compose-ready
sample; it already uses the compose service names and reads secrets from `.env`:
```bash
cp sample_configs/configuration.event.yml configuration.yml
```

For the `declarative` backend, start from `sample_configs/episodic_memory_config.cpu.sample`
(or `.gpu.sample`) and update:
- Replace `<YOUR_API_KEY>` with your OpenAI API key
- Change database hosts from `localhost` to the service names `postgres` and `neo4j`
- Under `profile_storage` (PostgreSQL), match `POSTGRES_*` in `.env`: set
  `user` and `db_name` to `memmachine` (or your `POSTGRES_USER` and
  `POSTGRES_DB`) and `password` to `$POSTGRES_PASSWORD`
- Under the Neo4j store, set `user` to your `NEO4J_USER` (default `neo4j`) and
  `password` to `$NEO4J_PASSWORD`

Only `password` and `api_key` read `$ENV_NAME` values from `.env`; `user` and
`db_name` must be written out literally.

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
- ✅ Pull and start all services (PostgreSQL, Qdrant or Neo4j per `COMPOSE_PROFILES`, MemMachine)
- ✅ Wait for all services to be healthy
- ✅ Display service URLs and connection info

#### Option B: Using Docker Compose Directly
```bash
docker-compose up -d
```

### 4. Access Services
Once started, you can access:

- **MemMachine API**: http://localhost:8080
- **Qdrant Dashboard** (`event`): http://localhost:6333/dashboard
- **Neo4j Browser** (`declarative`): http://localhost:7474
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

- **PostgreSQL** (port 5432): Episodes, sessions, and semantic memory with pgvector; also the segment store for the `event` backend
- **Qdrant** (ports 6333 REST, 6334 gRPC; `event` profile): Vector store for event long-term memory
- **Neo4j** (ports 7474, 7687; `declarative` profile): Vector graph store for declarative long-term memory
- **MemMachine** (port 8080): Main API server (uses pre-built `memmachine/memmachine` image)

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
- Database hostnames (use service names: `postgres`, `qdrant`, `neo4j`)
- Database ports (5432 for PostgreSQL, 6333/6334 for Qdrant, 7687 for Neo4j)
- Database credentials (usernames and passwords)
- Database names

**2. Configuration.yml Setup**
The `configuration.yml` file contains MemMachine-specific settings:
- **Model configuration**: OpenAI API settings for LLM and embeddings
- **Storage configuration**: PostgreSQL plus Qdrant (`event`) or Neo4j (`declarative`) connection details
- **Memory settings**: Session memory capacity and limits
- **Reranker configuration**: Search and ranking algorithms

**Key settings to update in configuration.yml:**
- Replace `<YOUR_API_KEY>` with your OpenAI API key (appears in both Model and embedder sections)
- For `declarative`, replace `<YOUR_PASSWORD_HERE>` with your Neo4j password
- Set database hosts to the service names (`postgres`, `qdrant`, `neo4j`), not `localhost`; inside a container `localhost` is the container itself

This ensures MemMachine can properly connect to the Docker services and use your OpenAI API key for embeddings and LLM operations.
