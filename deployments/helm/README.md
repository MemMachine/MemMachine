# MemMachine Helm Chart

Deploys MemMachine with optional in-cluster PostgreSQL (pgvector) and Neo4j. Both databases can be replaced with external instances via `postgres.enabled=false` / `neo4j.enabled=false`.

## Chart Info

| Field         | Value              |
|---------------|--------------------|
| Chart version | 0.1.0              |
| App version   | v0.1.1             |
| API version   | v2 (Helm 3)        |

---

## High-Level Architecture

```
  External client
        |
   NodePort (default: 31001)
        |
  memmachine-service          (NodePort, port 80 → pod 8080)
        |
  ┌─────────────────────┐
  │   MemMachine pod    │
  │   (port 8080)       │
  │                     │
  │  init: wait-for-postgres ──► postgres.host:postgres.port
  │  init: wait-for-neo4j    ──► neo4j.host:neo4j.port
  └──────────┬──────────┘
             │ reads configuration.yml + .env from ConfigMaps
             │ reads OPENAI_API_KEY, POSTGRES_PASSWORD,
             │   NEO4J_USER, NEO4J_PASSWORD from Secrets
             │ writes logs to memmachine-pvc (/app/data)
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
memmachine-postgres  memmachine-neo4j       ← only if enabled: true
(ClusterIP :5432)    (ClusterIP :7687 Bolt, :7474 HTTP, :7473 HTTPS)
     │                │
postgres-pvc         neo4j-pvc              ← only if enabled: true
(/var/lib/           (/data)
 postgresql/data)
```

**Startup order**: Two `initContainers` (`wait-for-postgres`, `wait-for-neo4j`) use `busybox` + `nc` to poll TCP connectivity before the main container starts. The host/port probed are taken from `postgres.host`/`postgres.port` and `neo4j.host`/`neo4j.port`, so they work for both in-cluster and external endpoints.

**Network model**: When deployed in-cluster, PostgreSQL and Neo4j are only reachable inside the cluster (ClusterIP). When `enabled: false`, MemMachine connects directly to the externally configured host. Only the MemMachine API is exposed externally via a NodePort.

---

## Deployment Structure

### Services and Deployments

| Component            | Kind       | Internal DNS name              | Ports                           | Conditional?           |
|----------------------|------------|--------------------------------|---------------------------------|------------------------|
| MemMachine           | Deployment | `memmachine-service`           | NodePort 31001 → :80 → pod:8080 | Always                 |
| PostgreSQL (pgvector)| Deployment | `memmachine-postgres`          | ClusterIP :5432                 | `postgres.enabled=true`|
| Neo4j                | Deployment | `memmachine-neo4j`             | ClusterIP :7687, :7474, :7473   | `neo4j.enabled=true`   |

### Persistent Storage

Up to three PVCs are created, all using the same `storageClass` and `pvcSize`:

| PVC name         | Mounted in     | Mount path                    | Purpose                         | Conditional?           |
|------------------|----------------|-------------------------------|---------------------------------|------------------------|
| `neo4j-pvc`      | Neo4j pod      | `/data`                       | Graph data, indexes, plugins    | `neo4j.enabled=true`   |
| `postgres-pvc`   | PostgreSQL pod | `/var/lib/postgresql/data`    | Relational/vector data          | `postgres.enabled=true`|
| `memmachine-pvc` | MemMachine pod | `/app/data`                   | Application logs and data files | Always                 |

All PVCs request `ReadWriteMany` (RWX) access mode. This requires a StorageClass that supports RWX (e.g., NFS-backed provisioners like `nfs-client`).

### Secrets

All three secrets are always created regardless of `postgres.enabled` / `neo4j.enabled`.

| Secret name          | Keys                                          | Consumed by                                                                   |
|----------------------|-----------------------------------------------|-------------------------------------------------------------------------------|
| `postgres-secret`    | `POSTGRES_PASSWORD`                           | PostgreSQL deployment (when in-cluster); MemMachine deployment env var        |
| `memmachine-secrets` | `OPENAI_API_KEY`                              | MemMachine deployment env var; `api_key` for LLM and embedder in configuration.yml |
| `neo4j-secret`       | `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_AUTH`  | Neo4j deployment `NEO4J_AUTH` (when in-cluster); MemMachine deployment env vars |

### ConfigMaps

| ConfigMap name          | Mounted as          | Purpose                                                                 |
|-------------------------|---------------------|-------------------------------------------------------------------------|
| `memmachine-config`     | `/app/configuration.yml` | Full application config: databases, LLM, embedder, reranker, memory |
| `memmachine-env-config` | `/app/.env`         | Env vars for the FastAPI/MCP server: DB URLs, gateway URL, log level   |

---

## Template Files

| File                          | Resources Created              | Description                                                     |
|-------------------------------|--------------------------------|-----------------------------------------------------------------|
| `templates/memmachine-deployment.yaml` | Deployment (memmachine)  | Main app pod with init containers, config/secret mounts         |
| `templates/memmachine-service.yaml`    | Service (NodePort)       | Exposes app externally on configurable NodePort                 |
| `templates/memmachine-configmaps.yaml` | ConfigMap × 2            | `memmachine-config` (configuration.yml) and `memmachine-env-config` (.env) |
| `templates/neo4j-deployment.yaml`      | Deployment (neo4j)       | Neo4j with APOC + GDS plugins, PVC for data                     |
| `templates/neo4j-service.yaml`         | Service (ClusterIP)      | Internal Neo4j access (Bolt 7687, HTTP 7474, HTTPS 7473)        |
| `templates/postgres-deployment.yaml`   | Deployment (memmachine-postgres) | PostgreSQL with pgvector, credentials from Secret           |
| `templates/postgres-service.yaml`      | Service (ClusterIP)      | Internal PostgreSQL access on port 5432                         |
| `templates/pvc.yaml`                   | PersistentVolumeClaim × 1–3 | `memmachine-pvc` always; `neo4j-pvc` if `neo4j.enabled`; `postgres-pvc` if `postgres.enabled` |
| `templates/secrets.yaml`               | Secret × 3               | `postgres-secret`, `memmachine-secrets`, `neo4j-secret` — all always created |

---

## Application Config Schema (`configuration.yml`)

The `memmachine-config` ConfigMap generates `/app/configuration.yml`. Its structure:

```yaml
logging:
  path: /app/data/memmachine.log
  level: info                      # hardcoded; use FAST_MCP_LOG_LEVEL env var to
                                   # control FastAPI/MCP server log level separately

episode_store:
  database: db_postgres            # references resources.databases.db_postgres

episodic_memory:
  long_term_memory:
    embedder: default_embedder     # references resources.embedders.default_embedder
    reranker: my_reranker_id       # references resources.rerankers.my_reranker_id
    vector_graph_store: db_neo4j   # references resources.databases.db_neo4j
  short_term_memory:
    llm_model: default_model       # references resources.language_models.default_model
    message_capacity: 500

semantic_memory:
  llm_model: default_model
  embedding_model: default_embedder
  database: db_postgres
  config_database: db_postgres

session_manager:
  database: db_postgres

prompt:
  default_project_categories:
    - profile_prompt

resources:
  databases:
    db_postgres: { provider: postgres, config: { host, port, user, password: $POSTGRES_PASSWORD, ... } }
    db_neo4j:    { provider: neo4j,    config: { uri, username: $NEO4J_USER, password: $NEO4J_PASSWORD, pool, ... } }
  embedders:
    default_embedder: { provider, config: { model, api_key: $OPENAI_API_KEY, base_url, dimensions } }
  language_models:
    default_model:    { provider, config: { model, api_key: $OPENAI_API_KEY, base_url } }
  rerankers:
    my_reranker_id:   { provider: rrf-hybrid, config: { reranker_ids: [...] } }
    id_ranker_id:     { provider: identity }
    bm_ranker_id:     { provider: bm25 }
```

Resource IDs used in top-level sections (`default_model`, `default_embedder`, `db_postgres`, `db_neo4j`, `my_reranker_id`) are resolved under `resources.*`.

---

## Values Reference

### Storage

| Value          | Default      | Description                                   |
|----------------|--------------|-----------------------------------------------|
| `storageClass` | `nfs-client` | StorageClass for all three PVCs               |
| `pvcSize`      | `5Gi`        | Storage request size for each PVC             |

### Neo4j (`neo4j.*`)

| Value                                 | Default               | Description                              |
|---------------------------------------|-----------------------|------------------------------------------|
| `neo4j.enabled`                       | `true`                | Deploy in-cluster Neo4j. Set to `false` to skip and use an external host |
| `neo4j.host`                          | `memmachine-neo4j`    | Bolt hostname; override with external host when `enabled: false` |
| `neo4j.port`                          | `7687`                | Bolt port; override if external uses a different port |
| `neo4j.image`                         | `neo4j:5.23-community`| Container image                          |
| `neo4j.auth`                          | `neo4j/memverge`      | `NEO4J_AUTH` env (format: `user/pass`)   |
| `neo4j.user`                          | `neo4j`               | Username for Bolt connections            |
| `neo4j.password`                      | `memverge`            | Password for Bolt connections            |
| `neo4j.plugins`                       | `[apoc, graph-data-science]` | Plugins auto-downloaded at startup|
| `neo4j.heap.initial`                  | `512m`                | JVM initial heap size                    |
| `neo4j.heap.max`                      | `1G`                  | JVM max heap size                        |
| `neo4j.pool.max_connection_pool_size` | `100`                 | Max Bolt connection pool size            |
| `neo4j.pool.connection_acquisition_timeout` | `60.0`          | Connection acquisition timeout (seconds) |
| `neo4j.pool.range_index_creation_threshold`  | `10000`        | Range index creation threshold           |
| `neo4j.pool.vector_index_creation_threshold` | `10000`        | Vector index creation threshold          |

### PostgreSQL (`postgres.*`)

| Value                    | Default                  | Description                        |
|--------------------------|--------------------------|------------------------------------|
| `postgres.enabled`       | `true`                   | Deploy in-cluster PostgreSQL. Set to `false` to skip and use an external host |
| `postgres.host`          | `memmachine-postgres`    | Hostname; override with external host when `enabled: false` |
| `postgres.port`          | `5432`                   | Port; override if external uses a different port |
| `postgres.image`         | `pgvector/pgvector:pg16` | Container image (includes pgvector)|
| `postgres.user`          | `memmachine`             | Database username                  |
| `postgres.password`      | `memverge`               | Database password                  |
| `postgres.database`      | `memmachine`             | Database name                      |
| `postgres.pool_size`     | `5`                      | SQLAlchemy pool size               |
| `postgres.max_overflow`  | `10`                     | SQLAlchemy max overflow            |

### MemMachine (`memmachine.*`)

| Value                              | Default                                          | Description                                   |
|------------------------------------|--------------------------------------------------|-----------------------------------------------|
| `memmachine.image`                 | `docker.io/memmachine/memmachine`                | Container image                               |
| `memmachine.tag`                   | `v0.2.6-cpu`                                     | Image tag                                     |
| `memmachine.pullPolicy`            | `IfNotPresent`                                   | Image pull policy                             |
| `memmachine.openaiApiKey`          | `<OPENAI_API_KEY>`                               | Stored in `memmachine-secrets`; injected as `OPENAI_API_KEY` env var and used as `api_key` for LLM and embedder |
| `memmachine.config.loggingLevel`   | `INFO`                                           | Controls `FAST_MCP_LOG_LEVEL` env var         |
| `memmachine.config.memoryConfigPath` | `/app/configuration.yml`                       | Path to configuration.yml inside the container |
| `memmachine.config.baseUrl`        | `http://127.0.0.1:8080`                          | `MCP_BASE_URL` env var                        |
| `memmachine.config.gatewayUrl`     | `http://localhost:8080`                          | `GATEWAY_URL` env var                         |
| `memmachine.model.provider`        | `openai-responses`                               | LLM provider type                             |
| `memmachine.model.base_url`        | `https://api.openai.com/v1`                      | LLM API base URL                              |
| `memmachine.model.model_path`      | `gpt-5-mini`                                     | LLM model name                                |
| `memmachine.embedder.provider`     | `openai`                                         | Embedder provider type                        |
| `memmachine.embedder.base_url`     | `https://api.openai.com/v1`                      | Embedder API base URL                         |
| `memmachine.embedder.model_path`   | `text-embedding-3-small`                         | Embedding model name                          |
| `memmachine.embedder.dimensions`   | `1536`                                           | Embedding vector dimensions                   |

### NodePorts (`nodePorts.*`)

| Value               | Default | Description                                                    |
|---------------------|---------|----------------------------------------------------------------|
| `nodePorts.http8080`| `31001` | NodePort for the MemMachine service (routes to pod port 8080)  |

---

## Known Issues / Notes

- **`{{ .Release.namespace }}` casing**: Several templates use `{{ .Release.namespace }}` (lowercase `n`) instead of the correct `{{ .Release.Namespace }}` (uppercase). In Helm, `.Release.Namespace` is the canonical field; the lowercase form resolves to an empty string. `pvc.yaml` correctly uses the uppercase form. If you encounter resources deployed to the wrong namespace, this is the cause. See the individual template files for details.

---

## Prerequisites

Before installing this chart, ensure the following:

1. **Kubernetes cluster** — A running cluster (1.19+ recommended) with `kubectl` configured.

2. **Helm 3** — Install from [helm.sh](https://helm.sh/docs/intro/install/). Verify with:
   ```bash
   helm version
   ```

3. **ReadWriteMany-capable StorageClass** — The default StorageClass is `nfs-client`. All three PVCs request `ReadWriteMany` access. Confirm your cluster has a suitable StorageClass:
   ```bash
   kubectl get storageclass
   ```
   To use a different class, override `storageClass` at install time.

4. **Access to the MemMachine container image** — The default image is hosted on docker hub (memmachine/memmachine).

5. **An OpenAI-compatible LLM backend** — By default, the chart points at `https://api.openai.com/v1` with the `openai-responses` provider. Set `memmachine.openaiApiKey` to your real key. Alternatives:
   - **Ollama**: set `memmachine.model.provider=openai-chat-completions`, `memmachine.model.base_url=http://ollama.ollama.svc.cluster.local:11434/v1`, `memmachine.model.api_key=EMPTY`.
   - **vLLM / other OpenAI-compatible**: set the appropriate `base_url` and `provider`.

---

## Usage Guide

### Basic install (openai backend, defaults)

```bash
helm upgrade --install memmachine . \
  --namespace memmachine --create-namespace
```

After deployment, check pod status (see [Verifying the deployment](#verifying-the-deployment) for the full checklist):

```bash
kubectl get pods -n memmachine
kubectl logs -n memmachine deployment/memmachine -f
```

Access the API from outside the cluster:
```bash
curl http://<node-ip>:31001/api/v2/health
```

---

### Verifying the deployment

Use these steps to confirm the release and resources are healthy. Default namespace is `memmachine`; release name is `memmachine` unless you used a different name.

**1. Helm release status**

```bash
helm status memmachine --namespace memmachine
```

**2. List all chart resources**

```bash
kubectl get deployments,services,configmaps,secrets,pvc --namespace memmachine
```

**3. Check pods and readiness**

```bash
kubectl get pods --namespace memmachine -o wide
```

All pods should show `Running` and `1/1` (or equivalent) ready. If any pod is not ready:

```bash
kubectl describe pod -l app=memmachine --namespace memmachine
kubectl get events --namespace memmachine --sort-by='.lastTimestamp'
```

**4. Check services and endpoints**

```bash
kubectl get services --namespace memmachine
kubectl get endpoints --namespace memmachine
```

**5. (Optional) View logs**

```bash
# MemMachine app
kubectl logs -l app=memmachine --namespace memmachine -f

# PostgreSQL (if in-cluster)
kubectl logs -l app=memmachine-postgres --namespace memmachine -f

# Neo4j (if in-cluster)
kubectl logs -l app=neo4j --namespace memmachine -f
```

**6. (Optional) Inspect values used by the release**

```bash
helm get values memmachine --namespace memmachine
```

**Quick overview (one command)**

```bash
helm status memmachine --namespace memmachine && \
  kubectl get deployments,pods,services,pvc --namespace memmachine
```

---

### Override the NodePort

```bash
helm upgrade --install memmachine . \
  --namespace memmachine --create-namespace \
  --set nodePorts.http8080=30005
```

---

### Use OpenAI as the LLM and embedder backend

OpenAI is the default backend. Only the API key and model names need to be set:

```bash
helm upgrade --install memmachine . \
  --namespace memmachine --create-namespace \
  --set memmachine.openaiApiKey=sk-... \
  --set memmachine.model.model_path=gpt-4o-mini \
  --set memmachine.embedder.model_path=text-embedding-3-small \
  --set memmachine.embedder.dimensions=1536
```

The `openaiApiKey` is stored in `memmachine-secrets` and injected as `$OPENAI_API_KEY`, which is referenced as `api_key` for both the LLM and embedder in `configuration.yml`.

---

### Use a values override file

Create `values-override.yaml` with your customizations, for example to use an Ollama backend instead of OpenAI:

```yaml
memmachine:
  openaiApiKey: dummy-key
  model:
    provider: openai-chat-completions
    base_url: http://ollama-service.ollama-30007.svc.cluster.local:11434/v1
    model_path: qwen3
  embedder:
    provider: openai
    base_url: http://ollama-service.ollama-30007.svc.cluster.local:11434/v1
    model_path: nomic-embed-text
    dimensions: 768
nodePorts:
  http8080: 30006
```

Then install:
```bash
helm upgrade --install memmachine-30006 . \
  --namespace memmachine-30006 --create-namespace \
  -f values-override.yaml
```

---

### External Databases

By default, the chart deploys PostgreSQL and Neo4j in-cluster. If you already operate your own database infrastructure, you can skip the in-cluster deployments and point MemMachine at external hosts.

#### External Postgres only

```bash
helm upgrade --install memmachine . \
  --namespace memmachine --create-namespace \
  --set postgres.enabled=false \
  --set postgres.host=my-pg.example.com \
  --set postgres.port=5432
```

The in-cluster PostgreSQL Deployment, Service, and PVC are not created. `postgres-secret` is still created (MemMachine always needs `POSTGRES_PASSWORD`). MemMachine connects to `my-pg.example.com:5432` instead.

#### External Neo4j only

```bash
helm upgrade --install memmachine . \
  --namespace memmachine --create-namespace \
  --set neo4j.enabled=false \
  --set neo4j.host=my-neo4j.example.com \
  --set neo4j.port=7687
```

The in-cluster Neo4j Deployment, Service, and PVC are not created. MemMachine connects to `bolt://my-neo4j.example.com:7687` instead.

#### Both external

```bash
helm upgrade --install memmachine . \
  --namespace memmachine --create-namespace \
  --set postgres.enabled=false --set postgres.host=pg.example.com \
  --set neo4j.enabled=false --set neo4j.host=neo4j.example.com
```

Only the MemMachine Deployment, Service, memmachine-pvc, two ConfigMaps, and three Secrets (`postgres-secret`, `memmachine-secrets`, `neo4j-secret`) are created.

---

### Change the StorageClass or PVC size

```bash
helm upgrade --install memmachine . \
  --namespace memmachine --create-namespace \
  --set storageClass=local-path \
  --set pvcSize=20Gi
```

> **Note:** `local-path` (Rancher) does not support ReadWriteMany. If your storage class only supports `ReadWriteOnce`, you will need to modify the PVC access modes in `templates/pvc.yaml` accordingly.

---

### Via deployment manager (multi-tenant setups)

The `deploy_cli.py` script in the parent directory handles NodePort allocation, namespace creation, and service registry automatically:

```bash
# Deploy with Ollama backend
python deploy_cli.py deploy-memmachine-ollama ollama-30000

# Deploy with OpenAI backend
python deploy_cli.py deploy-memmachine-openai --openaiApiKey sk-...

# Deploy with vLLM backend
python deploy_cli.py deploy-memmachine-vllm vllm-chat-31000 vllm-embedder-31003
```

---

### Uninstall

```bash
helm uninstall memmachine -n memmachine
```

> **Note:** Helm does not delete PVCs by default. To also remove persistent data:
> ```bash
> kubectl delete pvc -n memmachine neo4j-pvc postgres-pvc memmachine-pvc
> ```
