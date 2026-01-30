# MemMachine Helm Chart

This directory contains a Kubernetes Helm chart for deploying MemMachine, a memory service for AI agents.

## Overview

MemMachine is a comprehensive memory service designed to provide persistent, structured memory capabilities for AI agents. This Helm chart simplifies deployment on Kubernetes clusters by packaging MemMachine along with its required dependencies (PostgreSQL and Neo4j).

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- Sufficient persistent volume provisioners (for PostgreSQL and Neo4j storage)

## Chart Components

This Helm chart deploys the following components:

- **MemMachine Server**: The main API service
- **PostgreSQL with pgvector**: Vector database for embeddings and memory storage
- **Neo4j**: Graph database for relationship and semantic memory

## Installation

### Basic Installation

```bash
helm install memmachine ./deployments/helm
```

### Installation with Custom Release Name

```bash
helm install my-memmachine ./deployments/helm
```

### Installation in Specific Namespace

```bash
helm install memmachine ./deployments/helm -n memmachine --create-namespace
```

## Configuration

### Key Configuration Options

The chart is highly configurable through `values.yaml`. Below are the main configurable sections:

#### MemMachine Server Image

```yaml
image:
  registry: ""  # Docker registry (empty uses default)
  repository: "memmachine/memmachine"  # Container image
  tag: "v0.2.3-cpu"  # Tag (cpu, gpu variants available)
  pullPolicy: "IfNotPresent"  # Image pull policy
  pullSecrets: []  # Private registry credentials
```

#### Service Configuration

```yaml
service:
  type: ClusterIP  # Can be ClusterIP, NodePort, or LoadBalancer
  port: 8080  # Service port
```

#### PostgreSQL Configuration

```yaml
postgres:
  image:
    registry: "docker.io"
    repository: "pgvector/pgvector"
    tag: "pg16"
  port: 5432
  env:
    POSTGRES_DB: "memmachine"
    POSTGRES_USER: "memmachine"
    POSTGRES_PASSWORD: "memmachine_password"  # Change this!
  persistence:
    enabled: true
    size: "10Gi"  # Adjust based on your needs
```

#### Neo4j Configuration

```yaml
neo4j:
  image:
    registry: "docker.io"
    repository: "neo4j"
    tag: "5.23-community"
  env:
    NEO4J_AUTH: "neo4j/neo4j_password"  # Change this!
  persistence:
    enabled: true
    size: "10Gi"  # Adjust based on your needs
```

#### Model Configuration (Optional)

```yaml
model:
  embedding:
    baseUrl: ""  # Embedding model API endpoint
    apiKey: ""   # API key for embedding service
    model: ""    # Model name
    dimensions: 0  # Embedding dimensions
  chat:
    baseUrl: ""  # Chat model API endpoint
    apiKey: ""   # API key for chat service
    model: ""    # Model name
```

If no models are configured, endpoints will return 503 until models are configured at runtime.

#### Resource Limits

```yaml
resources: {}
  # limits:
  #   cpu: 1000m
  #   memory: 1Gi
  # requests:
  #   cpu: 500m
  #   memory: 512Mi
```

#### Persistence

```yaml
persistence:
  enabled: true
  size: "10Gi"  # Size of persistent volume
  storageClass: ""  # Use default or specify custom storage class
  mountPath: "/tmp/memory_logs"
```

### Advanced Configuration Options

#### Replica Count

```yaml
replicaCount: 1  # Number of MemMachine replicas
```

#### Pod Annotations and Labels

```yaml
podAnnotations: {}  # Add custom annotations
podLabels: {}  # Add custom labels
```

#### Node Selection and Affinity

```yaml
nodeSelector: {}  # Schedule on specific nodes
tolerations: []  # Tolerate node taints
affinity: {}  # Node affinity rules
```

#### Service Account

```yaml
serviceAccount:
  create: false  # Create service account if needed
  automountServiceAccountToken: false
  name: ""  # Custom service account name
```

#### Environment Variables

```yaml
env: {}  # Standard environment variables
extraEnv: []  # Additional environment variables
extraEnvFrom: []  # Environment variables from ConfigMaps/Secrets
```

## Custom Values Example

Create a `custom-values.yaml` file:

```yaml
replicaCount: 2

image:
  tag: "v0.2.3-gpu"

service:
  type: LoadBalancer
  port: 8000

postgres:
  env:
    POSTGRES_PASSWORD: "my-secure-password"
  persistence:
    size: "50Gi"
    storageClass: "fast-ssd"

neo4j:
  env:
    NEO4J_AUTH: "neo4j/my-secure-password"
  persistence:
    size: "50Gi"

model:
  embedding:
    baseUrl: "https://api.openai.com/v1"
    apiKey: "sk-..."
    model: "text-embedding-3-small"
    dimensions: 1536
  chat:
    baseUrl: "https://api.openai.com/v1"
    apiKey: "sk-..."
    model: "gpt-4"

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi
```

Then install with custom values:

```bash
helm install memmachine ./deployments/helm -f custom-values.yaml
```

## Post-Installation

### Verify Deployment

```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/name=memmachine

# Check service
kubectl get svc memmachine

# View logs
kubectl logs -l app.kubernetes.io/name=memmachine
```

### Port Forward for Local Testing

```bash
kubectl port-forward svc/memmachine 8080:8080
```

Then access MemMachine at `http://localhost:8080`.

### Initialize Databases

After deployment, databases should initialize automatically. You can verify:

```bash
# Check PostgreSQL logs
kubectl logs -l app.kubernetes.io/name=memmachine-postgres

# Check Neo4j logs
kubectl logs -l app.kubernetes.io/name=memmachine-neo4j
```

## Upgrade

To upgrade an existing release:

```bash
helm upgrade memmachine ./deployments/helm -f custom-values.yaml
```

## Uninstall

To remove the Helm release:

```bash
helm uninstall memmachine
```

**Note**: This does not automatically free persistent volumes. To clean up persistent volume claims:

```bash
kubectl delete pvc -l app.kubernetes.io/name=memmachine
```

## Troubleshooting

### Pods Not Starting

Check pod events:

```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### Database Connection Issues

Verify network connectivity:

```bash
# Test PostgreSQL connection
kubectl exec -it <memmachine-pod> -- psql -h memmachine-postgres -U memmachine -d memmachine

# Test Neo4j connection
kubectl exec -it <memmachine-pod> -- curl http://memmachine-neo4j:7474
```

### Persistent Volume Issues

Check PVC status:

```bash
kubectl get pvc
kubectl describe pvc <pvc-name>
```

Ensure your cluster has a default storage class or specify one in values.

### Memory/Resource Issues

Scale resources in `custom-values.yaml` or adjust pod limits.

## Chart Details

- **Chart Name**: memmachine
- **Chart Version**: 0.1.0
- **MemMachine Version**: v0.2.3-cpu
- **Home**: https://github.com/OpenCSGs/memmachine

## Related Documentation

- [MemMachine GitHub Repository](https://github.com/OpenCSGs/memmachine)
- [MemMachine Documentation](../../../docs/)
- [Docker Compose Alternative](../../memmachine-compose.sh)

## Support

For issues, feature requests, or questions, visit the [MemMachine GitHub Issues](https://github.com/OpenCSGs/memmachine/issues).
