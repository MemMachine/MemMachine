# Semantic Memory `set_metadata` Scoping — REST API Example

This Jupyter notebook demonstrates how to use **semantic memory set scoping** via
the MemMachine REST API. It shows how the same category/tag structure can hold
distinct data for different users (or any other dimension) by using `set_metadata`
when creating and querying semantic memory sets.

The notebook is in [`rest_api_set_metadata.ipynb`](rest_api_set_metadata.ipynb).

## What this demonstrates

- Creating a MemMachine **Project** via the REST API
- Defining a **Semantic Set Type** with a `user_id` metadata tag for per-user scoping
- Resolving per-user **Set IDs** using `set_metadata`
- Creating **Categories** and **Tags** inside each set
- Writing **Semantic Features** (key/value memories) into user-scoped sets
- **Searching** semantic memory filtered by `set_metadata`
- **Listing** semantic memory filtered by `set_metadata`
- Verifying that searches and listings return only the memories belonging to the
  correct user
- Cleaning up all created resources when done

## Prerequisites

1. **Python 3.12+**
2. **Jupyter** — to run the notebook interactively (`pip install jupyter`)
3. A running **MemMachine server** with:
   - Semantic memory enabled
   - An OpenAI-compatible embedding model configured
   - A supported database backend (PostgreSQL/pgvector recommended)

## Start MemMachine

From the repository root, the quickest way to start a local instance is:

```bash
./memmachine-compose.sh
```

Then verify it is up:

```bash
curl -s http://localhost:8080/health
```

## Install dependencies

```bash
pip install jupyter requests
```

If you are running from a checkout of this repository you can also use `uv`:

```bash
uv sync
```

## Configuration

The notebook reads a single environment variable:

| Variable | Purpose | Default |
|---|---|---|
| `MEMORY_BACKEND_URL` | Base URL of the MemMachine server | `http://127.0.0.1:8080` |

Example:

```bash
export MEMORY_BACKEND_URL="http://localhost:8080"
```

No MemMachine API key is required unless your server has authentication enabled.

## Run the notebook

```bash
cd examples/rest_api_set_metadata
jupyter notebook rest_api_set_metadata.ipynb
```

Or execute it non-interactively and save the output:

```bash
jupyter nbconvert --execute --to notebook --inplace rest_api_set_metadata.ipynb
```

## How the example works

### 1) Create a project

A unique project is created for each notebook run using a random suffix so that
repeated runs never collide.

### 2) Define a Semantic Set Type

A set type is registered with `metadata_tags: ["user_id"]`. This tells
MemMachine to use `user_id` as the scoping dimension — one independent memory set
per distinct `user_id` value.

### 3) Resolve per-user Set IDs

`POST /api/v2/memories/semantic/set_id/get` is called twice — once with
`set_metadata: {"user_id": "user-a"}` and once with
`set_metadata: {"user_id": "user-b"}` — to obtain the two isolated Set IDs.

### 4) Populate each set

A `profile / facts` category is created in each set, and a `favorite_color`
feature is written:

- **user-a** → `favorite_color = blue`
- **user-b** → `favorite_color = green`

### 5) Verify scoping

- A **search** scoped to `user-a` confirms only `blue` is returned and that all
  results belong to `set_id_a`.
- A **list** scoped to `user-b` confirms only `green` is returned and that all
  results belong to `set_id_b`.

### 6) Cleanup

All semantic features, the set type, and the project are deleted so the server is
left in a clean state.

## Troubleshooting

- **Connection refused**: make sure the MemMachine server is running and
  `MEMORY_BACKEND_URL` is reachable.
- **Semantic memory not enabled**: check your server configuration — semantic
  memory must be enabled and a `config_database` must be set (see
  `.github/ci/memmachine-openai-pgvector.yml` for a minimal reference
  configuration).
- **Embedding errors**: verify the OpenAI API key is set and
  `text-embedding-3-small` (or your configured model) is accessible.
