# Building Pip Packages for MemMachine

**This document is specifically for building pip packages** for distribution via PyPI or local installation.

This project builds the following pip packages:
- `memmachine-common` - Shared API models used by the other packages ([PyPI](https://pypi.org/project/memmachine-common/))
- `memmachine-core` - The core memory engine: memory systems, abstractions, and provider/storage implementations ([PyPI](https://pypi.org/project/memmachine-core/))
- `memmachine-client` - Contains only the REST client ([PyPI](https://pypi.org/project/memmachine-client/))
- `memmachine-server` - The deployable server (HTTP/MCP, resource manager, configuration); depends on `memmachine-core` ([PyPI](https://pypi.org/project/memmachine-server/))
- `memmachine` - A meta-package that installs both the client and server ([PyPI](https://pypi.org/project/memmachine/))

## Directory Hierarchy

The project uses a monorepo-like structure under the `/packages` directory:

- `/packages/common`: Source code and configuration for `memmachine-common`.
- `/packages/core`: Source code and configuration for `memmachine-core`.
- `/packages/client`: Source code and configuration for `memmachine-client`.
- `/packages/server`: Source code and configuration for `memmachine-server`.
- `/packages/meta`: Configuration for the `memmachine` meta-package.

This structure allows us to maintain separate build configurations and dependencies for each component while keeping them in the same repository. It also provides the flexibility to version the client and server independently if needed in the future, although we currently aim for lock-step versioning.

## Building

The project uses `uv` workspaces to manage multi-package builds. This allows for building specific packages without swapping configuration files.

### Prerequisites

- `uv` installed (see [uv documentation](https://github.com/astral-sh/uv))

### Building Client Package

```bash
uv build --project packages/client
```

This will create the distribution files in `/dist` within the root of the project.

### Building Server Package

```bash
uv build --project packages/server
```

This will create the distribution files in `/dist` within the root of the project.

### Building Meta-Package

```bash
uv build --project packages/meta
```

### Building All Packages

You can run all commands sequentially:

```bash
uv build --project packages/common && uv build --project packages/client && uv build --project packages/core && uv build --project packages/server && uv build --project packages/meta
```

## Installation

### Installing from Locally Built Packages

#### Installing Client
```bash
pip install dist/memmachine_client-*.whl
# or
pip install dist/memmachine_client-*.tar.gz
```

#### Installing Server
```bash
pip install dist/memmachine_server-*.whl
# or
pip install dist/memmachine_server-*.tar.gz
```

### Installing from PyPI.org

#### Installing Client
```bash
pip install memmachine-client
```

#### Installing Server
```bash
pip install memmachine-server
```

#### Installing Server with GPU Support
To install the server with GPU support (includes `sentence-transformers`):
```bash
pip install "memmachine-server[gpu]"
```

#### Installing Full Suite (Meta-Package)
```bash
pip install memmachine
```

## Usage

### Client Package Usage

After installing `memmachine-client`:

```python
from memmachine_client import MemMachineClient

client = MemMachineClient(base_url="http://localhost:8080")
memory = client.memory(
    group_id="my_group",
    agent_id="my_agent",
    user_id="user123"
)
memory.add("I like pizza")
results = memory.search("What do I like?")
```

### Server Package Usage

After installing `memmachine-server`:

```bash
# Run server
memmachine-server

# Run MCP stdio mode
memmachine-mcp-stdio

# Run MCP HTTP mode
memmachine-mcp-http
```

## Package Contents

### memmachine-client
- Contains only the `memmachine_client` module
- Dependencies: `requests`, `urllib3`
- Lightweight, suitable for scenarios that only need to call the MemMachine ServerAPI

### memmachine-core
- Contains the `memmachine_core` module: episodic and semantic memory systems,
  abstractions, data models, and provider/storage implementations.
- Carries the heavy provider dependencies (OpenAI, Neo4j, SQLAlchemy, etc.).

### memmachine-server
- Contains the `memmachine_server` module: the HTTP/MCP server, resource manager,
  configuration, and installation tooling.
- Depends on `memmachine-core` (and `memmachine-common`).
- Includes command-line tools.

### memmachine (Meta-Package)
- Empty package with no source code.
- Depends on `memmachine-client` and `memmachine-server`.
- Ensures both components are installed.

## Notes

1. **Namespace**: Both packages share the same namespace `memmachine`, but contain different subpackages
2. **Compatibility**: If both packages are installed, they can coexist because the modules they contain do not conflict
3. **Development Mode**: During development, it is recommended to use the main `pyproject.toml` for `pip install -e .` or `uv sync`

## Publishing to PyPI

To publish to PyPI:

1. Update version numbers in:
   - `packages/client/pyproject.toml`
   - `packages/server/pyproject.toml`
2. Build both packages as described above.
3. Upload to PyPI:
   ```bash
   # Upload both client and server packages
   twine upload packages/client/dist/* packages/server/dist/*
   ```
