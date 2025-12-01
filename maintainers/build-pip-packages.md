# Building Pip Packages for MemMachine

**This document is specifically for building pip packages** (`memmachine-client` and `memmachine-server`) for distribution via PyPI or local installation.

This project supports creating two independent pip packages:
- `memmachine-client` - Contains only the REST client
- `memmachine-server` - Contains the complete server functionality

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

### Building Both Packages

You can run both commands sequentially:

```bash
uv build --project packages/client && uv build --project packages/server
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

## Usage

### Client Package Usage

After installing `memmachine-client`:

```python
from memmachine.rest_client import MemMachineClient, Memory

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

# Sync database schema
memmachine-sync-profile-schema

# Run MCP stdio mode
memmachine-mcp-stdio

# Run MCP HTTP mode
memmachine-mcp-http
```

## Package Contents

### memmachine-client
- Contains only the `memmachine.rest_client` module
- Dependencies: `requests`, `urllib3`
- Lightweight, suitable for scenarios that only need to call the MemMachine ServerAPI

### memmachine-server
- Contains all server-related modules:
  - `memmachine.common` - Common components
  - `memmachine.episodic_memory` - Episodic memory
  - `memmachine.server` - Server application
- Contains all server dependencies (database, FastAPI, etc.)
- Includes command-line tools

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
