# Building Pip Packages for MemMachine

**This document is specifically for building pip packages** (`memmachine-client` and `memmachine-server`) for distribution via PyPI or local installation.

This project supports creating two independent pip packages:
- `memmachine-client` - Contains only the REST client
- `memmachine-server` - Contains the complete server functionality

## Building

### Building Client Package

```bash
# Backup original configuration
cp pyproject.toml pyproject.toml.backup

# Use client configuration
cp pyproject-client.toml pyproject.toml

# Build
python -m build

# Restore original configuration
mv pyproject.toml.backup pyproject.toml
```

### Building Server Package

```bash
# Backup original configuration
cp pyproject.toml pyproject.toml.backup

# Use server configuration
cp pyproject-server.toml pyproject.toml

# Build
python -m build

# Restore original configuration
mv pyproject.toml.backup pyproject.toml
```

## Installation

### Installing from Locally Built Packages

#### Installing Client
```bash
pip install dist/memmachine_client-*.whl
# or
pip install dist/memmachine-client-*.tar.gz
```

#### Installing Server
```bash
pip install dist/memmachine_server-*.whl
# or
pip install dist/memmachine-server-*.tar.gz
```

### Installing from PyPI (if published)

#### Installing Client
```bash
pip install memmachine-client
```

#### Installing Server
```bash
pip install memmachine-server
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
- Lightweight, suitable for scenarios that only need to call the API

### memmachine-server
- Contains all server-related modules:
  - `memmachine.common` - Common components
  - `memmachine.episodic_memory` - Episodic memory
  - `memmachine.profile_memory` - User profile memory
  - `memmachine.server` - Server application
- Contains all server dependencies (database, FastAPI, etc.)
- Includes command-line tools

## Notes

1. **Namespace**: Both packages share the same namespace `memmachine`, but contain different subpackages
2. **Compatibility**: If both packages are installed, they can coexist because the modules they contain do not conflict
3. **Development Mode**: During development, it is recommended to use the main `pyproject.toml` for `pip install -e .`

## Publishing to PyPI

To publish to PyPI:

1. Update version numbers (in both configuration files)
2. Build both packages
3. Upload to PyPI separately:
   ```bash
   # Upload client
   twine upload dist/memmachine_client-*
   
   # Upload server
   twine upload dist/memmachine_server-*
   ```

