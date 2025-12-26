# Klavis (Strata) HTTP Integration with MemMachine

This guide shows how to connect **Strata** (Klavis’ MCP router) to a **local MemMachine MCP server** over **HTTP (StreamableHTTP)**.

## Overview

You will run:

- **MemMachine MCP server (HTTP)**: `http://127.0.0.1:8080/mcp/`
- **Strata router (HTTP)**: `http://127.0.0.1:8090/mcp/`
- **Your MCP client** connects to Strata, and Strata proxies tool calls to MemMachine.

## Prerequisites

- MemMachine installed (or runnable via repo `PYTHONPATH=src`)
- Databases required by your MemMachine `configuration.yml` are running (commonly Postgres + Neo4j)
- Strata installed and on PATH (`strata`)
- If you have a system proxy (common on macOS), set `NO_PROXY` for localhost (see below)

## 0) Proxy / NO_PROXY (important on macOS)

If you ever see Strata using `httpx` and failing with `502 Bad Gateway` to `localhost` / `127.0.0.1`, you likely have a system proxy that intercepts localhost traffic.

Run Strata (and your test client) with:

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

## 1) Start MemMachine MCP HTTP server

From the MemMachine repo root:

```bash
export PYTHONPATH=src
export MEMORY_CONFIG="/absolute/path/to/MemMachine/configuration.yml"

python3 -m memmachine.server.mcp_http --host 127.0.0.1 --port 8080
```

Notes:

- The MCP endpoint is **`/mcp/`** (trailing slash matters). `http://127.0.0.1:8080/mcp` will often redirect (307) to `/mcp/`.
- If you see `FileNotFoundError: Config file cfg.yml not found`, it means `MEMORY_CONFIG` wasn’t set correctly.

## 2) Add MemMachine to Strata via HTTP

Use a dedicated config file (recommended) so you don’t modify your global Strata config:

```bash
export STRATA_CONFIG_PATH="/tmp/strata-memmachine.json"

strata --config-path "$STRATA_CONFIG_PATH" add --type http memmachine \
  "http://127.0.0.1:8080/mcp/" \
  --header "Accept:application/json, text/event-stream"
```

## 3) Start Strata router (HTTP)

```bash
strata --config-path "$STRATA_CONFIG_PATH" run --port 8090
```

Your Strata MCP endpoint will be:

- `http://127.0.0.1:8090/mcp/`

## 4) “Health / Ready” check (wait until init finished)

For MCP servers, the most reliable readiness probe is **the MCP handshake**:

- Send `initialize`
- Wait until you receive the first `event: message` JSON-RPC response

This proves the server is up **and** the MCP transport is working.

### A) Verify Strata responds to MCP initialize

```bash
curl -i -N "http://127.0.0.1:8090/mcp/" \
  -H "Accept: application/json, text/event-stream" \
  -H "Content-Type: application/json" \
  --data-binary '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
```

You should see an `event: message` with a JSON-RPC result.

### B) MemMachine note: `notifications/initialized` before `tools/list`

Some MCP servers (including MemMachine’s FastMCP stack) expect the client to send:

1. `initialize`
2. `notifications/initialized`
3. then call `tools/list`

Most real MCP clients do this automatically (including Python `ClientSession.initialize()`).

If you are testing manually with curl, you may need to send `notifications/initialized` yourself (as a JSON-RPC notification with **no** `id`).

## 5) Minimal Python MCP client (talk to Strata)

This is the “known good” pattern: **`ClientSession` must be used as an async context manager**, otherwise `initialize()` can hang/timeout.

```python
import asyncio
import httpx
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

MCP_URL = "http://127.0.0.1:8090/mcp/"  

def no_proxy_httpx_factory(headers=None, timeout=None, auth=None):
    return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth, trust_env=False)

async def main():
    print("MCP_URL =", MCP_URL)

    async with streamablehttp_client(
        MCP_URL,
        timeout=10,
        sse_read_timeout=30,
        httpx_client_factory=no_proxy_httpx_factory,
    ) as (read_stream, write_stream, get_session_id):

        async with ClientSession(read_stream, write_stream) as session:
            print("initializing...")
            caps = await session.initialize()
            print("session_id:", get_session_id())
            print("capabilities:", caps)

            print("listing tools...")
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])

if __name__ == "__main__":
    asyncio.run(main())
```

## Troubleshooting

### 307 Temporary Redirect

- Fix: always use the **trailing slash** form: **`/mcp/`**

### `502 Bad Gateway` (Strata -> MemMachine)

- Most common cause: **system proxy intercepting localhost**
- Fix: set `NO_PROXY=127.0.0.1,localhost` before running Strata

### MemMachine starts but tool calls fail

- Confirm your databases are running (Postgres + Neo4j if your config requires them)
- Confirm `MEMORY_CONFIG` points to the correct YAML


