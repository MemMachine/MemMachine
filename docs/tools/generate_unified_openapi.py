
import json
import sys
import os
import inspect

# Add src to sys.path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from memmachine.server.api_v2.router import load_v2_api_router
from memmachine.server.api_v2.mcp import mcp

def generate_unified_openapi():
    app = FastAPI(title="MemMachine Server", version="0.2.0")
    
    # Load v2 routes
    load_v2_api_router(app)
    
    # Generate base OpenAPI
    openapi_schema = get_openapi(
        title="MemMachine Server",
        version="0.2.0",
        routes=app.routes,
    )
    
    # Tag v2 routes
    for path, methods in openapi_schema["paths"].items():
        if path.startswith("/api/v2"):
            for method in methods.values():
                method.setdefault("tags", []).append("v2")
        elif not path.startswith("/mcp"):
            # Assume everything else is v1
            for method in methods.values():
                method.setdefault("tags", []).append("v1")

    # Add MCP tools
    # We'll manually construct paths for MCP tools for documentation purposes
    # Assuming they might be called via some generic MCP endpoint or just documenting them
    # Since FastMCP structure isn't fully clear on HTTP mapping, we'll document them as abstract operations
    # or if we knew the specific HTTP mapping FastMCP uses.
    # For now, we will list them under /mcp/{tool_name} to satisfy the user's request to "include MCP tools"
    
    # Inspect mcp object for tools
    # FastMCP stores tools in _tool_manager._tools
    if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
        for name, tool in mcp._tool_manager._tools.items():
            path_item = {}
            
            # Create a POST operation for the tool
            operation = {
                "tags": ["MCP"],
                "summary": f"MCP Tool: {name}",
                "description": tool.description if hasattr(tool, "description") else "MCP Tool",
                "operationId": f"mcp_tool_{name}",
                "responses": {
                    "200": {
                        "description": "Tool execution result"
                    }
                }
            }
            
            # Try to extract parameters
            if hasattr(tool, "fn"):
                sig = inspect.signature(tool.fn)
                properties = {}
                required = []
                for param_name, param in sig.parameters.items():
                     schema = {"type": "string"}
                     if param.annotation != inspect.Parameter.empty:
                        if param.annotation == int:
                            schema = {"type": "integer"}
                        elif param.annotation == bool:
                            schema = {"type": "boolean"}
                     properties[param_name] = schema
                     if param.default == inspect.Parameter.empty:
                         required.append(param_name)

                operation["requestBody"] = {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": properties,
                                "required": required
                            }
                        }
                    }
                }

            path_item["post"] = operation
            openapi_schema["paths"][f"/mcp/{name}"] = path_item

    print(json.dumps(openapi_schema, indent=2))

if __name__ == "__main__":
    generate_unified_openapi()
