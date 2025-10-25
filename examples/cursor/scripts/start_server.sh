#!/bin/bash

# Cursor MCP Server Startup Script

echo "Starting Cursor MCP Server for MemMachine..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠ Warning: Virtual environment not found. Creating one..."
    uv venv
    source .venv/bin/activate
    uv pip install -e .
fi

# Check if MemMachine backend is running
MEMORY_BACKEND_URL=${MEMORY_BACKEND_URL:-"http://localhost:8080"}
# echo "Checking MemMachine backend at $MEMORY_BACKEND_URL..."

# if curl -s --connect-timeout 5 "$MEMORY_BACKEND_URL/health" > /dev/null; then
#     echo "✓ MemMachine backend is running"
# else
#     echo "⚠ Warning: MemMachine backend is not responding at $MEMORY_BACKEND_URL"
#     echo "  Make sure MemMachine is running before using the MCP server"
# fi

# Set default port if not specified
export CURSOR_MCP_PORT=${CURSOR_MCP_PORT:-"8001"}

echo "Starting MCP server on port $CURSOR_MCP_PORT..."
echo "MemMachine backend URL: $MEMORY_BACKEND_URL"
echo ""
echo "To test the server, run: python test_mcp_server.py"
echo "To stop the server, press Ctrl+C"
echo ""

# Start the server
echo "Starting server from current directory..."
python -m server.main
