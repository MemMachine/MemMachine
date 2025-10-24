#!/bin/bash

# Cursor MCP Server and Web Application Docker Runner
set -e

echo "🚀 Starting Cursor MCP Server and Web Application with Docker Compose"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Using Docker named volumes for data persistence
echo "📦 Using Docker named volumes for data persistence"
echo ""

echo "🔧 Building and starting services..."
echo "   - MCP Server: http://localhost:8001 (internal)"
echo "   - Web App: http://localhost:3000"
echo ""

# Check if we should rebuild
if [[ "$1" == "--build" ]] || [[ "$1" == "-b" ]]; then
    echo "🔨 Building images..."
    docker compose up --build
else
    echo "⚡ Starting existing images (use --build to rebuild)..."
    docker compose up
fi
