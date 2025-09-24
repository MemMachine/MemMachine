#!/bin/bash

# Setup script to pull the nomic-embed-text model for Ollama
echo "Setting up Ollama with nomic-embed-text model..."

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
timeout=60
counter=0
while [ $counter -lt $timeout ]; do
    if curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    echo "Waiting for Ollama... ($counter/$timeout)"
    sleep 2
    counter=$((counter + 2))
done

if [ $counter -ge $timeout ]; then
    echo "Error: Ollama did not become ready within $timeout seconds"
    exit 1
fi

# Pull the nomic-embed-text model
echo "Pulling nomic-embed-text model..."
docker exec memmachine-ollama ollama pull nomic-embed-text

if [ $? -eq 0 ]; then
    echo "Successfully pulled nomic-embed-text model!"
    echo "You can now use Ollama for embeddings in MemMachine."
else
    echo "Error: Failed to pull nomic-embed-text model"
    exit 1
fi
