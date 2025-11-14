#!/bin/bash
set -e
if [ -f "pyproject.toml" ]; then
    cp pyproject.toml pyproject.toml.backup
    echo "Backed up original pyproject.toml"
fi

# Use client configuration
cp pyproject-client.toml pyproject.toml

# Check and install build module
if ! python -c "import build" 2>/dev/null; then
    echo "Installing build module..."
    pip install build
fi

# Build package
echo "Building with pyproject-client.toml..."
python -m build

# Restore original configuration
if [ -f "pyproject.toml.backup" ]; then
    mv pyproject.toml.backup pyproject.toml
    echo "Restored original pyproject.toml"
fi

echo ""
echo "memmachine-client package build completed!"
echo "Install command: pip install dist/memmachine_client-*.whl"
echo "Or: pip install dist/memmachine-client-*.tar.gz"

