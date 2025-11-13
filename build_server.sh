#!/bin/bash
# Build memmachine-server package

set -e

echo "Building memmachine-server package..."

# Backup original pyproject.toml
if [ -f "pyproject.toml" ]; then
    cp pyproject.toml pyproject.toml.backup
    echo "Backed up original pyproject.toml"
fi

# Use server configuration
cp pyproject-server.toml pyproject.toml

# Check and install build module
if ! python -c "import build" 2>/dev/null; then
    echo "Installing build module..."
    pip install build
fi

# Build package
echo "Building with pyproject-server.toml..."
python -m build

# Restore original configuration
if [ -f "pyproject.toml.backup" ]; then
    mv pyproject.toml.backup pyproject.toml
    echo "Restored original pyproject.toml"
fi

echo ""
echo "memmachine-server package build completed!"
echo "Install command: pip install dist/memmachine_server-*.whl"
echo "Or: pip install dist/memmachine-server-*.tar.gz"

