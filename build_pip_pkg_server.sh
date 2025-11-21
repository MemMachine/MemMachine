#!/usr/bin/env bash
# Build memmachine-server package

echo "Building memmachine-server package..."

# Backup original pyproject.toml
if [ -f "pyproject.toml" ]; then
    if ! cp pyproject.toml pyproject.toml.backup; then
        echo "ERROR: Failed to backup pyproject.toml"
        exit 1
    fi
    echo "Backed up original pyproject.toml"
fi

# Use server configuration
if ! cp pyproject-server.toml pyproject.toml; then
    echo "ERROR: Failed to copy pyproject-server.toml to pyproject.toml"
    if [ -f "pyproject.toml.backup" ]; then
        mv pyproject.toml.backup pyproject.toml
    fi
    exit 1
fi

# Check and install build module
if ! python -c "import build" 2>/dev/null; then
    echo "Installing build module..."
    if ! pip install build; then
        echo "ERROR: Failed to install build module. Please check your pip installation and network connection."
        if [ -f "pyproject.toml.backup" ]; then
            mv pyproject.toml.backup pyproject.toml
            echo "Restored original pyproject.toml"
        fi
        exit 1
    fi
    echo "Build module installed successfully"
fi

# Build package
echo "Building with pyproject-server.toml..."
if ! python -m build; then
    echo "ERROR: Package build failed. Please check the error messages above."
    if [ -f "pyproject.toml.backup" ]; then
        mv pyproject.toml.backup pyproject.toml
        echo "Restored original pyproject.toml"
    fi
    exit 1
fi

# Restore original configuration
if [ -f "pyproject.toml.backup" ]; then
    if ! mv pyproject.toml.backup pyproject.toml; then
        echo "WARNING: Failed to restore original pyproject.toml"
    else
        echo "Restored original pyproject.toml"
    fi
fi

echo ""
echo "memmachine-server package build completed!"
echo "Install command: pip install dist/memmachine_server-*.whl"
echo "Or: pip install dist/memmachine-server-*.tar.gz"

