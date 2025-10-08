#!/bin/bash
# Setup script for Python virtual environment

echo "Setting up Python virtual environment..."

# Create virtual environment with Python 3.12
python3 -m venv hack-venv

# Activate virtual environment
source hack-venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Virtual environment setup complete!"
echo "To activate the environment, run: source hack-venv/bin/activate"
echo "To deactivate, run: deactivate"
