#!/usr/bin/env python3
"""
Test script to verify the refactored server works correctly.

Usage:
  1. Activate the virtual environment first:
     cd /path/to/cursor
     source .venv/bin/activate
  
  2. Run the test:
     python tests/test_server.py
"""

import sys
import os

# Add parent directory to path so we can import server module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for required dependencies
from fastapi.testclient import TestClient
from server.app import create_custom_app

def test_server():
    """Test that the server can start and respond to requests."""
    print("Testing refactored Cursor MCP Server...")
    
    # Create the app
    app = create_custom_app()
    print("✓ Server app created successfully")
    
    # Create FastAPI test client
    client = TestClient(app)
    response = client.get("/openapi.json")
    if response.status_code == 200:
        print("✓ OpenAPI schema accessible")
    else:
        print(f"✗ OpenAPI schema failed: {response.status_code}")
        return False
    
    # Test health endpoint
    response = client.get("/health")
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "healthy":
            print("✓ Health endpoint working")
        else:
            print(f"✗ Health endpoint returned unexpected data: {data}")
            return False
    else:
        print(f"✗ Health endpoint failed: {response.status_code}")
        return False
    
    # Test debug endpoint
    response = client.get("/debug")
    if response.status_code == 200:
        data = response.json()
        if "server" in data and "memory_backend_url" in data:
            print("✓ Debug endpoint working")
        else:
            print(f"✗ Debug endpoint returned unexpected data: {data}")
            return False
    else:
        print(f"✗ Debug endpoint failed: {response.status_code}")
        return False
    
    print("\n🎉 All tests passed! The refactored server is working correctly.")
    return True

if __name__ == "__main__":
    success = test_server()
    sys.exit(0 if success else 1)
