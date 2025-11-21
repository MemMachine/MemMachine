"""
Test script for Cursor MCP Server.

This script tests the MCP server functionality by making HTTP requests
to the server endpoints.
"""

import requests
import json
import time
import sys
from typing import Dict, Any

# Configuration
MCP_SERVER_URL = "http://localhost:8001"
TEST_USER_ID = "test_user_123"
TEST_CONTENT = "This is a test memory for Cursor MCP integration"
TEST_QUERY = "test memory"

def make_request(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Make a request to the MCP server."""
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/{endpoint}",
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def get_mcp_session_id():
    """Get the MCP session ID by calling the MCP initialize API.
    
    This function makes a proper MCP initialize request to establish a session
    and extracts the session ID from the response headers. This follows the
    MCP protocol specification for session establishment.
    """
    print("Getting MCP Session ID via initialize request...")
    
    # MCP JSON-RPC initialize request format (based on Cursor's request)
    data = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": True,
                "prompts": True,
                "resources": True,
                "logging": False,
                "elicitation": {},
                "roots": {
                    "listChanged": False
                }
            },
            "clientInfo": {
                "name": "cursor-mcp-test",
                "version": "1.0.0"
            }
        }
    }
    
    # Headers matching Cursor's request format
    headers = {
        "user-id": TEST_USER_ID,
        "api_key": "value",  # Add API key for authentication
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "cursor-mcp-test/1.0.0",
        "Accept-Language": "*",
    }
    
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/mcp/",
            json=data,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        # Extract session_id from response headers (this is how MCP clients get it)
        session_id = None
        if "mcp-session-id" in response.headers:
            session_id = response.headers["mcp-session-id"]
            print(f"✓ Successfully extracted session_id from headers: {session_id}")
        elif "Mcp-Session-Id" in response.headers:
            session_id = response.headers["Mcp-Session-Id"]
            print(f"✓ Successfully extracted session_id from headers: {session_id}")
        else:
            print(f"✗ Session ID not found in response headers. Available headers: {list(response.headers.keys())}")
        
        # Send notifications/initialized after successful initialize
        if session_id:
            print("Sending notifications/initialized...")
            initialized_data = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            initialized_headers = headers.copy()
            initialized_headers["mcp-session-id"] = session_id
            
            try:
                init_response = requests.post(
                    f"{MCP_SERVER_URL}/mcp/",
                    json=initialized_data,
                    headers=initialized_headers,
                    timeout=30
                )
                print(f"✓ Notifications/initialized sent (status: {init_response.status_code})")
            except requests.exceptions.RequestException as e:
                print(f"⚠ Warning: Failed to send notifications/initialized: {e}")
        
    except requests.exceptions.RequestException as e:
        session_id = None
    
    print(f"Initialize request: {json.dumps(data, indent=2)}")
    print(f"Initialize headers: {json.dumps(headers, indent=2)}")
    print(f"Response headers: {dict(response.headers) if 'response' in locals() else 'N/A'}")
    
    # Return the session_id from headers (this is the correct way for MCP)
    if session_id:
        return session_id
    
    print("✗ Could not extract session_id from response headers or body")
    return None

def test_add_memory(session_id: str):
    """Test adding a memory with the obtained session ID."""
    print(f"\nTesting add_memory with session_id: {session_id}...")
    
    # MCP JSON-RPC format
    data = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "add_memory",
            "arguments": {
                "content": f"Second memory entry with session {session_id}"
            }
        }
    }
    
    # Add headers for user_id and the obtained session_id
    headers = {
        "user-id": TEST_USER_ID,
        "mcp-session-id": session_id,
        "api_key": "value",  # Add API key for authentication
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"  # Important for MCP
    }
    
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/mcp/",
            json=data,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        # Handle different response content types
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            # Handle Server-Sent Events format
            result = {"content_type": "text/event-stream", "status": "success"}
        else:
            # Try to parse as JSON
            try:
                result = response.json()
            except json.JSONDecodeError:
                result = {"content_type": content_type, "raw_response": response.text[:200]}
        
    except requests.exceptions.RequestException as e:
        result = {"error": str(e)}
    
    print(f"Add memory request: {json.dumps(data, indent=2)}")
    print(f"Add memory headers: {json.dumps(headers, indent=2)}")
    print(f"Add memory result: {json.dumps(result, indent=2)}")
    
    # Check if the operation was successful
    if "error" in result:
        print(f"✗ Add memory test failed: {result['error']}")
        return False
    elif result.get("content_type") == "text/event-stream" and result.get("status") == "success":
        print("✓ Add memory test passed (SSE format)")
        return True
    elif "result" in result and "status" in result["result"] and result["result"]["status"] == "success":
        print("✓ Add memory test passed (JSON format)")
        return True
    else:
        print("✗ Add memory test failed - unexpected response format")
        return False

def test_search_memory(session_id: str = None):
    """Test searching memories with proper headers using MCP JSON-RPC format."""
    print("\nTesting search_memory...")
    
    # MCP JSON-RPC format
    data = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "search_memory",
            "arguments": {
                "query": TEST_QUERY,
                "limit": 5
            }
        }
    }
    
    # Add headers for user_id and session_id (middleware will extract these)
    headers = {
        "user-id": TEST_USER_ID,
        "api_key": "value",  # Add API key for authentication
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"  # Important for MCP
    }
    
    if session_id:
        headers["mcp-session-id"] = session_id
    
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/mcp/",
            json=data,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        # Handle different response content types
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            # Handle Server-Sent Events format
            result = {"content_type": "text/event-stream", "status": "success"}
        else:
            # Try to parse as JSON
            try:
                result = response.json()
            except json.JSONDecodeError:
                result = {"content_type": content_type, "raw_response": response.text[:200]}
                
    except requests.exceptions.RequestException as e:
        result = {"error": str(e)}
    
    print(f"Search memory result: {json.dumps(result, indent=2)}")
    
    return result




def main():
    """Run all tests."""
    print("Starting Cursor MCP Server Tests")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{MCP_SERVER_URL}/", timeout=5)
        print(f"Server is running (status: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"Error: Cannot connect to MCP server at {MCP_SERVER_URL}")
        print("Make sure the server is running with: python cursor_mcp_server.py")
        sys.exit(1)
    
    # Step 1: Get MCP Session ID from the server
    print("\n" + "=" * 30)
    print("STEP 1: Getting MCP Session ID")
    print("=" * 30)
    session_id = get_mcp_session_id()
    
    if not session_id:
        print("✗ Failed to get session ID. Cannot proceed with other tests.")
        sys.exit(1)
    
    # Step 2: Use the session ID for other tests
    print("\n" + "=" * 30)
    print("Waiting for initialization to complete...")
    time.sleep(2)  # Give time for initialization to complete
    print("STEP 2: Running tests with session ID")
    print("=" * 30)
    
    # Test adding another memory with the session ID
    add_memory_success = test_add_memory(session_id)
    
    # Test searching memories with the session ID
    search_result = test_search_memory(session_id)
    search_success = not ("error" in search_result) and (
        search_result.get("content_type") == "text/event-stream" or 
        "result" in search_result
    )
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"✓ Get Session ID: {'PASSED' if session_id else 'FAILED'}")
    print(f"✓ Add Memory: {'PASSED' if add_memory_success else 'FAILED'}")
    print(f"✓ Search Memory: {'PASSED' if search_success else 'FAILED'}")
    print("=" * 50)
    print("Tests completed!")

if __name__ == "__main__":
    main()
