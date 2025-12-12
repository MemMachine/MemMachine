"""
Example usage of MemMachine tools with AWS Strands Agent SDK.

This example demonstrates how to integrate MemMachine memory operations
into AWS Strands Agent SDK workflows.
"""

# ruff: noqa: T201
import json
import os

from tool import MemMachineTools, create_tool_handler, get_memmachine_tools

# ============================================================================
# Configuration
# ============================================================================
MEMORY_BACKEND_URL = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
MEMORY_API_KEY = os.getenv("MEMORY_API_KEY", None)
STRANDS_ORG_ID = os.getenv("STRANDS_ORG_ID", "strands_org")
STRANDS_PROJECT_ID = os.getenv("STRANDS_PROJECT_ID", "strands_project")
STRANDS_GROUP_ID = os.getenv("STRANDS_GROUP_ID", "strands_demo")
STRANDS_AGENT_ID = os.getenv("STRANDS_AGENT_ID", "demo_agent")
STRANDS_USER_ID = os.getenv("STRANDS_USER_ID", "demo_user")
STRANDS_SESSION_ID = os.getenv("STRANDS_SESSION_ID", None)


def basic_tools_demo() -> None:
    """Show MemMachine tools functionality."""
    print("=" * 60)
    print("MemMachine AWS Strands Agent SDK Tools Demo")
    print("=" * 60)

    # Initialize tools
    print("\n1. Initializing MemMachine tools...")
    print("   Configuration:")
    print(f"     - Backend URL: {MEMORY_BACKEND_URL}")
    print(f"     - Org ID: {STRANDS_ORG_ID}")
    print(f"     - Project ID: {STRANDS_PROJECT_ID}")
    print(f"     - Group ID: {STRANDS_GROUP_ID}")
    print(f"     - Agent ID: {STRANDS_AGENT_ID}")
    print(f"     - User ID: {STRANDS_USER_ID}")
    print(f"     - Session ID: {STRANDS_SESSION_ID or 'Auto-generated'}")

    tools = MemMachineTools(
        base_url=MEMORY_BACKEND_URL,
        api_key=MEMORY_API_KEY,
        org_id=STRANDS_ORG_ID,
        project_id=STRANDS_PROJECT_ID,
        group_id=STRANDS_GROUP_ID,
        agent_id=STRANDS_AGENT_ID,
        user_id=STRANDS_USER_ID,
        session_id=STRANDS_SESSION_ID,
    )

    # Check if server is available
    try:
        health = tools.client.health_check()
        print(f"MemMachine server is healthy: {health.get('status', 'ok')}")
    except Exception as e:
        print(f"   MemMachine server not available: {e}")
        print(f"   Please start MemMachine server on {MEMORY_BACKEND_URL}")
        return

    # Add memories
    print("\n2. Adding memories...")
    memories_to_add = [
        {
            "content": "User prefers working with Python for backend development",
            "metadata": {"category": "preference", "technology": "Python"},
        },
        {
            "content": "User mentioned they are working on a project deadline this Friday",
            "metadata": {"category": "task", "urgency": "high"},
        },
        {
            "content": "User enjoys hiking on weekends and lives in San Francisco",
            "metadata": {"category": "personal", "hobby": "hiking"},
        },
        {
            "content": "User is interested in machine learning and AI agents",
            "metadata": {"category": "interest", "field": "AI"},
        },
    ]

    for mem in memories_to_add:
        result = tools.add_memory(
            content=mem["content"],
            metadata=mem["metadata"],
        )
        if result["status"] == "success":
            print(f"   Added: {mem['content'][:50]}...")
        else:
            print(f"   Failed: {result.get('message', 'Unknown error')}")

    # Search memories
    print("\n3. Searching memories...")
    search_queries = [
        "What does the user prefer for development?",
        "What are the user's upcoming deadlines?",
        "What are the user's hobbies?",
        "What is the user interested in?",
    ]

    for query in search_queries:
        print(f"\n   Query: {query}")
        result = tools.search_memory(query=query, limit=3)

        if result["status"] == "success":
            results = result["results"]
            episodic = results.get("episodic_memory", [])

            if episodic:
                print(f"   Found {len(episodic)} relevant memories:")
                for i, mem in enumerate(episodic[:3], 1):
                    content = (
                        mem.get("content", "") if isinstance(mem, dict) else str(mem)
                    )
                    print(f"      {i}. {content[:80]}...")
            else:
                print("   No memories found")
        else:
            print(f"   Search failed: {result.get('message', 'Unknown error')}")

    # Get context
    print("\n4. Getting context...")
    context = tools.get_context()
    print(f"   Context: {json.dumps(context, indent=2)}")

    # Cleanup
    tools.close()


def strands_integration_example() -> None:
    """
    Show how to integrate with AWS Strands Agent SDK.

    Note: This is a conceptual example. Actual integration may vary
    based on the specific Strands Agent SDK version and API.
    """
    print("\n" + "=" * 60)
    print("Strands Agent SDK Integration Example")
    print("=" * 60)

    # Initialize tools and get schemas
    print("\n1. Initializing tools and getting schemas...")
    tools, tool_schemas = get_memmachine_tools(
        base_url=MEMORY_BACKEND_URL,
        api_key=MEMORY_API_KEY,
        org_id=STRANDS_ORG_ID,
        project_id=STRANDS_PROJECT_ID,
        group_id=STRANDS_GROUP_ID,
        agent_id=STRANDS_AGENT_ID,
        user_id=STRANDS_USER_ID,
        session_id=STRANDS_SESSION_ID,
    )

    print(f"   Created {len(tool_schemas)} tool schemas:")
    for schema in tool_schemas:
        # Tool schemas are now in flat format (toolSpec content directly)
        tool_name = schema.get("name", "unknown")
        print(f"      - {tool_name}")

    # Create tool handler
    print("\n2. Creating tool handler...")
    tool_handler = create_tool_handler(tools)
    print(f"   Tool handler created with {len(tool_handler)} tools")

    # Example: Simulate tool calls
    print("\n3. Simulating tool calls...")

    # Simulate add_memory call
    print("\n   Calling add_memory tool...")
    add_result = tool_handler["add_memory"](
        content="User mentioned they will be traveling to Japan next month",
        metadata={"type": "travel", "destination": "Japan"},
    )
    print(f"   Result: {add_result.get('status', 'unknown')}")
    if add_result.get("status") == "success":
        print(f"   Message: {add_result.get('message', '')}")

    # Simulate search_memory call
    print("\n   Calling search_memory tool...")
    search_result = tool_handler["search_memory"](
        query="Where is the user traveling?",
        limit=3,
    )
    print(f"   Result: {search_result.get('status', 'unknown')}")
    if search_result.get("status") == "success":
        summary = search_result.get("summary", "")
        print(f"   Summary: {summary[:200]}...")

    # Simulate get_context call
    print("\n   Calling get_context tool...")
    context_result = tool_handler["get_context"]()
    print(f"   Context: {json.dumps(context_result, indent=2)}")

    # Cleanup
    tools.close()

    print("\n" + "=" * 60)
    print("Integration Example Complete")
    print("=" * 60)
    print("\nTo use with actual Strands Agent SDK:")
    print("  1. Import Agent from strands")
    print("  2. Pass tool_schemas to Agent(tools=tool_schemas)")
    print("  3. The agent will automatically use the tools when needed")
    print("  4. Use tool_handler to execute tool calls from agent responses")


def main() -> None:
    """Run the main demo."""
    try:
        # Run basic demo
        basic_tools_demo()

        # Run integration example
        strands_integration_example()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
