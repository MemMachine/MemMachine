import json
import os
from typing import Annotated, TypedDict

from tool import MemMachineTools, create_add_memory_tool, create_search_memory_tool

# ============================================================================
# Configuration
# ============================================================================
# Configuration values can be set via environment variables or use defaults
MEMORY_BACKEND_URL = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
LANGGRAPH_ORG_ID = os.getenv("LANGGRAPH_ORG_ID", "langgraph_org")
LANGGRAPH_PROJECT_ID = os.getenv("LANGGRAPH_PROJECT_ID", "langgraph_project")
LANGGRAPH_GROUP_ID = os.getenv("LANGGRAPH_GROUP_ID", "langgraph_demo")
LANGGRAPH_AGENT_ID = os.getenv("LANGGRAPH_AGENT_ID", "demo_agent")
LANGGRAPH_USER_ID = os.getenv("LANGGRAPH_USER_ID", "demo_user")
LANGGRAPH_SESSION_ID = os.getenv("LANGGRAPH_SESSION_ID", "demo_session_001")


# State definition for LangGraph workflow
class AgentState(TypedDict):
    """State for the agent workflow."""

    messages: Annotated[list, "List of messages in the conversation"]
    user_id: str
    context: str
    memory_tool_results: Annotated[list, "Results from memory tool calls"]


def simple_memory_workflow_demo() -> None:
    """Simple demo showing basic memory operations without LangGraph dependency.

    This demonstrates the MemMachine tools functionality that can be integrated
    into LangGraph workflows.
    """
    print("=" * 60)
    print("MemMachine LangGraph Tools Demo")
    print("=" * 60)

    # Initialize tools
    print("\n1. Initializing MemMachine tools...")
    print("   Configuration:")
    print(f"     - Backend URL: {MEMORY_BACKEND_URL}")
    print(f"     - Org ID: {LANGGRAPH_ORG_ID}")
    print(f"     - Project ID: {LANGGRAPH_PROJECT_ID}")
    print(f"     - Group ID: {LANGGRAPH_GROUP_ID}")
    print(f"     - Agent ID: {LANGGRAPH_AGENT_ID}")
    print(f"     - User ID: {LANGGRAPH_USER_ID}")
    print(f"     - Session ID: {LANGGRAPH_SESSION_ID}")

    tools = MemMachineTools(
        base_url=MEMORY_BACKEND_URL,
        org_id=LANGGRAPH_ORG_ID,
        project_id=LANGGRAPH_PROJECT_ID,
        group_id=LANGGRAPH_GROUP_ID,
        agent_id=LANGGRAPH_AGENT_ID,
        user_id=LANGGRAPH_USER_ID,
        session_id=LANGGRAPH_SESSION_ID,
    )

    # Server connection will be checked automatically on first use
    print(f"✅ MemMachine tools initialized, server: {MEMORY_BACKEND_URL}")

    # Create tool functions
    add_memory = create_add_memory_tool(tools)
    search_memory = create_search_memory_tool(tools)

    print("\n2. Adding memories...")
    # Add some memories
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
        result = add_memory(
            content=mem["content"],
            metadata=mem["metadata"],
        )
        if result["status"] == "success":
            print(f"   ✅ Added: {mem['content'][:50]}...")
        else:
            print(f"   ❌ Failed: {result.get('message', 'Unknown error')}")

    print("\n3. Searching memories...")
    # Search for memories
    search_queries = [
        "What does the user prefer for development?",
        "What are the user's upcoming deadlines?",
        "What are the user's hobbies?",
        "What is the user interested in?",
    ]

    for query in search_queries:
        print(f"\n   🔍 Query: {query}")
        result = search_memory(query=query, limit=3)

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
            print(f"   ❌ Search failed: {result.get('message', 'Unknown error')}")

    print("\n4. Getting context...")
    context = tools.get_context()
    print(f"   Context: {json.dumps(context, indent=2)}")

    # Cleanup
    tools.close()


def main() -> None:
    """Main demo function."""
    try:
        # Run simple demo
        simple_memory_workflow_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
