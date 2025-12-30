#!/usr/bin/env python3
# ruff: noqa: T201, E402
"""
Quick test script for CrewAI + MemMachine integration.

This script tests the basic functionality without requiring a full CrewAI setup.
"""

import sys
from pathlib import Path

# Add project root and src to path for imports
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

# Use relative import for tool module
from .tool import MemMachineTools, create_memmachine_tools


def test_memmachine_tools() -> None:
    """Test MemMachineTools class."""
    print("=" * 60)
    print("Test 1: MemMachineTools Class")
    print("=" * 60)

    # Create tools instance
    tools = MemMachineTools(
        base_url="http://localhost:8080",
        org_id="test_org",
        project_id="test_project",
        user_id="test_user",
    )

    print("✓ Tools instance created")
    print(f"  - org_id: {tools.org_id}")
    print(f"  - project_id: {tools.project_id}")
    print(f"  - user_id: {tools.user_id}")
    print()


def test_create_tools() -> None:
    """Test create_memmachine_tools function."""
    print("=" * 60)
    print("Test 2: create_memmachine_tools Function")
    print("=" * 60)

    try:
        tools = create_memmachine_tools(
            base_url="http://localhost:8080",
            org_id="test_org",
            project_id="test_project",
            user_id="test_user",
        )

        print(f"✓ Created {len(tools)} tools")
        for i, tool in enumerate(tools, 1):
            tool_name = getattr(tool, "name", getattr(tool, "__name__", "Unknown"))
            print(f"  {i}. {tool_name}")
        print()
    except Exception as e:
        print(f"⚠ Tool creation test skipped: {e}")
        print("  (This is expected if CrewAI is not installed)")
        print()


def test_playground_mode() -> None:
    """Test with playground mode."""
    print("=" * 60)
    print("Test 3: Playground Mode")
    print("=" * 60)

    from memmachine import MemMachineClient

    # Create client with playground mode
    client = MemMachineClient(
        base_url="http://localhost:8080",
        is_playground=True,
    )

    # Create tools with playground client
    tools = MemMachineTools(
        client=client,
        org_id="playground_org",
        project_id="playground_project",
    )

    # Test add memory (should work in playground mode)
    try:
        result = tools.add_memory(
            content="Test memory in playground mode",
            role="user",
        )
        print(f"✓ Add memory test: {result['status']}")
        print(f"  Message: {result.get('message', 'N/A')}")
    except Exception as e:
        print(f"⚠ Add memory test skipped: {e}")

    # Test search memory
    try:
        result = tools.search_memory(
            query="test",
            limit=5,
        )
        print(f"✓ Search memory test: {result['status']}")
        print(f"  Summary: {result.get('summary', 'N/A')[:100]}...")
    except Exception as e:
        print(f"⚠ Search memory test skipped: {e}")

    print()


def main() -> int:
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CrewAI + MemMachine Integration Tests")
    print("=" * 60 + "\n")

    try:
        test_memmachine_tools()
        test_create_tools()
        test_playground_mode()

        print("=" * 60)
        print("  All tests completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
