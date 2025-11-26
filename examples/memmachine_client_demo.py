"""
MemMachine Client Demo

This script demonstrates the MemMachine client library usage.

Configuration:
    The client can be configured via:
    1. Environment variable: MEMORY_BACKEND_URL (recommended)
       export MEMORY_BACKEND_URL="http://localhost:8080"
    2. Explicit parameter: base_url="http://localhost:8080"

Features demonstrated:
- Basic client initialization and health checks
- Memory operations (add, search)
- Multiple users and contexts
- Error handling and validation
- Advanced memory search with filters
- Memory context and metadata management
"""

import json
import os
import sys

from memmachine import MemMachineClient


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_memory_results(results, query=""):
    """Pretty print memory search results."""
    if query:
        print(f"\n🔍 Query: {query}")

    if not results or not results.get("episodic_memory"):
        print("   No memories found.")
        return

    episodic_memories = results["episodic_memory"]
    if episodic_memories and len(episodic_memories) > 0:
        print(f"   Found {len(episodic_memories[0])} relevant memories:")
        for i, memory in enumerate(episodic_memories[0][:3], 1):  # Show top 3
            print(f"      Time: {memory['timestamp']}")
            print(f"   {i}. {memory['content']}")
            if memory.get("user_metadata"):
                print(f"      Metadata: {memory['user_metadata']}")
            print()


def demo_advanced_memory_features():
    """Demonstrate advanced memory features."""
    print_section("Advanced Memory Features")

    # Use environment variable or explicit base_url
    base_url = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
    client = MemMachineClient(base_url=base_url)

    # Create memory instance with specific context
    memory = client.memory(
        group_id="advanced_demo",
        agent_id="smart_agent",
        user_id="demo_user",
        session_id="advanced_session",
    )

    print("Adding memories with rich metadata...")

    # Add memories with different episode types and metadata
    advanced_memories = [
        {
            "content": "I prefer Python over Java for backend development",
            "episode_type": "preference",
            "metadata": {
                "category": "programming",
                "language": "Python",
                "context": "work",
                "confidence": 0.9,
            },
        },
        {
            "content": "I attended the PyCon 2024 conference in Pittsburgh",
            "episode_type": "event",
            "metadata": {
                "category": "conference",
                "location": "Pittsburgh",
                "year": 2024,
                "type": "professional",
            },
        },
        {
            "content": "I'm learning machine learning with TensorFlow",
            "episode_type": "learning",
            "metadata": {
                "category": "education",
                "technology": "TensorFlow",
                "field": "machine_learning",
                "status": "in_progress",
            },
        },
    ]

    for mem_data in advanced_memories:
        try:
            memory.add(
                content=mem_data["content"],
                episode_type=mem_data["episode_type"],
                metadata=mem_data["metadata"],
            )
            print(f"✅ Added: {mem_data['content']}")
        except Exception as e:
            print(f"❌ Failed to add memory: {e}")

    # Demonstrate filtered search
    print("\nSearching with filters...")
    try:
        # Search for programming-related memories
        query = "What programming languages do I know?"
        results = memory.search(
            query=query,
            filter_dict={"category": "programming"},
            limit=5,
        )
        print_memory_results(results, query)

        # Search for conference memories
        query = "What conferences have I attended?"
        results = memory.search(
            query=query,
            filter_dict={"category": "conference"},
            limit=5,
        )
        print_memory_results(results, query)

    except Exception as e:
        print(f"❌ Advanced search failed: {e}")

    # Show memory context
    print(f"\nMemory context: {json.dumps(memory.get_context(), indent=2)}")


def demo_basic_client():
    """Demonstrate basic client usage."""
    print_section("Basic Client Usage")

    # Initialize client - use environment variable or explicit base_url
    print("Initializing MemMachine client...")
    base_url = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
    client = MemMachineClient(base_url=base_url, timeout=30)

    print(f"Client created: {client}")

    # Check server health
    print("\nChecking server health...")
    try:
        health = client.health_check()
        print(f"✅ Server is healthy: {health}")
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        print("Make sure MemMachine server is running on http://localhost:8080")
        return False

    return True


def demo_memory_operations():
    """Demonstrate memory operations."""
    print_section("Memory Operations")

    # Use environment variable or explicit base_url
    base_url = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
    client = MemMachineClient(base_url=base_url)

    # Create memory instance
    print("Creating memory instance...")
    memory = client.memory(
        group_id="demo_group",
        agent_id="demo_agent",
        user_id="demo_user",
        session_id="demo_session",
    )

    print(f"Memory instance: {memory}")

    # Add memories
    print("\nAdding memories...")
    memories_to_add = [
        ("I like pizza and pasta", {"type": "preference", "category": "food"}),
        ("I work as a software engineer", {"type": "fact", "category": "work"}),
        ("I live in San Francisco", {"type": "fact", "category": "location"}),
        ("I prefer working remotely", {"type": "preference", "category": "work"}),
        ("I enjoy hiking on weekends", {"type": "preference", "category": "hobby"}),
    ]

    for content, metadata in memories_to_add:
        try:
            memory.add(content, metadata=metadata)
            print(f"✅ Added: {content}")
        except Exception as e:
            print(f"❌ Failed to add '{content}': {e}")

    # Search memories
    print("\nSearching memories...")
    search_queries = [
        "What do I like to eat?",
        "Tell me about my work",
        "What are my hobbies?",
        "Where do I live?",
    ]

    for query in search_queries:
        try:
            results = memory.search(query, limit=3)
            print_memory_results(results, query)
        except Exception as e:
            print(f"❌ Search failed for '{query}': {e}")

    # Get context
    print(f"\nMemory context: {json.dumps(memory.get_context(), indent=2)}")


def demo_multiple_users():
    """Demonstrate multiple users scenario."""
    print_section("Multiple Users Scenario")

    # Use environment variable or explicit base_url
    base_url = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
    client = MemMachineClient(base_url=base_url)

    # Create memory instances for multiple users
    users = ["alice", "bob", "charlie"]
    memories = {}

    print("Creating memory instances for multiple users...")
    for user in users:
        # Each user should have their own session_id to avoid conflicts
        memories[user] = client.memory(
            group_id="team_group",
            agent_id="team_agent",
            user_id=user,
            session_id=f"session_{user}",  # Unique session per user
        )
        print(f"✅ Created memory instance for {user}")

    # Add user-specific memories
    print("\nAdding user-specific memories...")
    user_memories = {
        "alice": [
            ("I'm a frontend developer", {"role": "frontend", "skill": "React"}),
            ("I love design and UX", {"interest": "design"}),
        ],
        "bob": [
            ("I'm a backend developer", {"role": "backend", "skill": "Python"}),
            ("I specialize in APIs", {"interest": "api_design"}),
        ],
        "charlie": [
            ("I'm a DevOps engineer", {"role": "devops", "skill": "Docker"}),
            ("I love automation", {"interest": "automation"}),
        ],
    }

    for user, memory_list in user_memories.items():
        for content, metadata in memory_list:
            try:
                memories[user].add(content, metadata=metadata)
                print(f"✅ Added memory for {user}: {content}")
            except Exception as e:
                print(f"❌ Failed to add memory for {user}: {e}")

    # Search across users
    print("\nSearching across users...")
    for user, memory in memories.items():
        try:
            results = memory.search("What is your role?", limit=2)
            print(f"\n🔍 {user.capitalize()}'s role:")
            print_memory_results(results, f"What is {user}'s role?")
        except Exception as e:
            print(f"❌ Search failed for {user}: {e}")

    # Show memory contexts for all users
    print("\nMemory contexts for all users:")
    for user, memory in memories.items():
        print(f"\n{user.capitalize()}: {json.dumps(memory.get_context(), indent=2)}")


def main():
    """Main demo function."""
    print("MemMachine Client Demo")
    print("This demo shows how to use the MemMachine client library")

    # Show configuration info
    base_url = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
    print("\nConfiguration:")
    print(
        f"  MEMORY_BACKEND_URL: {os.getenv('MEMORY_BACKEND_URL', 'not set (using default)')}"
    )
    print(f"  Using base_url: {base_url}")
    print(f"\nMake sure MemMachine server is running on {base_url}")

    # Check if server is available
    if not demo_basic_client():
        print("\n❌ Server not available. Please start MemMachine server first.")
        print("You can start it with: python -m memmachine.server.app")
        sys.exit(1)

    # Run demos
    try:
        demo_memory_operations()
        demo_advanced_memory_features()
        demo_multiple_users()

        print_section("Demo Complete")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
