# ruff: noqa: T201,SLF001
"""
Use MemMachine with LangChain.

This example demonstrates how to integrate MemMachine memory with LangChain to
provide persistent memory capabilities for conversational AI applications.
"""

import os

# Configuration
MEMORY_BACKEND_URL = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
LANGCHAIN_ORG_ID = os.getenv("LANGCHAIN_ORG_ID", "langchain_org")
LANGCHAIN_PROJECT_ID = os.getenv("LANGCHAIN_PROJECT_ID", "langchain_project")
LANGCHAIN_USER_ID = os.getenv("LANGCHAIN_USER_ID", "demo_user")
LANGCHAIN_AGENT_ID = os.getenv("LANGCHAIN_AGENT_ID", "demo_agent")
LANGCHAIN_SESSION_ID = os.getenv("LANGCHAIN_SESSION_ID", "demo_session_001")


def basic_memory_example() -> None:
    """Show MemMachine memory with LangChain."""
    print("=" * 60)
    print("LangChain + MemMachine Integration Example")
    print("=" * 60)
    print()

    try:
        from langchain.chains import ConversationChain
        from langchain.llms import OpenAI
        from langchain.prompts import PromptTemplate

        try:
            from .memory import MemMachineMemory
        except ImportError:
            from memory import MemMachineMemory

        # Initialize MemMachine memory
        print("1. Initializing MemMachine memory...")
        print(f"   Server URL: {MEMORY_BACKEND_URL}")
        print(f"   Org ID: {LANGCHAIN_ORG_ID}")
        print(f"   Project ID: {LANGCHAIN_PROJECT_ID}")
        print(f"   User ID: {LANGCHAIN_USER_ID}")

        memory = MemMachineMemory(
            base_url=MEMORY_BACKEND_URL,
            org_id=LANGCHAIN_ORG_ID,
            project_id=LANGCHAIN_PROJECT_ID,
            user_id=LANGCHAIN_USER_ID,
            agent_id=LANGCHAIN_AGENT_ID,
            session_id=LANGCHAIN_SESSION_ID,
            search_limit=5,
        )
        print("   ✓ Memory initialized")
        print()

        # Create LLM (you'll need to set OPENAI_API_KEY)
        print("2. Creating LLM...")
        try:
            llm = OpenAI(temperature=0)
            print("   ✓ LLM created")
        except Exception as e:
            print(f"   ⚠ LLM creation failed: {e}")
            print("   Note: Set OPENAI_API_KEY environment variable")
            return
        print()

        # Create conversation chain with MemMachine memory
        print("3. Creating conversation chain...")
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="""The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context.

Relevant context from memory:
{memmachine_context}

Conversation history:
{history}

Current conversation:
Human: {input}
AI:""",
        )

        chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=True,
        )
        print("   ✓ Chain created")
        print()

        # Run conversations
        print("4. Running conversations...")
        print()

        # First conversation
        print("   User: Hello, my name is Alice and I love Python programming.")
        response = chain.run("Hello, my name is Alice and I love Python programming.")
        print(f"   AI: {response}")
        print()

        # Second conversation - should remember the name
        print("   User: What's my name?")
        response = chain.run("What's my name?")
        print(f"   AI: {response}")
        print()

        # Third conversation - should remember the preference
        print("   User: What programming language do I like?")
        response = chain.run("What programming language do I like?")
        print(f"   AI: {response}")
        print()

        print("=" * 60)
        print("Example completed successfully!")
        print("=" * 60)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies:")
        print("  pip install langchain openai")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


def memory_operations_example() -> None:
    """Show direct memory operations."""
    print("=" * 60)
    print("MemMachine Memory Operations Example")
    print("=" * 60)
    print()

    try:
        from .memory import MemMachineMemory
    except ImportError:
        from memory import MemMachineMemory

    # Initialize memory
    memory = MemMachineMemory(
        base_url=MEMORY_BACKEND_URL,
        org_id=LANGCHAIN_ORG_ID,
        project_id=LANGCHAIN_PROJECT_ID,
        user_id=LANGCHAIN_USER_ID,
        agent_id=LANGCHAIN_AGENT_ID,
        session_id=LANGCHAIN_SESSION_ID,
    )

    # Add some memories directly
    print("1. Adding memories directly...")
    memories_to_add = [
        "I prefer working in the morning",
        "I'm interested in machine learning",
        "I live in San Francisco",
    ]

    for mem_content in memories_to_add:
        try:
            memory._memory.add(content=mem_content, role="user")
            print(f"   ✓ Added: {mem_content}")
        except Exception as e:
            print(f"   ❌ Failed to add: {e}")

    print()

    # Search memories
    print("2. Searching memories...")
    try:
        results = memory.load_memory_variables({"input": "What are my preferences?"})
        print("   Search results:")
        print(f"   History: {results.get('history', '')[:100]}...")
        print(f"   Context: {results.get('memmachine_context', '')[:100]}...")
    except Exception as e:
        print(f"   ❌ Search failed: {e}")

    print()
    print("=" * 60)


def main() -> None:
    """Run examples."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "operations":
        memory_operations_example()
    else:
        basic_memory_example()


if __name__ == "__main__":
    main()
