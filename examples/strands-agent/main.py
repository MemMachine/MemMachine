import sys

from agents.advisor_buddy import make_advisor_buddy


def run():
    # Get user ID from command line or use default
    user_id = sys.argv[1] if len(sys.argv) > 1 else "default_user"

    print("\n☀️ Morning Brief — TRUE Multi-Agent System")
    print(f"👤 User: {user_id}")
    print("🤖 3 Specialized Agents Working Together")
    print("━" * 50)

    # Create the multi-agent system
    buddy = make_advisor_buddy(user_id=user_id)

    print("\n💬 Multi-Agent System Ready!")
    print("   🎙️  AdvisorBuddy - Your main host & orchestrator")
    print("   🧠 MemoryKeeper - Remembers everything about you")
    print("   📰 NewsScout - Finds the latest news")
    print("\n💡 Chat naturally - the agents work together automatically!\n")

    while True:
        q = input("> ").strip()
        if q.lower() in ["quit", "exit", "bye"]:
            print("👋 Goodbye! See you next time!")
            break

        if not q:
            continue

        try:
            # Call the agent - it will understand and respond
            print()  # Blank line before response
            buddy(q)
            print()  # Blank line after response
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            print()


if __name__ == "__main__":
    run()
