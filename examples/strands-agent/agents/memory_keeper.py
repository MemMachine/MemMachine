# agents/memory_keeper.py
"""
MemoryKeeper Agent - Specialized in managing user context and memories
Handles all interactions with MemMachine
"""

import hashlib

from strands import Agent, tool
from tools.memmachine import (
    get_user_context,
    search_memories,
    store_user_preference,
)


@tool
def store_user_info(user_id: str, info_type: str, info_value: str) -> dict:
    """
    Store information about the user (name, preferences, investments, interests).

    Args:
        user_id: The user's unique identifier
        info_type: Type of information (name, investment, interest, preference)
        info_value: The actual value to store

    Returns:
        Status of the storage operation
    """
    return store_user_preference(
        user_id=user_id,
        preference_type=info_type,
        preference_value=info_value,
    )


@tool
def recall_user_info(user_id: str, query: str | None = None) -> dict:
    """
    Retrieve stored information about the user.

    Args:
        user_id: The user's unique identifier
        query: Optional specific query (e.g., "investments", "sports preferences")

    Returns:
        User context including name, preferences, interests, and relevant memories
    """
    # Get full context
    context = get_user_context(user_id=user_id)

    if context.get("status") == "success":
        ctx = context.get("context", {})

        # If specific query, do semantic search
        if query:
            search_result = search_memories(query=query, user_id=user_id, limit=10)
            relevant_memories = []
            if search_result.get("status") == "success":
                for mem in search_result.get("episodic_memories", []):
                    content = mem.get("content", "")
                    if content:
                        relevant_memories.append(content)

            return {
                "status": "success",
                "name": ctx.get("name"),
                "interests": ctx.get("interests", []),
                "preferences": ctx.get("preferences", []),
                "relevant_memories": relevant_memories[:5],
                "has_context": True,
            }

        return {
            "status": "success",
            "name": ctx.get("name"),
            "interests": ctx.get("interests", []),
            "preferences": ctx.get("preferences", []),
            "recent_topics": ctx.get("recent_topics", []),
            "has_context": True,
        }

    return {
        "status": "error",
        "has_context": False,
        "message": "No stored context found for this user",
    }


@tool
def search_user_memories(user_id: str, query: str, limit: int = 5) -> dict:
    """
    Search through user's past conversations and memories.

    Args:
        user_id: The user's unique identifier
        query: What to search for
        limit: Maximum number of results

    Returns:
        Relevant memories matching the query
    """
    result = search_memories(query=query, user_id=user_id, limit=limit)

    if result.get("status") == "success":
        memories = [
            {
                "content": mem.get("content", ""),
                "timestamp": mem.get("timestamp", ""),
                "type": mem.get("episode_type", ""),
            }
            for mem in result.get("episodic_memories", [])
        ]

        return {"status": "success", "memories": memories, "count": len(memories)}

    return result


def make_memory_keeper(user_id: str = "default_user"):
    """
    Creates a MemoryKeeper agent specialized in managing user context.

    Args:
        user_id: The user this agent will manage memory for

    Returns:
        Configured Agent instance
    """
    system_prompt = f"""You are MemoryKeeper 🧠, a specialized AI agent focused on memory management.

USER: {user_id}

YOUR ROLE:
You manage all user information, preferences, and conversation history.
You're the expert on "what does this user care about?"

CAPABILITIES:
- Store user information (name, investments, interests, preferences)
- Recall user context and past conversations
- Search through user's memory for specific information
- Maintain user profiles and preferences

TOOLS:
- store_user_info(): Save new information about the user
- recall_user_info(): Get stored user context
- search_user_memories(): Search past conversations

BEHAVIOR:
- Always be thorough when storing information
- Extract key details from conversations
- Organize information by type (name, investment, interest, preference)
- Provide clear, structured responses about what you know

EXAMPLES:

Request: "Store that the user invested in NVIDIA for $1450"
Response: "Stored investment: NVIDIA - $1450"
[Call: store_user_info(user_id, "investment", "NVIDIA - $1450")]

Request: "What do we know about this user's investments?"
Response: [Call: recall_user_info(user_id, "investments")] then summarize findings

Request: "Search for what they told us about basketball"
Response: [Call: search_user_memories(user_id, "basketball", 5)] then present results

You are precise, organized, and never forget a detail! 🎯
"""

    agent = Agent(
        system_prompt=system_prompt,
        tools=[store_user_info, recall_user_info, search_user_memories],
    )

    agent.user_id = user_id
    agent.agent_name = "MemoryKeeper"

    return agent
