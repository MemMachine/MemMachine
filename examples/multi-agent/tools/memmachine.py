# tools/memmachine.py
"""
MemMachine integration tools for persistent memory and conversation context.
"""
import httpx
from strands import tool
from typing import Optional, Dict, List
import json
from datetime import datetime
import os

# MemMachine server endpoint
MEMMACHINE_URL = os.getenv("MEMMACHINE_URL", "http://localhost:8080")

# Session configuration
DEFAULT_SESSION = {
    "group_id": "morning-brief",
    "agent_id": ["advisor_buddy"],
    "user_id": ["default_user"],  # Will be updated with actual user
    "session_id": "default_session"  # Will be updated per session
}


def _get_session_for_user(user_id: str, session_id: Optional[str] = None) -> dict:
    """Generate session config for a specific user
    
    CRITICAL: Use unique group_id per user for proper isolation!
    MemMachine groups memories by group_id first, so each user needs their own namespace.
    
    CRITICAL: Include ALL agent IDs that may store memories!
    The produced_for field must match one of the agent_ids in the session.
    
    IMPORTANT: Using original group_id to access existing memories.
    When storing new memories, MemMachine will update the group to include all agent_ids.
    """
    return {
        "group_id": f"morning-brief-{user_id}",  # Original format - keeps existing memories accessible
        "agent_id": ["advisor_buddy", "news_scout", "memory_keeper"],  # Include all agents!
        "user_id": [user_id],
        "session_id": session_id or f"session_{user_id}"
    }


@tool
def store_memory(
    content: str,
    producer: str,
    produced_for: str = "advisor_buddy",
    episode_type: str = "message",
    metadata: Optional[Dict] = None,
    user_id: str = "default_user",
    session_id: Optional[str] = None
) -> dict:
    """
    Store a memory in MemMachine for later retrieval.
    
    Args:
        content: The memory content to store (e.g., user preference, fact, conversation)
        producer: Who created this memory (user_id or agent_id)
        produced_for: Who this memory is for (usually agent_id)
        episode_type: Type of memory ("message", "preference", "fact", "event")
        metadata: Additional metadata to store with the memory
        user_id: The user ID for session context
        session_id: Optional session ID (defaults to user-based session)
    
    Returns:
        dict with status and any error messages
    
    Examples:
        # Store user preference
        store_memory(
            content="User prefers tech and finance news",
            producer="user_123",
            episode_type="preference"
        )
        
        # Store conversation context
        store_memory(
            content="User asked about Tesla stock price",
            producer="user_123",
            episode_type="message"
        )
    """
    try:
        # Use HEADERS instead of body session (deprecated)
        session = _get_session_for_user(user_id, session_id)
        
        headers = {
            "Content-Type": "application/json",
            "group-id": session["group_id"],
            "session-id": session["session_id"],
            "agent-id": ",".join(session["agent_id"]),  # Convert list to comma-separated string
            "user-id": ",".join(session["user_id"])      # Convert list to comma-separated string
        }
        
        # Build payload matching example_server.py format
        # Include session data in payload (as per example server)
        payload = {
            "session": {
                "group_id": session["group_id"],
                "agent_id": session["agent_id"],
                "user_id": session["user_id"],
                "session_id": session["session_id"]
            },
            "producer": producer,
            "produced_for": produced_for,
            "episode_content": content,
            "episode_type": episode_type,
            "metadata": metadata or {}
        }
        
        # Use /v1/memories like example_server.py (line 40)
        # This endpoint stores to episodic memory AND queues for profile extraction
        # Both memory types updated, matching the example server behavior
        response = httpx.post(
            f"{MEMMACHINE_URL}/v1/memories",
            json=payload,
            headers=headers,
            timeout=30.0  # Increased for cold starts
        )
        
        if response.status_code == 200:
            return {
                "status": "success",
                "message": "Memory stored successfully"
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to store memory: {response.status_code}",
                "details": response.text
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error storing memory: {str(e)}"
        }


@tool
def search_memories(
    query: str,
    user_id: str = "default_user",
    session_id: Optional[str] = None,
    limit: int = 5,
    filter_dict: Optional[Dict] = None
) -> dict:
    """
    Search for relevant memories in MemMachine.
    
    Args:
        query: Search query (natural language)
        user_id: The user ID for session context
        session_id: Optional session ID
        limit: Maximum number of results to return
        filter_dict: Optional filters for the search
    
    Returns:
        dict with search results including episodic and profile memories
    
    Examples:
        # Find user preferences
        search_memories("What does the user like?", user_id="user_123")
        
        # Find past conversations about a topic
        search_memories("previous conversations about stocks", user_id="user_123")
    """
    try:
        # Use HEADERS instead of body session (deprecated)
        session = _get_session_for_user(user_id, session_id)
        
        headers = {
            "Content-Type": "application/json",
            "group-id": session["group_id"],
            "session-id": session["session_id"],
            "agent-id": ",".join(session["agent_id"]),
            "user-id": ",".join(session["user_id"])
        }
        
        payload = {
            "query": query,
            "filter": filter_dict or {},
            "limit": limit
        }
        
        response = httpx.post(
            f"{MEMMACHINE_URL}/v1/memories/search",
            json=payload,
            headers=headers,
            timeout=30.0  # Increased for cold starts
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Parse the nested response structure
            content = result.get("content", {})
            episodic_raw = content.get("episodic_memory", [])
            profile_raw = content.get("profile_memory", [])
            
            # Episodic memory is a nested array - can be [[memories], [], [""]] or [[], [memories], [""]]
            # Find the non-empty array with actual dict objects
            episodic_memories = []
            if isinstance(episodic_raw, list):
                for item in episodic_raw:
                    if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
                        episodic_memories = item
                        break
            
            profile_memories = profile_raw if isinstance(profile_raw, list) else []
            
            return {
                "status": "success",
                "episodic_memories": episodic_memories,
                "profile_memories": profile_memories,
                "count": len(episodic_memories) + len(profile_memories)
            }
        else:
            return {
                "status": "error",
                "message": f"Search failed: {response.status_code}",
                "details": response.text
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error searching memories: {str(e)}"
        }


@tool
def get_user_context(
    user_id: str,
    session_id: Optional[str] = None,
    topics: Optional[List[str]] = None
) -> dict:
    """
    Retrieve comprehensive context about a user from their memory.
    
    This is useful at the start of a conversation to load:
    - User preferences (news categories they like)
    - User's name and personal info
    - Recent conversation topics
    - User habits and patterns
    
    Args:
        user_id: The user ID
        session_id: Optional session ID
        topics: Optional list of topics to search for (e.g., ["preferences", "name", "interests"])
    
    Returns:
        dict with consolidated user context
    
    Examples:
        # Get all user context at conversation start
        context = get_user_context(user_id="user_123")
        if context.get("name"):
            print(f"Welcome back, {context['name']}!")
    """
    try:
        # Search for various aspects of user context
        searches = [
            "user name and personal information",
            "user preferences and interests",
            "topics user likes to read about",
            "user's conversation history"
        ]
        
        if topics:
            searches = topics
        
        all_memories = []
        for search_query in searches:
            result = search_memories(
                query=search_query,
                user_id=user_id,
                session_id=session_id,
                limit=3
            )
            
            if result.get("status") == "success":
                all_memories.extend(result.get("episodic_memories", []))
                all_memories.extend(result.get("profile_memories", []))
        
        # Parse and structure the context
        context = {
            "user_id": user_id,
            "name": None,
            "preferences": [],
            "interests": [],
            "recent_topics": [],
            "all_memories": all_memories
        }
        
        # Extract structured info from memories
        for memory in all_memories:
            content = memory.get("content", "")
            metadata = memory.get("user_metadata", {})
            
            # Extract from structured metadata first
            if metadata.get("preference_type") == "name":
                context["name"] = metadata.get("preference_value")
            
            # Extract name from content if not in metadata
            if not context["name"]:
                content_lower = content.lower()
                if "name is" in content_lower or "i'm" in content_lower or "call me" in content_lower:
                    # Simple extraction
                    parts = content.split()
                    for i, word in enumerate(parts):
                        if word.lower() in ["is", "i'm", "im", "me"] and i + 1 < len(parts):
                            potential_name = parts[i + 1].strip(".,!?").capitalize()
                            if len(potential_name) > 1 and potential_name not in ["the", "a", "an", "The", "A", "An"]:
                                context["name"] = potential_name
                                break
            
            # Extract preferences from metadata
            pref_type = metadata.get("preference_type", "")
            pref_value = metadata.get("preference_value", "")
            if pref_type and pref_value:
                context["preferences"].append(f"{pref_type}: {pref_value}")
            
            # Extract interests from content
            content_lower = content.lower()
            for topic in ["tech", "finance", "sports", "politics", "movies", "nvidia", "stocks", "investing"]:
                if topic in content_lower:
                    if topic not in context["interests"]:
                        context["interests"].append(topic)
        
        return {
            "status": "success",
            "context": context
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting user context: {str(e)}"
        }


@tool
def store_user_preference(
    user_id: str,
    preference_type: str,
    preference_value: str,
    session_id: Optional[str] = None
) -> dict:
    """
    Store a specific user preference for easy retrieval.
    
    Args:
        user_id: The user ID
        preference_type: Type of preference (e.g., "name", "favorite_topics", "reading_style")
        preference_value: The actual preference value
        session_id: Optional session ID
    
    Returns:
        dict with status
    
    Examples:
        # Store user's name
        store_user_preference("user_123", "name", "Anirudh")
        
        # Store favorite topics
        store_user_preference("user_123", "favorite_topics", "tech,finance,sports")
    """
    content = f"User preference: {preference_type} = {preference_value}"
    metadata = {
        "preference_type": preference_type,
        "preference_value": preference_value,
        "stored_at": datetime.now().isoformat()
    }
    
    return store_memory(
        content=content,
        producer=user_id,
        produced_for="advisor_buddy",
        episode_type="preference",
        metadata=metadata,
        user_id=user_id,
        session_id=session_id
    )


@tool 
def check_memmachine_health() -> dict:
    """
    Check if MemMachine service is running and accessible.
    
    Returns:
        dict with health status
    """
    try:
        response = httpx.get(
            f"{MEMMACHINE_URL}/health",
            timeout=5.0
        )
        return {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "url": MEMMACHINE_URL,
            "response_code": response.status_code
        }
    except Exception as e:
        return {
            "status": "unavailable",
            "url": MEMMACHINE_URL,
            "error": str(e),
            "message": "Make sure MemMachine server is running on localhost:8080"
        }

