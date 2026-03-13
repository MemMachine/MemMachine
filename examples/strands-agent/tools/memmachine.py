# tools/memmachine.py
"""
MemMachine integration tools for persistent memory and conversation context.
Updated to use MemMachine v2 API with MemMachineClient, Project, and Memory classes.
"""
from strands import tool
from typing import Optional, Dict, List
from datetime import datetime
import os

# Try to import MemMachine client classes
try:
    from memmachine import MemMachineClient
    MEMMACHINE_AVAILABLE = True
except ImportError:
    # Fallback: Try adding the path manually (for editable installs where .pth isn't processed)
    import sys
    import os
    # Go up from tools/memmachine.py -> strands-agent -> examples -> MemMachine -> src
    _memmachine_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src'))
    if os.path.exists(_memmachine_src_path) and _memmachine_src_path not in sys.path:
        sys.path.insert(0, _memmachine_src_path)
    try:
        from memmachine import MemMachineClient
        MEMMACHINE_AVAILABLE = True
    except ImportError:
        # Final fallback if memmachine is not installed
        MemMachineClient = None
        MEMMACHINE_AVAILABLE = False
        print("Warning: memmachine package not found. Install with: pip install -e ../../")

# MemMachine server endpoint
MEMMACHINE_URL = os.getenv("MEMMACHINE_URL", "http://localhost:8080")

# Project configuration (can be overridden via environment variables)
DEFAULT_ORG_ID = os.getenv("MEMMACHINE_ORG_ID", "strands-agent")
DEFAULT_PROJECT_ID = os.getenv("MEMMACHINE_PROJECT_ID", "morning-brief")

# Global client and project instances (lazy initialization)
_client = None
_project = None


def _get_client() -> Optional[MemMachineClient]:
    """Get or create MemMachine client instance."""
    global _client
    if not MEMMACHINE_AVAILABLE:
        return None
    if _client is None and MemMachineClient is not None:
        try:
            _client = MemMachineClient(
                base_url=MEMMACHINE_URL,
                timeout=30
            )
        except Exception as e:
            print(f"Warning: Failed to create MemMachine client: {e}")
            return None
    return _client


def _get_project() -> Optional:
    """Get or create MemMachine project instance."""
    global _project
    if _project is None:
        client = _get_client()
        if client is None:
            return None
        try:
            # Try to get existing project first
            try:
                _project = client.get_project(
                    org_id=DEFAULT_ORG_ID,
                    project_id=DEFAULT_PROJECT_ID
                )
            except Exception as get_error:
                # If project doesn't exist (404), try to create it
                # If it's a 409 (already exists), try getting it again
                error_str = str(get_error)
                if "404" in error_str or "Not Found" in error_str or "does not exist" in error_str:
                    try:
                        # Pass empty strings for embedder/reranker to use server defaults
                        # "default" is not a valid value - it must be empty string
                        _project = client.create_project(
                            org_id=DEFAULT_ORG_ID,
                            project_id=DEFAULT_PROJECT_ID,
                            description="Morning Brief multi-agent system",
                            embedder="",  # Empty string = use server default
                            reranker=""   # Empty string = use server default
                        )
                    except Exception as create_error:
                        create_str = str(create_error)
                        # If project already exists (409), try getting it again
                        if "409" in create_str or "already exists" in create_str:
                            try:
                                _project = client.get_project(
                                    org_id=DEFAULT_ORG_ID,
                                    project_id=DEFAULT_PROJECT_ID
                                )
                            except Exception:
                                print(f"Warning: Failed to get project after creation conflict: {create_error}")
                                return None
                        else:
                            print(f"Warning: Failed to create project: {create_error}")
                            return None
                else:
                    print(f"Warning: Failed to get project: {get_error}")
                    return None
        except Exception as e:
            print(f"Warning: Failed to get/create project: {e}")
            return None
    return _project


def _get_memory_for_user(user_id: str, agent_id: str = "advisor_buddy"):
    """Get Memory instance for a specific user.
    
    Args:
        user_id: The user ID
        agent_id: The agent ID that will store/access this memory (default: "advisor_buddy")
                  Use a single string, not a list, to avoid API validation errors
    """
    project = _get_project()
    if project is None:
        return None
    
    # Use single string values, not lists, to match API v2 requirements (dict[str, str] for metadata)
    return project.memory(
        agent_id=agent_id,  # Single string, not a list
        user_id=user_id,    # Single string, not a list
    )


@tool
def store_memory(
    content: str,
    producer: str,
    produced_for: str = "advisor_buddy",
    episode_type: str = "message",
    metadata: Optional[Dict] = None,
    user_id: str = "default_user"
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
        # Use the produced_for agent_id for the memory instance
        memory = _get_memory_for_user(user_id, agent_id=produced_for)
        if memory is None:
            return {
                "status": "error",
                "message": "MemMachine client not available. Make sure MemMachine server is running."
            }
        
        # Determine role based on producer
        role = "user" if producer == user_id or producer.startswith("user_") else "assistant"
        
        # Ensure metadata values are strings (API v2 requirement: dict[str, str])
        clean_metadata = {}
        if metadata:
            for k, v in metadata.items():
                # Convert all values to strings
                if isinstance(v, (list, tuple)):
                    clean_metadata[k] = ",".join(str(item) for item in v)
                else:
                    clean_metadata[k] = str(v)
        
        # CRITICAL: Override agent_id in metadata to ensure it's always a string
        # The library's _build_metadata may add agent_id as a list, but API requires strings
        clean_metadata["agent_id"] = str(produced_for) if produced_for else "advisor_buddy"
        # Also ensure user_id is a string
        clean_metadata["user_id"] = str(user_id)
        
        # Monkey-patch the library's _build_metadata method to convert ALL metadata values to strings
        # This ensures compliance with API v2 requirement: dict[str, str]
        # CRITICAL: Capture variables in closure to ensure they're available
        captured_produced_for = produced_for
        captured_user_id = user_id
        
        original_build_metadata = memory._build_metadata
        def patched_build_metadata(additional_metadata=None):
            metadata = original_build_metadata(additional_metadata)
            # Convert ALL values to strings (API v2 requirement: dict[str, str])
            # This handles agent_id, user_id, and any other metadata that might be lists
            for k, v in list(metadata.items()):
                if isinstance(v, (list, tuple)):
                    # Join list items with comma
                    metadata[k] = ",".join(str(item) for item in v)
                elif v is None:
                    # Convert None to empty string
                    metadata[k] = ""
                elif not isinstance(v, str):
                    # Convert any other type to string
                    metadata[k] = str(v)
            
            # CRITICAL: Override agent_id with the actual produced_for value (single string)
            # The library may have set it to a list, but we want the specific agent that's storing this memory
            if captured_produced_for:
                metadata["agent_id"] = str(captured_produced_for)
            elif "agent_id" not in metadata or not metadata["agent_id"]:
                metadata["agent_id"] = "advisor_buddy"
            
            # Also ensure user_id is a string
            metadata["user_id"] = str(captured_user_id)
            
            return metadata
        
        memory._build_metadata = patched_build_metadata
        
        # Add the memory using v2 API
        # Ensure episode_type is a string
        success = memory.add(
            content=content,
            role=role,
            producer=producer,
            produced_for=produced_for,
            episode_type=str(episode_type) if episode_type else "",
            metadata=clean_metadata
        )
        
        if success:
            return {
                "status": "success",
                "message": "Memory stored successfully"
            }
        else:
            return {
                "status": "error",
                "message": "Failed to store memory"
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
    limit: int = 5,
    filter_dict: Optional[Dict] = None
) -> dict:
    """
    Search for relevant memories in MemMachine.
    
    Args:
        query: Search query (natural language)
        user_id: The user ID for session context
        limit: Maximum number of results to return
        filter_dict: Optional filters for the search
    
    Returns:
        dict with search results including episodic and semantic memories
    
    Examples:
        # Find user preferences
        search_memories("What does the user like?", user_id="user_123")
        
        # Find past conversations about a topic
        search_memories("previous conversations about stocks", user_id="user_123")
    """
    try:
        memory = _get_memory_for_user(user_id)
        if memory is None:
            return {
                "status": "error",
                "message": "MemMachine client not available. Make sure MemMachine server is running."
            }
        
        # Search using v2 API
        result = memory.search(
            query=query,
            limit=limit,
            filter_dict=filter_dict
        )
        
        # Parse the v2 API response format
        # v2 API returns: {"episodic_memory": [...], "semantic_memory": [...]}
        episodic_memories = result.get("episodic_memory", [])
        semantic_memories = result.get("semantic_memory", [])
        
        # Convert to format expected by existing code
        # The episodic_memory is a list of episode objects
        # Each episode has: content, producer, produced_for, timestamp, role, metadata
        formatted_episodic = []
        for ep in episodic_memories:
            if isinstance(ep, dict):
                formatted_episodic.append({
                    "content": ep.get("content", ""),
                    "producer": ep.get("producer", ""),
                    "produced_for": ep.get("produced_for", ""),
                    "timestamp": ep.get("timestamp", ""),
                    "role": ep.get("role", "user"),
                    "episode_type": ep.get("metadata", {}).get("episode_type", "message"),
                    "user_metadata": ep.get("metadata", {})
                })
        
        # Semantic memories (profile memories) need normalization
        # They have 'value' instead of 'content', so normalize them
        formatted_semantic = []
        for sem in (semantic_memories if isinstance(semantic_memories, list) else []):
            if isinstance(sem, dict):
                # Normalize: add 'content' field from 'value' if it exists
                normalized = dict(sem)
                if "value" in normalized and "content" not in normalized:
                    normalized["content"] = normalized["value"]
                formatted_semantic.append(normalized)
        
        return {
            "status": "success",
            "episodic_memories": formatted_episodic,
            "semantic_memories": formatted_semantic,
            "count": len(formatted_episodic) + len(formatted_semantic)
        }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error searching memories: {str(e)}"
        }


@tool
def get_user_context(
    user_id: str,
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
            f"user name {user_id}",  # More specific search for name
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
                limit=3
            )
            
            if result.get("status") == "success":
                all_memories.extend(result.get("episodic_memories", []))
                all_memories.extend(result.get("semantic_memories", []))
        
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
        # First, check if user_id itself might be a name (common case)
        if user_id and user_id.lower() not in ["default_user", "user", "test_user"]:
            # Check if user_id looks like a name (capitalized, not too long, not all numbers)
            if (user_id[0].isupper() and 
                len(user_id) > 2 and 
                not user_id.replace("_", "").isdigit() and
                not user_id.startswith("user_")):
                context["name"] = user_id.strip(".,!?")
        
        for memory in all_memories:
            # Handle both 'content' (episodic) and 'value' (semantic) fields
            content = memory.get("content", "") or memory.get("value", "")
            metadata = memory.get("user_metadata", {}) or memory.get("metadata", {})
            
            if not content:
                continue
            
            content_lower = content.lower()
            
            # Extract from structured metadata first (if available)
            if metadata.get("preference_type") == "name":
                context["name"] = metadata.get("preference_value")
            
            # Extract name from content if not in metadata
            # Look for patterns like "name is X", "I'm X", "call me X", "user X", "bola", etc.
            if not context["name"]:
                # Check for explicit name patterns
                if "name is" in content_lower or "i'm" in content_lower or "call me" in content_lower:
                    parts = content.split()
                    for i, word in enumerate(parts):
                        if word.lower() in ["is", "i'm", "im", "me"] and i + 1 < len(parts):
                            potential_name = parts[i + 1].strip(".,!?").capitalize()
                            if len(potential_name) > 1 and potential_name not in ["the", "a", "an", "The", "A", "An"]:
                                context["name"] = potential_name
                                break
                
                # Check if content is just a name (e.g., "Bola", "John", "Sarah")
                if not context["name"]:
                    content_stripped = content.strip(".,!?")
                    words = content_stripped.split()
                    # If it's a single word or short phrase that looks like a name
                    if len(words) <= 2:
                        # Check if it starts with capital letter and is not a common word
                        first_word = words[0]
                        common_words = ["user", "the", "this", "that", "i", "my", "we", "they", "name", "is", "are"]
                        if (first_word[0].isupper() and 
                            len(first_word) > 2 and 
                            first_word.lower() not in common_words and
                            not first_word.lower().startswith("user")):
                            context["name"] = first_word.strip(".,!?")
                
                # Also check if content starts with a name (e.g., "Bola", "User bola")
                if not context["name"]:
                    # Look for capitalized words at the start that might be names
                    words = content.split()
                    if words and words[0][0].isupper() and len(words[0]) > 2:
                        # Check if it's not a common word
                        common_words = ["User", "The", "This", "That", "I", "My", "We", "They"]
                        if words[0] not in common_words:
                            context["name"] = words[0].strip(".,!?")
            
            # Extract preferences from metadata (if available)
            pref_type = metadata.get("preference_type", "")
            pref_value = metadata.get("preference_value", "")
            if pref_type and pref_value:
                context["preferences"].append(f"{pref_type}: {pref_value}")
            
            # Extract interests from content text
            # Look for patterns like "interested in X", "likes X", "prefers X", etc.
            interest_keywords = [
                "interested in", "likes", "loves", "enjoys", "prefers", 
                "favorite", "into", "passionate about", "fan of"
            ]
            
            for keyword in interest_keywords:
                if keyword in content_lower:
                    # Extract the interest after the keyword
                    idx = content_lower.find(keyword)
                    if idx != -1:
                        # Get text after the keyword
                        after_keyword = content[idx + len(keyword):].strip()
                        # Take first few words as the interest
                        interest_words = after_keyword.split()[:3]  # Take up to 3 words
                        interest = " ".join(interest_words).strip(".,!?")
                        if interest and len(interest) > 2:
                            # Clean up common words
                            if interest.lower() not in ["the", "a", "an", "user", "is", "are"]:
                                if interest not in context["interests"]:
                                    context["interests"].append(interest)
            
            # Also check for common topics in content
            common_topics = [
                "basketball", "football", "soccer", "baseball", "tennis", "sports",
                "tech", "technology", "ai", "artificial intelligence",
                "finance", "stocks", "investing", "trading", "crypto", "bitcoin",
                "politics", "movies", "music", "gaming", "games",
                "nvidia", "tesla", "apple", "microsoft",
                "cats", "dogs", "pets", "animals"
            ]
            
            for topic in common_topics:
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
    preference_value: str
) -> dict:
    """
    Store a specific user preference for easy retrieval.
    
    Args:
        user_id: The user ID
        preference_type: Type of preference (e.g., "name", "favorite_topics", "reading_style")
        preference_value: The actual preference value
    
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
        user_id=user_id
    )


@tool 
def check_memmachine_health() -> dict:
    """
    Check if MemMachine service is running and accessible.
    
    Returns:
        dict with health status
    """
    try:
        client = _get_client()
        if client is None:
            return {
                "status": "unavailable",
                "url": MEMMACHINE_URL,
                "error": "MemMachineClient not available",
                "message": "Make sure memmachine package is installed and MemMachine server is running"
            }
        
        # Use the correct v2 API health endpoint directly
        import requests
        health_url = f"{MEMMACHINE_URL}/api/v2/health"
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        health_data = response.json()
        
        return {
            "status": "healthy",
            "url": MEMMACHINE_URL,
            "response_code": 200,
            "details": health_data
        }
    except Exception as e:
        return {
            "status": "unavailable",
            "url": MEMMACHINE_URL,
            "error": str(e),
            "message": "Make sure MemMachine server is running on localhost:8080"
        }
