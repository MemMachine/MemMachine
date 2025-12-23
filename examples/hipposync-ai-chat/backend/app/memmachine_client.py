"""
MemMachine V2 Client - Hybrid Architecture
Supports both personal (user-level) and project-level (team) memory organization.

KEY CHANGES FROM V1:
- profile_memory → semantic_memory
- New project-based organization
- Hybrid approach: personal threads use user org, project threads use shared org
- Uses /v2/memories endpoint for unified storage
"""
import requests
import logging
from typing import Optional, Dict, Any, List
from app.config import settings

logger = logging.getLogger(__name__)

# MemMachine V2 Configuration
BASE = settings.memmachine_base_url
APP_ORG_ID = "HippoSync"  # Shared org for team projects


def _get_org_and_project(user_email: str, project_id: Optional[int] = None, project_owner_email: Optional[str] = None) -> tuple[str, str]:
    """
    Determine org_id and project_id based on context.
    
    HYBRID ARCHITECTURE:
    - Personal threads: org_id = "user-{email}", project_id = "personal"
    - Team projects: org_id = "org-{owner_email}", project_id = "proj-{id}-{project_name}"
    
    Args:
        user_email: User's email address (used as unique identifier)
        project_id: Optional project ID (if None, assumes personal)
        project_owner_email: Project owner's email (for team projects)
    
    Returns:
        Tuple of (org_id, mm_project_id)
    """
    # Sanitize email for use as org_id (replace @ and . with -)
    sanitized_email = user_email.replace('@', '-at-').replace('.', '-')
    
    if project_id is None:
        # Personal scope: user's own org based on email
        return f"user-{sanitized_email}", "personal"
    else:
        # Team scope: organization based on project owner
        if project_owner_email:
            sanitized_owner = project_owner_email.replace('@', '-at-').replace('.', '-')
            return f"org-{sanitized_owner}", f"proj-{project_id}"
        else:
            # Fallback: use shared org (backward compatibility)
            return APP_ORG_ID, f"proj-{project_id}"


def ensure_project_exists(user_email: str, project_id: Optional[int] = None, project_owner_email: Optional[str] = None) -> bool:
    """
    Ensure a project exists in MemMachine V2.
    Creates if it doesn't exist.
    
    Args:
        user_email: User's email address
        project_id: Optional project ID
        project_owner_email: Project owner's email (for team projects)
    
    Returns:
        bool: Success status
    """
    org_id, mm_project_id = _get_org_and_project(user_email, project_id, project_owner_email)
    
    try:
        # Check if project exists
        response = requests.post(
            f"{BASE}/api/v2/projects/get",
            json={"org_id": org_id, "project_id": mm_project_id},
            timeout=5
        )
        
        if response.status_code == 200:
            return True
        
        # Project doesn't exist, create it
        if project_id is None:
            project_name = f"Personal Chat - {user_email}"
        else:
            owner_display = project_owner_email or user_email
            project_name = f"Project {project_id} - Owner: {owner_display}"
        
        create_response = requests.post(
            f"{BASE}/api/v2/projects",
            json={
                "org_id": org_id,
                "project_id": mm_project_id,
                "name": project_name,
                "description": f"Memory space for {project_name}"
            },
            timeout=5
        )
        create_response.raise_for_status()
        
        logger.info(f"✅ Created project: org={org_id}, project={mm_project_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to ensure project exists: {e}")
        return False


def add_memory(
    user_email: str,
    content: str,
    thread_id: int,
    project_id: Optional[int] = None,
    project_owner_email: Optional[str] = None,
    role: str = "user",
    metadata: Optional[Dict] = None,
    store_as_semantic: bool = False
) -> bool:
    """
    Add episodic memory to MemMachine V2.
    Used for conversation messages and interactions.
    
    Args:
        user_email: User's email address
        content: Message content
        thread_id: Thread/conversation ID
        project_id: Optional project ID
        project_owner_email: Project owner's email (for team projects)
        role: Message role (user/assistant)
        metadata: Additional metadata
        store_as_semantic: If True, also store in semantic memory
    
    Returns:
        bool: Success status
    """
    org_id, mm_project_id = _get_org_and_project(user_email, project_id, project_owner_email)
    
    # Ensure project exists
    ensure_project_exists(user_email, project_id, project_owner_email)
    
    # Build metadata - convert all values to strings (V2 requirement)
    meta = metadata or {}
    meta.update({
        "thread_id": str(thread_id),
        "user_email": user_email,
        "role": role,
        "source": "HippoSync"
    })
    
    if project_id:
        meta["project_id"] = str(project_id)
        meta["shared"] = "true"
        if project_owner_email:
            meta["project_owner"] = project_owner_email
    
    # Convert any integer values in metadata to strings (V2 requirement)
    for key, value in list(meta.items()):
        if isinstance(value, (int, bool)):
            meta[key] = str(value).lower() if isinstance(value, bool) else str(value)
    
    # Build message payload for V2 API
    message = {
        "content": content,
        "producer": "user" if role == "user" else "assistant",
        "produced_for": f"ai-thread-{thread_id}",
        "role": role,
        "metadata": meta
    }
    
    # V2 payload structure
    payload = {
        "org_id": org_id,
        "project_id": mm_project_id,
        "messages": [message]
    }
    
    try:
        # Store using V2 unified endpoint
        response = requests.post(
            f"{BASE}/api/v2/memories",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        
        # If semantic storage requested, store in semantic memory too
        if store_as_semantic:
            add_semantic_memory(user_email, content, project_id, project_owner_email, meta)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to store memory: {e}")
        return False


def add_semantic_memory(
    user_email: str,
    content: str,
    project_id: Optional[int] = None,
    project_owner_email: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> bool:
    """
    Add semantic (long-term) memory to MemMachine V2.
    Used for:
    - Document content
    - Project guidelines
    - User preferences
    - Long-term knowledge
    
    Args:
        user_email: User's email address
        content: Semantic content
        project_id: Optional project ID
        project_owner_email: Project owner's email (for team projects)
        metadata: Additional metadata
    
    Returns:
        bool: Success status
    """
    org_id, mm_project_id = _get_org_and_project(user_email, project_id, project_owner_email)
    
    # Ensure project exists
    ensure_project_exists(user_email, project_id, project_owner_email)
    
    # Build metadata - convert all values to strings (V2 requirement)
    meta = metadata or {}
    
    # Convert all metadata values to strings (V2 requirement)
    meta.update({
        "user_email": user_email,
        "type": "semantic",
        "source": "HippoSync"
    })
    
    if project_id:
        meta["project_id"] = str(project_id)
        meta["shared"] = "true"
        if project_owner_email:
            meta["project_owner"] = project_owner_email
    
    # Add thread_id if present in metadata
    if "thread_id" in meta:
        meta["thread_id"] = str(meta["thread_id"])
    
    # Convert any other integer values to strings
    for key, value in meta.items():
        if isinstance(value, int):
            meta[key] = str(value)
    
    # Build semantic message
    message = {
        "content": content,
        "producer": "system",
        "produced_for": f"semantic-{user_email}",
        "role": "semantic",
        "metadata": meta
    }
    
    # V2 payload
    payload = {
        "org_id": org_id,
        "project_id": mm_project_id,
        "messages": [message]
    }
    
    try:
        # Use semantic-specific endpoint
        response = requests.post(
            f"{BASE}/api/v2/memories/semantic/add",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to store semantic memory: {e}")
        return False


def search_memories(
    user_email: str,
    query: str,
    project_id: Optional[int] = None,
    project_owner_email: Optional[str] = None,
    thread_id: Optional[int] = None,
    limit: int = 20,
    search_semantic: bool = True,
    search_episodic: bool = True
) -> Dict[str, Any]:
    """
    Search memories in MemMachine V2 with cross-thread support.
    
    CROSS-THREAD SEARCH:
    - If thread_id is None: Searches ALL threads in the scope
    - If thread_id is provided: Optionally filter by thread in post-processing
    
    Args:
        user_email: User's email address
        query: Search query
        project_id: Optional project ID (changes search scope)
        project_owner_email: Project owner's email (for team projects)
        thread_id: Optional thread ID (for filtering, not API-level restriction)
        limit: Max results to return
        search_semantic: Include semantic memories
        search_episodic: Include episodic memories
    
    Returns:
        Dict with episodic_results and semantic_results
    """
    org_id, mm_project_id = _get_org_and_project(user_email, project_id, project_owner_email)
    
    results = {
        "episodic_results": [],
        "semantic_results": [],
        "total": 0
    }
    
    try:
        # Build search types
        types = []
        if search_semantic:
            types.append("semantic")
        if search_episodic:
            types.append("episodic")
        
        if not types:
            return results
        
        # V2 search payload
        payload = {
            "org_id": org_id,
            "project_id": mm_project_id,
            "query": query,
            "top_k": limit,
            "types": types,
            "filter": "",
            "group_by": None,
            "rerank": True,
            "include_metadata": True
        }
        
        response = requests.post(
            f"{BASE}/api/v2/memories/search",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Parse V2 response format
        if isinstance(data, dict):
            content = data.get("content", {})
            
            # Extract episodic memories (V2 structure: episodic_memory with nested episodes)
            if "episodic_memory" in content:
                episodic_data = content["episodic_memory"]
                
                # V2 has both long_term_memory and short_term_memory
                all_episodes = []
                
                if isinstance(episodic_data, dict):
                    # Get long-term memory episodes
                    if "long_term_memory" in episodic_data:
                        ltm = episodic_data["long_term_memory"]
                        if isinstance(ltm, dict) and "episodes" in ltm:
                            all_episodes.extend(ltm["episodes"])
                    
                    # Get short-term memory episodes
                    if "short_term_memory" in episodic_data:
                        stm = episodic_data["short_term_memory"]
                        if isinstance(stm, dict) and "episodes" in stm:
                            all_episodes.extend(stm["episodes"])
                
                # Parse episodes
                for episode in all_episodes:
                    if isinstance(episode, dict) and "content" in episode:
                        memory_item = {
                            "episode_content": episode["content"],
                            "metadata": episode.get("metadata", {}),
                            "episode_type": episode.get("episode_type", "message"),
                            "score": episode.get("score", 1.0),
                            "created_at": episode.get("created_at", "")
                        }
                        
                        # Filter by thread_id if specified (client-side filtering)
                        if thread_id:
                            item_thread_id = memory_item["metadata"].get("thread_id")
                            if item_thread_id and str(item_thread_id) == str(thread_id):
                                results["episodic_results"].append(memory_item)
                        else:
                            # No thread filter - include all (cross-thread search)
                            results["episodic_results"].append(memory_item)
            
            # Extract semantic memories (V2 structure: semantic_memory array with "value" field)
            if "semantic_memory" in content:
                semantic_data = content["semantic_memory"]
                if isinstance(semantic_data, list):
                    for item in semantic_data:
                        if isinstance(item, dict) and "value" in item:
                            memory_item = {
                                "profile_content": item["value"],
                                "metadata": item.get("metadata", {}),
                                "profile_type": item.get("category", "profile"),
                                "feature_name": item.get("feature_name", ""),
                                "tag": item.get("tag", ""),
                                "score": 1.0
                            }
                            results["semantic_results"].append(memory_item)
        
        results["total"] = len(results["episodic_results"]) + len(results["semantic_results"])
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Memory search failed: {e}")
        return results


def get_project_episode_count(
    user_email: str,
    project_id: Optional[int] = None,
    project_owner_email: Optional[str] = None
) -> int:
    """
    Get episode count for a project in MemMachine V2.
    
    Args:
        user_email: User's email address
        project_id: Optional project ID
        project_owner_email: Project owner's email (for team projects)
    
    Returns:
        int: Number of episodes
    """
    org_id, mm_project_id = _get_org_and_project(user_email, project_id, project_owner_email)
    
    try:
        response = requests.post(
            f"{BASE}/api/v2/projects/get",
            json={"org_id": org_id, "project_id": mm_project_id},
            timeout=5
        )
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, dict):
            content = data.get("content", {})
            return content.get("episode_count", 0)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Failed to get episode count: {e}")
        return 0


def delete_project_memories(
    user_email: str,
    project_id: Optional[int] = None,
    project_owner_email: Optional[str] = None
) -> bool:
    """
    Delete all memories for a project in MemMachine V2.
    
    Args:
        user_email: User's email address
        project_id: Optional project ID
        project_owner_email: Project owner's email (for team projects)
    
    Returns:
        bool: Success status
    """
    org_id, mm_project_id = _get_org_and_project(user_email, project_id, project_owner_email)
    
    try:
        response = requests.delete(
            f"{BASE}/api/v2/projects",
            json={"org_id": org_id, "project_id": mm_project_id},
            timeout=10
        )
        response.raise_for_status()
        
        logger.info(f"✅ Deleted project: org={org_id}, project={mm_project_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to delete project: {e}")
        return False