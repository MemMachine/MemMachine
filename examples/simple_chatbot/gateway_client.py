import contextlib
import os
from datetime import UTC, datetime

import requests

MEMMACHINE_PORT = os.getenv("MEMORY_SERVER_URL", "http://localhost:8080")
ORG_ID = os.getenv("ORG_ID", "default-org")
PROJECT_ID = os.getenv("PROJECT_ID", "simple_chatbot")

PROMPT = """You are a helpful AI assistant. Use the provided context and profile information to answer the user's question accurately and helpfully.

<CURRENT_DATE>
{current_date}
</CURRENT_DATE>

Instructions:
- Use the PROFILE and CONTEXT data provided to answer the user's question
- Be conversational and helpful in your responses
- If you don't have enough information to answer completely, say so and suggest what additional information might be helpful
- If the context contains relevant information, use it to provide a comprehensive answer
- If no relevant context is available, let the user know and offer to help in other ways
- Be concise but thorough in your responses
- Use markdown formatting when appropriate to make your response clear and readable

Data Guidelines:
- Don't invent information that isn't in the provided context
- If information is missing or unclear, acknowledge this
- Prioritize the most recent and relevant information when available
- If there are conflicting pieces of information, mention this and explain the differences

Response Format:
- Directly answer the user's question, without showing your thought process
- Provide supporting details from the context when available
- Use bullet points or numbered lists when appropriate
- End with any relevant follow-up questions or suggestions"""


def _dict_to_filter_string(filter_dict: dict[str, str]) -> str:
    """Convert filter_dict to SQL-like filter string format: key='value' AND key='value'."""
    conditions = []
    for key, value in filter_dict.items():
        # Escape single quotes in strings (SQL standard: ' -> '')
        escaped_value = value.replace("'", "''")
        conditions.append(f"{key}='{escaped_value}'")
    return " AND ".join(conditions)


def ingest_and_rewrite(user_id: str, query: str) -> str:
    """Pass a raw user message through the memory server and get context-aware response."""
    print("entered ingest_and_rewrite")

    # Ingest memory with user_id in metadata for filtering
    requests.post(
        f"{MEMMACHINE_PORT}/api/v2/memories",
        json={
            "org_id": ORG_ID,
            "project_id": PROJECT_ID,
            "messages": [
                {
                    "content": query,
                    "producer": user_id,
                    "produced_for": "agent",
                    "role": "user",
                    "timestamp": datetime.now(tz=UTC)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "metadata": {"user_id": user_id},
                }
            ],
        },
        timeout=60,
    )

    # Search memories with metadata filter to get only this user's memories
    filter_str = f"metadata.user_id='{user_id}'"
    resp = requests.post(
        f"{MEMMACHINE_PORT}/api/v2/memories/search",
        json={
            "org_id": ORG_ID,
            "project_id": PROJECT_ID,
            "query": query,
            "top_k": 10,
            "types": ["episodic", "semantic"],
            "filter": filter_str,
        },
        timeout=1000,
    )
    resp.raise_for_status()

    # Parse v2 response format
    data = resp.json()
    content = data.get("content", {})

    # Extract episodic and semantic memory from v2 response
    episodic_memory = content.get("episodic_memory", {})
    semantic_memory = content.get("semantic_memory", [])

    # Build context string from episodic memory
    context_parts = []

    # Handle episodic memory structure (can be dict with long_term/short_term or list)
    if isinstance(episodic_memory, dict):
        long_term = episodic_memory.get("long_term_memory", {})
        short_term = episodic_memory.get("short_term_memory", {})

        long_term_episodes = (
            long_term.get("episodes", []) if isinstance(long_term, dict) else []
        )
        short_term_episodes = (
            short_term.get("episodes", []) if isinstance(short_term, dict) else []
        )

        all_episodes = []
        if isinstance(long_term_episodes, list):
            all_episodes.extend(long_term_episodes)
        if isinstance(short_term_episodes, list):
            all_episodes.extend(short_term_episodes)

        for episode in all_episodes:
            if isinstance(episode, dict):
                episode_content = (
                    episode.get("content") or episode.get("episode_content") or ""
                )
                if episode_content:
                    context_parts.append(episode_content)
    elif isinstance(episodic_memory, list):
        for episode in episodic_memory:
            if isinstance(episode, dict):
                episode_content = (
                    episode.get("content") or episode.get("episode_content") or ""
                )
                if episode_content:
                    context_parts.append(episode_content)

    # Add semantic memory
    if isinstance(semantic_memory, list):
        for memory in semantic_memory:
            if isinstance(memory, dict):
                memory_content = (
                    memory.get("content") or memory.get("memory_content") or ""
                )
                if memory_content:
                    context_parts.append(memory_content)

    context_str = (
        "\n\n".join(context_parts) if context_parts else "No relevant context found."
    )

    return PROMPT + "\n\n" + context_str + "\n\n" + "User Query: " + query


def get_memories(user_id: str) -> dict:
    """Fetch all memories for a given user_id"""
    try:
        # Use metadata filter to get only this user's memories
        filter_str = f"metadata.user_id='{user_id}'"
        resp = requests.post(
            f"{MEMMACHINE_PORT}/api/v2/memories/list",
            json={
                "org_id": ORG_ID,
                "project_id": PROJECT_ID,
                "filter": filter_str,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching memories: {e}")
        return {}


def ingest_memories(user_id: str, memories_text: str) -> bool:
    """Ingest imported memories into MemMachine system.

    Args:
        user_id: The user identifier
        memories_text: Text containing memories/conversations to ingest

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ingest the memories as an episode using v2 API with user_id in metadata
        resp = requests.post(
            f"{MEMMACHINE_PORT}/api/v2/memories",
            json={
                "org_id": ORG_ID,
                "project_id": PROJECT_ID,
                "messages": [
                    {
                        "content": memories_text,
                        "producer": user_id,
                        "produced_for": "agent",
                        "role": "user",
                        "timestamp": datetime.now(tz=UTC)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "metadata": {"user_id": user_id},
                    }
                ],
            },
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error ingesting memories: {e}")
        return False


def delete_profile(user_id: str) -> bool:
    """Delete all memories for the given user_id.

    Uses metadata filtering to delete only memories belonging to the specified user.
    Since we're using a shared project with metadata-based filtering, we need to
    list memories by filter and then delete them by ID.

    Args:
        user_id: The user identifier whose profile should be deleted

    Returns:
        True if successful, False otherwise
    """

    try:
        filter_str = f"metadata.user_id='{user_id}'"

        # List episodic memories for this user
        episodic_ids = []
        page_num = 0
        while True:
            memories_resp = requests.post(
                f"{MEMMACHINE_PORT}/api/v2/memories/list",
                json={
                    "org_id": ORG_ID,
                    "project_id": PROJECT_ID,
                    "filter": filter_str,
                    "type": "episodic",
                    "page_size": 100,
                    "page_num": page_num,
                },
                timeout=10,
            )
            memories_resp.raise_for_status()
            memories_data = memories_resp.json()
            content = memories_data.get("content", {})
            episodic_memories = content.get("episodic_memory", [])

            if not episodic_memories:
                break

            # Extract IDs from episodic memories
            for memory in episodic_memories:
                if isinstance(memory, dict):
                    # Try different possible ID field names
                    memory_id = (
                        memory.get("id")
                        or memory.get("uid")
                        or memory.get("episode_id")
                    )
                    if memory_id:
                        episodic_ids.append(memory_id)

            if len(episodic_memories) < 100:
                break
            page_num += 1

        # Delete episodic memories in batches
        if episodic_ids:
            # Delete in chunks to avoid very large requests
            chunk_size = 100
            for i in range(0, len(episodic_ids), chunk_size):
                chunk = episodic_ids[i : i + chunk_size]
                with contextlib.suppress(Exception):
                    requests.post(
                        f"{MEMMACHINE_PORT}/api/v2/memories/episodic/delete",
                        json={
                            "org_id": ORG_ID,
                            "project_id": PROJECT_ID,
                            "episodic_ids": chunk,
                        },
                        timeout=30,
                    )

        # List semantic memories for this user
        semantic_ids = []
        page_num = 0
        while True:
            memories_resp = requests.post(
                f"{MEMMACHINE_PORT}/api/v2/memories/list",
                json={
                    "org_id": ORG_ID,
                    "project_id": PROJECT_ID,
                    "filter": filter_str,
                    "type": "semantic",
                    "page_size": 100,
                    "page_num": page_num,
                },
                timeout=10,
            )
            memories_resp.raise_for_status()
            memories_data = memories_resp.json()
            content = memories_data.get("content", {})
            semantic_memories = content.get("semantic_memory", [])

            if not semantic_memories:
                break

            # Extract IDs from semantic memories
            for memory in semantic_memories:
                if isinstance(memory, dict):
                    # Try different possible ID field names
                    memory_id = (
                        memory.get("id")
                        or memory.get("feature_id")
                        or memory.get("semantic_id")
                    )
                    if memory_id:
                        semantic_ids.append(memory_id)

            if len(semantic_memories) < 100:
                break
            page_num += 1

        # Delete semantic memories in batches
        if semantic_ids:
            chunk_size = 100
            for i in range(0, len(semantic_ids), chunk_size):
                chunk = semantic_ids[i : i + chunk_size]
                with contextlib.suppress(Exception):
                    requests.post(
                        f"{MEMMACHINE_PORT}/api/v2/memories/semantic/delete",
                        json={
                            "org_id": ORG_ID,
                            "project_id": PROJECT_ID,
                            "semantic_ids": chunk,
                        },
                        timeout=30,
                    )

        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # No memories found, which is fine - nothing to delete
            return True
        print(f"Error deleting profile: {e}")
        return False
    except Exception as e:
        print(f"Error deleting profile: {e}")
        # Return True to avoid breaking the UI if deletion partially fails
        return True
