import os
from datetime import UTC, datetime

import requests

MEMMACHINE_PORT = os.getenv("MEMORY_SERVER_URL", "http://localhost:8080")
ORG_ID = os.getenv("ORG_ID", "default-org")

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

    # Use project_id per user for isolation
    project_id = f"project_{user_id}"

    # Ingest memory
    requests.post(
        f"{MEMMACHINE_PORT}/api/v2/memories",
        json={
            "org_id": ORG_ID,
            "project_id": project_id,
            "messages": [
                {
                    "content": query,
                    "producer": user_id,
                    "produced_for": "agent",
                    "role": "user",
                    "timestamp": datetime.now(tz=UTC)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "metadata": {},
                }
            ],
        },
        timeout=5,
    )

    # Search memories - no filter needed since each user has their own project
    resp = requests.post(
        f"{MEMMACHINE_PORT}/api/v2/memories/search",
        json={
            "org_id": ORG_ID,
            "project_id": project_id,
            "query": query,
            "top_k": 10,
            "types": ["episodic", "semantic"],
            "filter": "",
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
        # Use project_id per user for isolation
        project_id = f"project_{user_id}"

        resp = requests.post(
            f"{MEMMACHINE_PORT}/api/v2/memories/list",
            json={
                "org_id": ORG_ID,
                "project_id": project_id,
                "filter": "",
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
        # Use project_id per user for isolation
        project_id = f"project_{user_id}"

        # Ingest the memories as an episode using v2 API
        resp = requests.post(
            f"{MEMMACHINE_PORT}/api/v2/memories",
            json={
                "org_id": ORG_ID,
                "project_id": project_id,
                "messages": [
                    {
                        "content": memories_text,
                        "producer": user_id,
                        "produced_for": "agent",
                        "role": "user",
                        "timestamp": datetime.now(tz=UTC)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "metadata": {},
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

    Uses project-level isolation to delete the entire project, which removes
    all memories (episodic and semantic) for the user.

    Args:
        user_id: The user identifier whose profile should be deleted

    Returns:
        True if successful, False otherwise
    """
    try:
        # Use project_id per user for isolation
        project_id = f"project_{user_id}"

        # Delete the entire project, which removes all memories
        resp = requests.post(
            f"{MEMMACHINE_PORT}/api/v2/projects/delete",
            json={
                "org_id": ORG_ID,
                "project_id": project_id,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # Project doesn't exist, which is fine - nothing to delete
            return True
        print(f"Error deleting profile: {e}")
        return False
    except Exception as e:
        print(f"Error deleting profile: {e}")
        return False
