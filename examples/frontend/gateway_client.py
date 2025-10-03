import os
from datetime import datetime

import requests

EXAMPLE_SERVER_PORT = os.getenv("EXAMPLE_SERVER_PORT", "http://localhost:8000")


def ingest_and_rewrite(user_id: str, query: str, model_type: str = "openai") -> str:
    """Pass a raw user message through the memory server and get context-aware response."""
    print("entered ingest_and_rewrite")

    resp = requests.post(
        f"{EXAMPLE_SERVER_PORT}/memory/store-and-search",
        params={"user_id": user_id, "query": query},
        timeout=1000,
    )
    resp.raise_for_status()

    return resp.text


def add_session_message(user_id: str, msg: str) -> None:
    """Add a raw message into memory via memory server."""
    requests.post(
        f"{EXAMPLE_SERVER_PORT}/memory",
        params={"user_id": user_id, "query": msg},
        timeout=5,
    )


def create_persona_query(user_id: str, query: str) -> str:
    """Create a persona-aware query by searching memory context via memory server."""
    resp = requests.get(
        f"{EXAMPLE_SERVER_PORT}/memory",
        params={
            "query": query,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
        },
        timeout=1000,
    )
    resp.raise_for_status()

    search_results = resp.json()

    if search_results.get("profile"):
        return f"Based on your profile: {search_results['profile']}\n\nQuery: {query}"
    else:
        return f"Query: {query}"


def add_new_session_message(user_id: str, msg: str) -> None:
    """Alias for add_session_message for backward compatibility."""
    add_session_message(user_id, msg)


def delete_profile(user_id: str) -> bool:
    """Delete all memory for the given user_id via the CRM server."""
    # NOT IMPLEMENTED
    return False
