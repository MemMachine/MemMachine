import logging
import os
from datetime import datetime

import requests
from fastapi import FastAPI
from query_constructor import HealthAssistantQueryConstructor

# Configuration
MEMORY_BACKEND_URL = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8000"))

app = FastAPI(title="Health Server", description="Simple middleware")

query_constructor = HealthAssistantQueryConstructor()


@app.post("/memory")
async def store_data(user_id: str, query: str):
    try:
        session_data = {
            "group_id": user_id,
            "agent_id": ["assistant"],
            "user_id": [user_id],
            "session_id": f"session_{user_id}",
        }
        episode_data = {
            "session": session_data,
            "producer": user_id,
            "produced_for": "assistant",
            "episode_content": query,
            "episode_type": "message",
            "metadata": {
                "speaker": user_id,
                "timestamp": datetime.now().isoformat(),
                "type": "message",
            },
        }

        response = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories", json=episode_data, timeout=1000
        )
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except Exception:
        logging.exception("Error occurred in /memory get_data")
        return {"status": "error", "message": "Internal error in /memory get_data"}


@app.get("/memory")
async def get_data(query: str, user_id: str, timestamp: str):
    try:
        session_data = {
            "group_id": user_id,
            "agent_id": ["assistant"],
            "user_id": [user_id],
            "session_id": f"session_{user_id}",
        }
        search_data = {
            "session": session_data,
            "query": query,
            "limit": 5,
            "filter": {"producer_id": user_id},
        }

        logging.debug(
            f"Sending POST request to {MEMORY_BACKEND_URL}/v1/memories/search"
        )
        logging.debug(f"Search data: {search_data}")

        response = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories/search", json=search_data, timeout=1000
        )

        logging.debug(f"Response status: {response.status_code}")
        logging.debug(f"Response headers: {dict(response.headers)}")

        if response.status_code != 200:
            logging.error(f"Backend returned {response.status_code}: {response.text}")
            return {
                "status": "error",
                "message": "Failed to retrieve memory data",
            }

        response_data = response.json()
        logging.debug(f"Response data: {response_data}")

        content = response_data.get("content", {})
        episodic_memory = content.get("episodic_memory", [])
        profile_memory = content.get("profile_memory", [])

        profile_str = ""
        if profile_memory:
            if isinstance(profile_memory, list):
                profile_str = "\n".join([str(p) for p in profile_memory])
            else:
                profile_str = str(profile_memory)

        context_str = ""
        if episodic_memory:
            if isinstance(episodic_memory, list):
                context_str = "\n".join([str(c) for c in episodic_memory])
            else:
                context_str = str(episodic_memory)

        formatted_query = query_constructor.create_query(
            profile=profile_str, context=context_str, query=query
        )

        return {
            "status": "success",
            "data": {"profile": profile_memory, "context": episodic_memory},
            "formatted_query": formatted_query,
            "query_type": "example",
        }
    except Exception:
        logging.exception("Error occurred in /memory get_data")
        return {"status": "error", "message": "Internal error in /memory get_data"}


@app.post("/memory/store-and-search")
async def store_and_search_data(user_id: str, query: str):
    try:
        session_data = {
            "group_id": user_id,
            "agent_id": ["assistant"],
            "user_id": [user_id],
            "session_id": f"session_{user_id}",
        }
        episode_data = {
            "session": session_data,
            "producer": user_id,
            "produced_for": "assistant",
            "episode_content": query,
            "episode_type": "message",
            "metadata": {
                "speaker": user_id,
                "timestamp": datetime.now().isoformat(),
                "type": "message",
            },
        }

        resp = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories", json=episode_data, timeout=1000
        )

        logging.debug(f"Store-and-search response status: {resp.status_code}")
        if resp.status_code != 200:
            logging.error(f"Store failed with {resp.status_code}: {resp.text}")
            return {
                "status": "error",
                "message": "Failed to store memory data",
            }

        search_data = {
            "session": session_data,
            "query": query,
            "limit": 5,
            "filter": {"producer_id": user_id},
        }

        search_resp = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories/search", json=search_data, timeout=1000
        )

        logging.debug(f"Store-and-search response status: {search_resp.status_code}")
        if search_resp.status_code != 200:
            logging.error(
                f"Search failed with {search_resp.status_code}: {search_resp.text}"
            )
            return {
                "status": "error",
                "message": "Failed to search memory data",
            }

        search_resp.raise_for_status()

        search_results = search_resp.json()

        content = search_results.get("content", {})
        episodic_memory = content.get("episodic_memory", [])
        profile_memory = content.get("profile_memory", [])

        profile_str = ""
        if profile_memory:
            if isinstance(profile_memory, list):
                profile_str = "\n".join([str(p) for p in profile_memory])
            else:
                profile_str = str(profile_memory)

        context_str = ""
        if episodic_memory:
            if isinstance(episodic_memory, list):
                context_str = "\n".join([str(c) for c in episodic_memory])
            else:
                context_str = str(episodic_memory)

        formatted_response = query_constructor.create_query(
            profile=profile_str, context=context_str, query=query
        )

        if profile_memory and episodic_memory:
            return f"Profile: {profile_memory}\n\nContext: {episodic_memory}\n\nFormatted Response:\n{formatted_response}"
        elif profile_memory:
            return f"Profile: {profile_memory}\n\nFormatted Response:\n{formatted_response}"
        elif episodic_memory:
            return f"Context: {episodic_memory}\n\nFormatted Response:\n{formatted_response}"
        else:
            return f"Message ingested successfully. No relevant context found yet.\n\nFormatted Response:\n{formatted_response}"

    except Exception:
        logging.exception("Error occurred in store_and_search_data")
        return {"status": "error", "message": "Internal error in store_and_search"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=HEALTH_PORT)
