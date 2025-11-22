import logging
import os
from datetime import UTC, datetime

import requests
from fastapi import FastAPI
from query_constructor import FinancialAnalystQueryConstructor

logger = logging.getLogger(__name__)

# Configuration
MEMORY_BACKEND_URL = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
FINANCIAL_SERVER_PORT = int(os.getenv("FINANCIAL_SERVER_PORT", "8000"))

app = FastAPI(title="Financial Analyst Server", description="Simple middleware")

query_constructor = FinancialAnalystQueryConstructor()


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
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "type": "message",
            },
        }

        response = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories",
            json=episode_data,
            timeout=1000,
        )
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except Exception:
        logger.exception("Error occurred in /memory get_data")
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

        logger.debug(
            "Sending POST request to %s/v1/memories/search",
            MEMORY_BACKEND_URL,
        )
        logger.debug("Search data: %s", search_data)

        response = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories/search",
            json=search_data,
            timeout=1000,
        )

        logger.debug("Response status: %s", response.status_code)
        logger.debug("Response headers: %s", dict(response.headers))

        if response.status_code != 200:
            logger.error(
                "Backend returned %s: %s",
                response.status_code,
                response.text,
            )
            return {
                "status": "error",
                "message": "Failed to retrieve memory data",
            }

        response_data = response.json()
        logger.debug("Response data: %s", response_data)

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
            profile=profile_str,
            context=context_str,
            query=query,
        )

        return {
            "status": "success",
            "data": {"profile": profile_memory, "context": episodic_memory},
            "formatted_query": formatted_query,
            "query_type": "example",
        }
    except Exception:
        logger.exception("Error occurred in /memory get_data")
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
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "type": "message",
            },
        }

        resp = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories",
            json=episode_data,
            timeout=1000,
        )

        logger.debug("Store-and-search response status: %s", resp.status_code)
        if resp.status_code != 200:
            logger.error("Store failed with %s: %s", resp.status_code, resp.text)
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
            f"{MEMORY_BACKEND_URL}/v1/memories/search",
            json=search_data,
            timeout=1000,
        )

        logger.debug(
            "Store-and-search response status: %s",
            search_resp.status_code,
        )
        if search_resp.status_code != 200:
            logger.error(
                "Search failed with %s: %s", search_resp.status_code, search_resp.text
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
            profile=profile_str,
            context=context_str,
            query=query,
        )

        if profile_memory and episodic_memory:
            return f"Profile: {profile_memory}\n\nContext: {episodic_memory}\n\nFormatted Response:\n{formatted_response}"
        if profile_memory:
            return f"Profile: {profile_memory}\n\nFormatted Response:\n{formatted_response}"
        if episodic_memory:
            return f"Context: {episodic_memory}\n\nFormatted Response:\n{formatted_response}"
        return f"Message ingested successfully. No relevant context found yet.\n\nFormatted Response:\n{formatted_response}"

    except Exception:
        logger.exception("Error occurred in store_and_search_data")
        return {"status": "error", "message": "Internal error in store_and_search"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=FINANCIAL_SERVER_PORT)
