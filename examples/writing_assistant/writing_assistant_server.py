import os
import requests
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from query_constructor import WritingAssistantQueryConstructor

# Configuration
MEMORY_BACKEND_URL = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
WRITING_ASSISTANT_PORT = int(os.getenv("WRITING_ASSISTANT_PORT", "8000"))

app = FastAPI(
    title="Writing Assistant Server", description="Writing Assistant middleware"
)

writing_assistant_constructor = WritingAssistantQueryConstructor()


# Pydantic models for request/response
class MemoryRequest(BaseModel):
    user_id: str
    query: str


class MemoryResponse(BaseModel):
    status: str
    data: dict | None = None
    formatted_query: str | None = None
    query_type: str | None = None
    message: str | None = None


@app.post("/memory", response_model=MemoryResponse)
async def store_data(request: MemoryRequest):
    """Store user data and handle writing style submissions"""
    try:
        user_id = request.user_id
        query = request.query

        # Check if this is a /submit command
        submit_info = writing_assistant_constructor.detect_submit_command(query)

        session_data = {
            "group_id": None,
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
                "is_submission": submit_info["is_submission"],
                "content_type": (
                    submit_info["content_type"]
                    if submit_info["is_submission"]
                    else None
                ),
                "writing_sample": (
                    submit_info["writing_sample"]
                    if submit_info["is_submission"]
                    else None
                ),
            },
        }

        response = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories", json=episode_data, timeout=1000
        )
        response.raise_for_status()

        if submit_info["is_submission"]:
            return MemoryResponse(
                status="success",
                data=response.json(),
                message="Say this: 'Thank you for submitting your writing sample. Your writing sample has been analyzed and stored.'",
            )
        else:
            return MemoryResponse(
                status="success",
                data=response.json(),
                message="Message stored successfully",
            )

    except Exception as e:
        logging.exception("Error occurred in /memory store_data")
        return MemoryResponse(
            status="error", message=f"Internal error in /memory store_data: {str(e)}"
        )


@app.get("/memory", response_model=MemoryResponse)
async def get_data(query: str, user_id: str, timestamp: str = None):
    """Retrieve memory data and format for writing assistant"""
    try:
        session_data = {
            "group_id": None,
            "agent_id": ["assistant"],
            "user_id": [user_id],
            "session_id": f"session_{user_id}",
        }

        search_data = {
            "session": session_data,
            "query": query,
            "limit": 10,  # Increased limit to get more writing style context
            "filter": {"producer_id": user_id},
        }

        logging.debug(f"Sending POST request to {MEMORY_BACKEND_URL}/search_memory")
        logging.debug(f"Search data: {search_data}")

        response = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories/search", json=search_data, timeout=1000
        )

        logging.debug(f"Response status: {response.status_code}")
        logging.debug(f"Response headers: {dict(response.headers)}")

        if response.status_code != 200:
            logging.error(f"Backend returned {response.status_code}: {response.text}")
            return MemoryResponse(
                status="error", message="Failed to retrieve memory data"
            )

        response_data = response.json()
        logging.debug(f"Response data: {response_data}")

        content = response_data.get("content", {})
        episodic_memory = content.get("episodic_memory", [])
        profile_memory = content.get("profile_memory", [])

        # Format profile memory (writing style characteristics)
        profile_str = ""
        if profile_memory:
            if isinstance(profile_memory, list):
                profile_str = "\n".join([str(p) for p in profile_memory])
            else:
                profile_str = str(profile_memory)

        # Format context (episodic memory - conversation history)
        context_str = ""
        if episodic_memory:
            if isinstance(episodic_memory, list):
                context_str = "\n".join([str(c) for c in episodic_memory])
            else:
                context_str = str(episodic_memory)

        # Create formatted query using the writing assistant constructor
        formatted_query = writing_assistant_constructor.create_query(
            profile=profile_str, context=context_str, query=query
        )

        # Check if this is a submission request
        submit_info = writing_assistant_constructor.detect_submit_command(query)
        query_type = (
            "writing_style_submission"
            if submit_info["is_submission"]
            else "writing_assistant"
        )

        return MemoryResponse(
            status="success",
            data={"profile": profile_memory, "context": episodic_memory},
            formatted_query=formatted_query,
            query_type=query_type,
        )

    except Exception as e:
        logging.exception("Error occurred in /memory get_data")
        return MemoryResponse(
            status="error", message=f"Internal error in /memory get_data: {str(e)}"
        )


@app.post("/memory/store-and-search")
async def store_and_search_data(
    request: MemoryRequest = None, user_id: str = Query(None), query: str = Query(None)
):
    """Store user data and immediately search for relevant context"""
    try:
        # Handle both JSON body and query parameters for backward compatibility
        if request is not None:
            user_id = request.user_id
            query = request.query
        else:
            if user_id is None or query is None:
                raise HTTPException(
                    status_code=422, detail="Missing user_id or query parameter"
                )
            user_id = user_id
            query = query

        # Check if this is a /submit command
        submit_info = writing_assistant_constructor.detect_submit_command(query)

        session_data = {
            "group_id": None,
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
                "is_submission": submit_info["is_submission"],
                "content_type": (
                    submit_info["content_type"]
                    if submit_info["is_submission"]
                    else None
                ),
                "writing_sample": (
                    submit_info["writing_sample"]
                    if submit_info["is_submission"]
                    else None
                ),
            },
        }

        resp = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories", json=episode_data, timeout=1000
        )

        logging.debug(f"Store-and-search response status: {resp.status_code}")
        if resp.status_code != 200:
            logging.error(f"Store failed with {resp.status_code}: {resp.text}")
            return MemoryResponse(status="error", message="Failed to store memory data")

        # Search for relevant context
        search_data = {
            "session": session_data,
            "query": query,
            "limit": 10,  # Increased limit for better context
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
            return MemoryResponse(
                status="error", message="Failed to search memory data"
            )

        search_resp.raise_for_status()
        search_results = search_resp.json()

        content = search_results.get("content", {})
        episodic_memory = content.get("episodic_memory", [])
        profile_memory = content.get("profile_memory", [])

        # Format profile memory (writing style characteristics)
        profile_str = ""
        if profile_memory:
            if isinstance(profile_memory, list):
                profile_str = "\n".join([str(p) for p in profile_memory])
            else:
                profile_str = str(profile_memory)

        # Format context data (episodic memory)
        context_str = ""
        if episodic_memory:
            if isinstance(episodic_memory, list):
                context_str = "\n".join([str(c) for c in episodic_memory])
            else:
                context_str = str(episodic_memory)

        # Create formatted query using the writing assistant constructor
        formatted_response = writing_assistant_constructor.create_query(
            profile=profile_str, context=context_str, query=query
        )

        # Determine query type
        query_type = (
            "writing_style_submission"
            if submit_info["is_submission"]
            else "writing_assistant"
        )

        # Create response message
        if submit_info["is_submission"]:
            message = "Say this: 'Thank you for submitting your writing sample. Your writing sample has been analyzed and stored.'"
            # For submissions, return a plain text response so existing frontend shows only the message
            return PlainTextResponse(content=message)
        elif profile_memory and episodic_memory:
            message = "Profile and context data retrieved successfully"
        elif profile_memory:
            message = "Profile data retrieved successfully"
        elif episodic_memory:
            message = "Context data retrieved successfully"
        else:
            message = "Message stored successfully. No relevant context found yet."

        return MemoryResponse(
            status="success",
            data={"profile": profile_memory, "context": episodic_memory},
            formatted_query=formatted_response,
            query_type=query_type,
            message=message,
        )

    except Exception as e:
        logging.exception("Error occurred in store_and_search_data")
        return MemoryResponse(
            status="error", message=f"Internal error in store_and_search: {str(e)}"
        )


@app.post("/analyze-writing-style", response_model=MemoryResponse)
async def analyze_writing_style(request: MemoryRequest):
    """Analyze a writing sample and extract style characteristics"""
    try:
        user_id = request.user_id
        query = request.query

        # Check if this is a /submit command
        submit_info = writing_assistant_constructor.detect_submit_command(query)

        if not submit_info["is_submission"]:
            return MemoryResponse(
                status="error",
                message="This endpoint is only for writing style submissions. Use /submit command.",
            )

        # Store the submission first
        store_response = await store_data(request)
        if store_response.status != "success":
            return store_response

        # Get existing profile to understand current writing style
        session_data = {
            "group_id": None,
            "agent_id": ["assistant"],
            "user_id": [user_id],
            "session_id": f"session_{user_id}",
        }

        search_data = {
            "session": session_data,
            "query": f"writing_style_{submit_info['content_type']}",
            "limit": 20,
            "filter": {"producer_id": user_id},
        }

        search_resp = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories/search", json=search_data, timeout=1000
        )

        if search_resp.status_code != 200:
            logging.error(
                f"Search failed with {search_resp.status_code}: {search_resp.text}"
            )
            return MemoryResponse(
                status="error", message="Failed to retrieve existing writing style data"
            )

        search_results = search_resp.json()
        content = search_results.get("content", {})
        profile_memory = content.get("profile_memory", [])

        # Format existing profile
        profile_str = ""
        if profile_memory:
            if isinstance(profile_memory, list):
                profile_str = "\n".join([str(p) for p in profile_memory])
            else:
                profile_str = str(profile_memory)

        # Create specialized analysis prompt
        analysis_prompt = writing_assistant_constructor.create_submission_query(
            profile=profile_str, context="", submit_info=submit_info
        )

        return MemoryResponse(
            status="success",
            data={
                "profile": profile_memory,
                "writing_sample": submit_info["writing_sample"],
            },
            formatted_query=analysis_prompt,
            query_type="writing_style_analysis",
            message=f"Writing sample for {submit_info['content_type']} content type ready for analysis",
        )

    except Exception as e:
        logging.exception("Error occurred in analyze_writing_style")
        return MemoryResponse(
            status="error", message=f"Internal error in analyze_writing_style: {str(e)}"
        )


@app.get("/writing-styles/{user_id}")
async def get_user_writing_styles(user_id: str):
    """Get all writing styles for a user"""
    try:
        session_data = {
            "group_id": None,
            "agent_id": ["assistant"],
            "user_id": [user_id],
            "session_id": f"session_{user_id}",
        }

        # Search for all writing style profiles
        search_data = {
            "session": session_data,
            "query": "writing_style",
            "limit": 50,
            "filter": {"producer_id": user_id},
        }

        search_resp = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories/search", json=search_data, timeout=1000
        )

        if search_resp.status_code != 200:
            return {"status": "error", "message": "Failed to retrieve writing styles"}

        search_results = search_resp.json()
        content = search_results.get("content", {})
        profile_memory = content.get("profile_memory", [])

        # Organize by content type
        writing_styles = {}
        if profile_memory:
            for profile in profile_memory:
                if isinstance(profile, dict) and "tag" in profile:
                    tag = profile["tag"]
                    if tag.startswith("writing_style_"):
                        content_type = tag.replace("writing_style_", "")
                        if content_type not in writing_styles:
                            writing_styles[content_type] = []
                        writing_styles[content_type].append(profile)

        return {
            "status": "success",
            "user_id": user_id,
            "writing_styles": writing_styles,
            "available_content_types": list(writing_styles.keys()),
        }

    except Exception as e:
        logging.exception("Error occurred in get_user_writing_styles")
        return {"status": "error", "message": "Internal server error"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=WRITING_ASSISTANT_PORT)
