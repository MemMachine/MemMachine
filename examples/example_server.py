from http.client import HTTPException
import os
import requests
from datetime import datetime
from urllib.parse import urljoin
from fastapi import FastAPI
from default_query_constructor import DefaultQueryConstructor

from jaeger_client import Config


def init_tracer(service_name='example_server', jaeger_host='jaeger', jaeger_port=6831):
    """Initialize and return the global tracer"""
    print(f"Initializing Jaeger tracer with host '{jaeger_host}' and port {jaeger_port}")
    config = Config(
        config={
            'sampler': {'type': 'const', 'param': 1},
            'logging': True,
            'local_agent': {
                'reporting_host': jaeger_host,
                'reporting_port': jaeger_port,
            },
        },
        service_name=service_name,
        validate=True
    )
    tracer = config.initialize_tracer()
    return tracer
 
service_name= os.getenv("JAEGER_SERVICE_NAME", "example_server")
jaeger_host = os.getenv("JAEGER_HOST", "jaeger")
## with no jaeger running, the span is not send - which is fine
tracer = init_tracer(jaeger_host=jaeger_host, service_name=service_name)

# Configuration
MEMORY_BACKEND_URL = os.getenv("MEMORY_BACKEND_URL", "http://memmachine:8080")
EXAMPLE_SERVER_PORT = int(os.getenv("EXAMPLE_SERVER_PORT", "8000"))

app = FastAPI(title="Server", description="Simple middleware")

query_constructor = DefaultQueryConstructor()

# === Health Check Endpoint ===
@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    print("Health check endpoint called")
    with tracer.start_active_span("health") as scope:
        span = scope.span
        span.set_tag("http.method", "GET")
        span.set_tag("http.url", "/health")
        try:
            
            # Basic health check - could be extended to check database connectivity
            span.log_kv({"event": "health_check", "value": "checking service health"})
            span.set_tag("http.status_code", 200)
            return {
                "status": "healthy",
                "service": "example_server",
                "version": "1.0.0",
            }
        except Exception as e:
            span.set_tag("http.status_code", 503)
            span.log_kv({"event": "error", "error.object": str(e)})
            raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/memory")
async def store_data(user_id: str, query: str):
    with tracer.start_active_span("memory") as scope:
        span = scope.span
        span.set_tag("http.method", "GET")
        span.set_tag("http.url", "/memory")
    
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
            url = urljoin(MEMORY_BACKEND_URL, "/v1/memories")
            print(f"DEBUG: Sending POST request to {url} with data: {episode_data}")
            response = requests.post(
                url, json=episode_data, timeout=1000
            )
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "message": str(e)}


@app.get("/memory")
async def get_data(query: str, user_id: str, timestamp: str):
    with tracer.start_active_span("memory") as scope:
        span = scope.span
        span.set_tag("http.method", "GET")
        span.set_tag("http.url", "/memory")
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

            print(f"DEBUG: Sending POST request to {MEMORY_BACKEND_URL}/v1/memories/search")
            print(f"DEBUG: Search data: {search_data}")
            url = urljoin(MEMORY_BACKEND_URL, "/v1/memories/search")
            response = requests.post(
            url, json=search_data, timeout=1000
            )

            print(f"DEBUG: Response status: {response.status_code}")
            print(f"DEBUG: Response headers: {dict(response.headers)}")

            if response.status_code != 200:
                print(f"DEBUG: Error response body: {response.text}")
                return {
                    "status": "error",
                    "message": f"Backend returned {response.status_code}: {response.text}",
                }

            response_data = response.json()
            print(f"DEBUG: Response data: {response_data}")

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
        except Exception as e:
            print(f"DEBUG: Exception occurred: {str(e)}")
            print(f"DEBUG: Exception type: {type(e)}")
            import traceback

            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return {"status": "error", "message": str(e)}


@app.post("/memory/store-and-search")
async def store_and_search_data(user_id: str, query: str):
    with tracer.start_active_span("store_and_search") as scope:
        span = scope.span
        span.set_tag("http.method", "POST")
        span.set_tag("http.url", "/memory/store-and-search")
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
            print("DEBUG: Teststring - store_and_search_data called")
            url = urljoin(MEMORY_BACKEND_URL, "/v1/memories")
            postSpan = tracer.start_span("post_episode_data", child_of=span)
            postSpan.set_tag("episode_data", str(episode_data))
            resp = requests.post(
                url, json=episode_data, timeout=1000
            )
            postSpan.finish()
            print(f"DEBUG: Store-and-search response status: {resp.status_code}")
            if resp.status_code != 200:
                print(f"DEBUG: Store-and-search error response: {resp.text}")
                return {
                    "status": "error",
                    "message": f"Store failed with {resp.status_code}: {resp.text}",
                }

            search_data = {
                "session": session_data,
                "query": query,
                "limit": 5,
                "filter": {"producer_id": user_id},
            }
            searchSpan = tracer.start_span("post_memory_search", child_of=span)
            searchSpan.set_tag("search_data", str(search_data))

            search_resp = requests.post(
            urljoin(MEMORY_BACKEND_URL, "/v1/memories/search"), json=search_data, timeout=1000
            )
            searchSpan.set_tag("http.status_code", search_resp.status_code)
            print(f"DEBUG: Store-and-search response status: {search_resp.status_code}")
            if search_resp.status_code != 200:
                print(f"DEBUG: Store-and-search error response: {search_resp.text}")
                searchSpan.finish()
                return {
                    "status": "error",
                    "message": f"Search failed with {search_resp.status_code}: {search_resp.text}",
                }
            searchSpan.finish()
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

        except Exception as e:
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=EXAMPLE_SERVER_PORT)
