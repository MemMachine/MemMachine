import json
import os
import time
from datetime import datetime

import requests

episodic_memory_path = "memories/episodic"


class MemMachineRestClient:
    def __init__(
        self,
        base_url="http://localhost:8080",
        session=None,
        producer=None,
        produced_for=None,
        verbose=False,
        statistic_file=None,
    ):
        self.base_url = base_url
        self.api_version = "v1"
        self.session = session
        if self.session is None:
            self.session = {
                "group_id": "test_group",
                "agent_id": ["test_agent"],
                "user_id": ["test_user"],
                "session_id": "session_123",
            }
        self.producer = producer
        if self.producer is None:
            self.producer = "test_user"
        self.produced_for = produced_for
        if self.produced_for is None:
            self.produced_for = "test_agent"
        self.verbose = verbose
        self.statistic_file = statistic_file
        if self.statistic_file is None:
            timestamp = datetime.now().isoformat()
            self.statistic_file = f"output/statistic_{timestamp}.csv"
        if not os.path.exists(self.statistic_file):
            os.makedirs(os.path.dirname(self.statistic_file), exist_ok=True)
        with open(self.statistic_file, "w") as f:
            f.write("timestamp,method,url,latency_ms\n")
        self.statistic_fp = open(self.statistic_file, "a")

    def __del__(self):
        self.statistic_fp.close()

    def _get_url(self, path):
        return f"{self.base_url}/{self.api_version}/{path}"

    def _trace_request(self, method, url, payload=None, response=None, latency_ms=None):
        """Trace API request details including latency and response info"""
        timestamp = datetime.now().isoformat()

        trace_info = {
            "timestamp": timestamp,
            "method": method,
            "url": url,
            "latency_ms": latency_ms,
            "request_size_bytes": (
                len(json.dumps(payload).encode("utf-8")) if payload else 0
            ),
            "response_size_bytes": len(response.content) if response else 0,
            "status_code": response.status_code if response else None,
            "response_headers": dict(response.headers) if response else None,
        }

        print(f"\n🔍 API TRACE [{timestamp}]")
        print(f"   Method: {method}")
        print(f"   URL: {url}")
        print(f"   Latency: {latency_ms}ms" if latency_ms else "   Latency: N/A")
        print(f"   Request Size: {trace_info['request_size_bytes']} bytes")
        print(f"   Response Size: {trace_info['response_size_bytes']} bytes")
        print(f"   Status Code: {trace_info['status_code']}")

        if response and response.headers:
            print(f"   Response Headers: {dict(response.headers)}")

        return trace_info

    """
    curl -X POST "http://localhost:8080/v1/memories/episodic" \
    -H "Content-Type: application/json" \
    -d '{
      "session": {
        "group_id": "test_group",
        "agent_id": ["test_agent"],
        "user_id": ["test_user"],
        "session_id": "session_123"
      },
      "producer": "test_user",
      "produced_for": "test_agent",
      "episode_content": "This is a simple test memory.",
      "episode_type": "message",
      "metadata": {}
    }'
    """

    def post_episodic_memory(self, message, session_id=None):
        episodic_memory_endpoint = self._get_url(episodic_memory_path)
        if session_id is not None:
            self.session["session_id"] = session_id
        payload = {
            "session": self.session,
            "producer": self.producer,
            "produced_for": self.produced_for,
            "episode_content": message,
            "episode_type": "message",
            "metadata": {},
        }

        start_time = time.time()
        response = requests.post(episodic_memory_endpoint, json=payload, timeout=300)
        end_time = time.time()

        latency_ms = round((end_time - start_time) * 1000, 2)
        # Trace the request
        if self.verbose:
            self._trace_request(
                "POST", episodic_memory_endpoint, payload, response, latency_ms
            )
        else:
            self.statistic_fp.write(
                f"{datetime.now().isoformat()},POST,{episodic_memory_endpoint},{latency_ms}\n"
            )

        if response.status_code != 200:
            raise Exception(f"Failed to post episodic memory: {response.text}")
        return response.json()

    """
    curl -X POST "http://localhost:8080/v1/memories/episodic/search" \
    -H "Content-Type: application/json" \
    -d '{
      "session": {
        "group_id": "test_group",
        "agent_id": ["test_agent"],
        "user_id": ["test_user"],
        "session_id": "session_123"
      },
      "query": "simple test memory",
      "filter": {},
      "limit": 5
    }'
    """

    def search_episodic_memory(self, query_str, limit=5):
        search_episodic_memory_endpoint = self._get_url(
            f"{episodic_memory_path}/search"
        )
        query = {
            "session": self.session,
            "query": query_str,
            "filter": {},
            "limit": limit,
        }

        start_time = time.time()
        response = requests.post(
            search_episodic_memory_endpoint, json=query, timeout=300
        )
        end_time = time.time()
        latency_ms = round((end_time - start_time) * 1000, 2)

        if self.verbose:
            self._trace_request(
                "POST", search_episodic_memory_endpoint, query, response, latency_ms
            )
        else:
            self.statistic_fp.write(
                f"{datetime.now().isoformat()},POST,{search_episodic_memory_endpoint},{latency_ms}\n"
            )

        if response.status_code != 200:
            raise Exception(f"Failed to search episodic memory: {response.text}")
        return response.json()


if __name__ == "__main__":
    client = MemMachineRestClient(base_url="http://localhost:8080")
    client.post_episodic_memory(
        "I will start to write a new story today. There are 1 main characters in my story, lilith. she transmigrates into a game, After experiencing a series of bad endings, she breaks free in her final reincarnation, joining forces with her female companions to rebel and overthrow the corrupt dynasty."
    )
    results = client.search_episodic_memory("main character of my story")
    if results["status"] != 0:
        raise Exception(f"Failed to search episodic memory: {results}")
    if results["content"] is None:
        print("No results found")
        exit(1)
    if "episodic_memory" not in results["content"]:
        print("No episodic memory found")
    else:
        episodic_memory = results["content"]["episodic_memory"]
        if episodic_memory is not None:
            for memories in episodic_memory:
                if len(memories) == 0:
                    print("--- warn: empty memories found")
                    continue
                for memory in memories:
                    if isinstance(memory, dict) and "content" in memory:
                        print(memory["content"])
                    elif isinstance(memory, str) and memory.strip():
                        print(memory)
                    else:
                        # Skip empty strings or invalid data
                        print(f"--- warn: invalid memory data found: {memory}")
        else:
            print("Episodic memory is empty")
