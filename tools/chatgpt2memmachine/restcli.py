import json
import os
import time
from datetime import datetime

import requests


class MemMachineRestClient:
    def __init__(
        self,
        base_url="http://localhost:8080",
        api_version="v2",
        verbose=False,
        statistic_file=None,
    ):
        self.base_url = base_url
        self.api_version = api_version
        self.verbose = verbose
        self.statistic_file = statistic_file
        if self.statistic_file is None:
            # Use a filename-safe timestamp (Windows paths cannot contain colons)
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
            self.statistic_file = f"output/statistic_{timestamp}.csv"
        if not os.path.exists(self.statistic_file):
            os.makedirs(os.path.dirname(self.statistic_file), exist_ok=True)
        with open(self.statistic_file, "w") as f:
            f.write("timestamp,method,url,latency_ms\n")
        self.statistic_fp = open(self.statistic_file, "a")

    def __del__(self):
        if hasattr(self, "statistic_fp") and self.statistic_fp is not None:
            self.statistic_fp.close()

    def _get_url(self, path):
        return f"{self.base_url}/api/{self.api_version}/{path}"

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
    curl -X POST "http://localhost:8080/api/v2/memories" \
    -H "Content-Type: application/json" \
    -d '{
      "org_id": "my-org",
      "project_id": "my-project",
      "messages": [
        {
          "content": "This is a simple test memory.",
          "producer": "user-alice",
          "role": "user",
          "timestamp": "2025-11-24T10:00:00Z",
          "metadata": {
            "user_id": "user-alice",
          }
        }
      ],
      "types": ["episodic", "semantic"]
    }'
    """

    def add_memory(self, org_id, project_id, messages):
        add_memory_endpoint = self._get_url("memories")
        payload = {
            "org_id": org_id,
            "project_id": project_id,
            "messages": messages,
        }

        start_time = time.time()
        response = requests.post(add_memory_endpoint, json=payload, timeout=300)
        end_time = time.time()

        latency_ms = round((end_time - start_time) * 1000, 2)
        # Trace the request
        if self.verbose:
            self._trace_request(
                "POST",
                add_memory_endpoint,
                payload,
                response,
                latency_ms,
            )
        else:
            self.statistic_fp.write(
                f"{datetime.now().isoformat()},POST,{add_memory_endpoint},{latency_ms}\n",
            )

        if response.status_code != 200:
            raise Exception(f"Failed to post episodic memory: {response.text}")
        return response.json()

    """
    curl -X POST "http://localhost:8080/api/v2/memories/search" \
    -H "Content-Type: application/json" \
    -d '{
      "org_id": "my-org",
      "project_id": "my-project",
      "query": "simple test memory",
      "top_k": 5,
      "filter": "",
      "types": ["episodic", "semantic"]
    }'
    """

    def search_memory(self, org_id, project_id, query_str, limit=5):
        search_memory_endpoint = self._get_url("memories/search")
        query = {
            "org_id": org_id,
            "project_id": project_id,
            "query": query_str,
            "top_k": limit,
            "types": ["episodic", "semantic"],
        }

        start_time = time.time()
        response = requests.post(
            search_memory_endpoint,
            json=query,
            timeout=300,
        )
        end_time = time.time()
        latency_ms = round((end_time - start_time) * 1000, 2)

        if self.verbose:
            self._trace_request(
                "POST",
                search_memory_endpoint,
                query,
                response,
                latency_ms,
            )
        else:
            self.statistic_fp.write(
                f"{datetime.now().isoformat()},POST,{search_memory_endpoint},{latency_ms}\n",
            )

        if response.status_code != 200:
            raise Exception(f"Failed to search episodic memory: {response.text}")
        return response.json()


if __name__ == "__main__":
    print("Initializing client...")
    client = MemMachineRestClient(base_url="http://localhost:8080")
    print("Client initialized")
    print("Adding memory...")
    org_id = "my-org"
    project_id = "my-project"
    client.add_memory(
        org_id,
        project_id,
        [
            {
                "content": (
                    "Starting a new story about lilith, who transmigrates into a game."
                ),
            }
        ],
    )
    results = client.search_memory(org_id, project_id, "main character of my story")
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
            long_term_memory = episodic_memory.get("long_term_memory", {})
            short_term_memory = episodic_memory.get("short_term_memory", {})
            if long_term_memory is not None:
                episodes_in_long_term_memory = long_term_memory.get("episodes", [])
                print("Number of episodes in long term memory: ", len(episodes_in_long_term_memory))
                for episode in long_term_memory.get("episodes", []):
                    print(f"Episode: {episode['content']}")
            if short_term_memory is not None:
                episodes_in_short_term_memory = short_term_memory.get("episodes", [])
                print("Number of episodes in short term memory: ", len(episodes_in_short_term_memory))
                for episode in episodes_in_short_term_memory:
                    print(f"Episode: {episode['content']}")
        else:
            print("Episodic memory is empty")
