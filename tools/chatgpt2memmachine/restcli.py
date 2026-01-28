import json
import os
import time
from datetime import datetime

import requests

from utils import get_filename_safe_timestamp


class MemMachineRestClient:
    def __init__(
        self, base_url="http://localhost:8080", api_version="v2", verbose=False
    ):
        self.base_url = base_url
        self.api_version = api_version
        self.verbose = verbose
        if self.verbose:
            # Use a filename-safe timestamp (Windows paths cannot contain colons)
            timestamp = get_filename_safe_timestamp()
            self.statistic_file = f"output/statistic_{timestamp}.csv"
            os.makedirs(os.path.dirname(self.statistic_file), exist_ok=True)
            with open(self.statistic_file, "w") as f:
                f.write("timestamp,method,url,latency_ms\n")
            self.statistic_fp = open(self.statistic_file, "a")
        else:
            self.statistic_fp = None

    def __del__(self):
        if hasattr(self, "statistic_fp") and self.statistic_fp is not None:
            self.statistic_fp.close()

    def _get_url(self, path):
        return f"{self.base_url}/api/{self.api_version}/{path}"

    def _trace_request(self, method, url, payload=None, response=None, latency_ms=None):
        """Trace API request details for debugging and reproduction"""
        print(f"\n🔍 API TRACE")
        print(f"   {method} {url}")
        if payload:
            print(f"   Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        if response:
            print(f"   Status: {response.status_code}")
            if response.status_code != 200:
                print(f"   Error: {response.text[:200]}")
        if latency_ms:
            print(f"   Latency: {latency_ms}ms")

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

    def add_memory(self, org_id="", project_id="", messages=None):
        add_memory_endpoint = self._get_url("memories")
        payload = {
            "messages": messages,
        }
        if org_id:
            payload["org_id"] = org_id
        if project_id:
            payload["project_id"] = project_id

        start_time = time.time()
        response = requests.post(add_memory_endpoint, json=payload, timeout=300)
        end_time = time.time()

        latency_ms = round((end_time - start_time) * 1000, 2)

        # Write to statistic file
        self.statistic_fp.write(
            f"{datetime.now().isoformat()},POST,{add_memory_endpoint},{latency_ms}\n",
        )

        # Trace the request if verbose
        if self.verbose:
            self._trace_request(
                "POST",
                add_memory_endpoint,
                payload,
                response,
                latency_ms,
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

        # Trace the request if verbose
        if self.verbose:
            self._trace_request(
                "POST",
                search_memory_endpoint,
                query,
                response,
                latency_ms,
            )
            # Write to statistic file
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
                print(
                    "Number of episodes in long term memory: ",
                    len(episodes_in_long_term_memory),
                )
                for episode in long_term_memory.get("episodes", []):
                    print(f"Episode: {episode['content']}")
            if short_term_memory is not None:
                episodes_in_short_term_memory = short_term_memory.get("episodes", [])
                print(
                    "Number of episodes in short term memory: ",
                    len(episodes_in_short_term_memory),
                )
                for episode in episodes_in_short_term_memory:
                    print(f"Episode: {episode['content']}")
        else:
            print("Episodic memory is empty")
