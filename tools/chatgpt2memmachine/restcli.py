import json
import os
import time
from datetime import datetime

import requests

from utils import get_filename_safe_timestamp


class MemMachineRestClient:
    def __init__(
        self,
        base_url="http://localhost:8080",
        api_version="v2",
        verbose=False,
        run_id=None,
    ):
        self.base_url = base_url
        self.api_version = api_version
        self.verbose = verbose
        if self.verbose:
            if not run_id:
                run_id = get_filename_safe_timestamp()
            self.api_requests_file = f"output/api_requests_{run_id}.csv"
            self.trace_file = f"output/trace_{run_id}.txt"
            os.makedirs(os.path.dirname(self.api_requests_file), exist_ok=True)
            with open(self.api_requests_file, "w") as f:
                f.write("timestamp,method,url,latency_ms,response_code\n")
            self.api_requests_fp = open(self.api_requests_file, "a")
            self.trace_fp = open(self.trace_file, "w")
        else:
            self.api_requests_fp = None
            self.trace_fp = None

    def __del__(self):
        if hasattr(self, "api_requests_fp") and self.api_requests_fp is not None:
            self.api_requests_fp.close()
        if hasattr(self, "trace_fp") and self.trace_fp is not None:
            self.trace_fp.close()

    def _get_url(self, path):
        return f"{self.base_url}/api/{self.api_version}/{path}"

    def _trace_request(self, method, url, payload=None, response=None, latency_ms=None):
        """Trace API request details for debugging and reproduction"""
        if not self.verbose or self.trace_fp is None:
            return

        trace_lines = []
        trace_lines.append(f"\n🔍 API TRACE")
        trace_lines.append(f"   {method} {url}")
        if payload:
            trace_lines.append(
                f"   Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}"
            )

        # Always try to write response information
        if response is not None:
            try:
                response_code = getattr(response, "status_code", None)
                response_text = None
                try:
                    response_text = getattr(response, "text", None) or ""
                except Exception:
                    response_text = "<unable to read>"

                trace_lines.append(f"   Response Code: {response_code}")
                trace_lines.append(
                    f"   Response Text: {response_text[:500] if response_text else '<empty>'}"
                )

                if response_code and response_code != 200:
                    error_text = (
                        response_text[:200]
                        if response_text and response_text != "<unable to read>"
                        else ""
                    )
                    trace_lines.append(f"   Error: {error_text}")
            except Exception as e:
                trace_lines.append(
                    f"   Response Code: <error reading response: {str(e)}>"
                )
        else:
            trace_lines.append(f"   Response Code: <no response object>")

        if latency_ms is not None:
            trace_lines.append(f"   Latency: {latency_ms}ms")

        # Write to trace file
        self.trace_fp.write("\n".join(trace_lines) + "\n")
        self.trace_fp.flush()  # Ensure immediate write

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

    def add_memory(self, org_id="", project_id="", messages=None) -> dict:
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

        # Trace the request if verbose
        if self.verbose:
            self._trace_request(
                "POST",
                add_memory_endpoint,
                payload,
                response,
                latency_ms,
            )
            # Write to API requests log file
            response_code = response.status_code if response is not None else ""
            self.api_requests_fp.write(
                f"{datetime.now().isoformat()},POST,{add_memory_endpoint},{latency_ms},{response_code}\n",
            )
            self.api_requests_fp.flush()  # Ensure immediate write

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
            # Write to API requests log file
            response_code = response.status_code if response is not None else ""
            self.api_requests_fp.write(
                f"{datetime.now().isoformat()},POST,{search_memory_endpoint},{latency_ms},{response_code}\n",
            )
            self.api_requests_fp.flush()  # Ensure immediate write

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
