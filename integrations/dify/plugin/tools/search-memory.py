from __future__ import annotations

from collections.abc import Generator
from typing import Any

import requests

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from tools._memmachine import DEFAULT_BASE_URL, DEFAULT_ORG_ID, DEFAULT_PROJECT_ID, MemMachineClient


class SearchMemoryTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        api_key = (self.runtime.credentials or {}).get("memmachine_api_key")
        base_url = (self.runtime.credentials or {}).get("memmachine_base_url")
        base_url_str = str(base_url).strip() if base_url is not None else ""
        client = MemMachineClient(
            api_key=str(api_key or ""),
            base_url=base_url_str or DEFAULT_BASE_URL,
        )

        query = str(tool_parameters.get("query") or "").strip()
        if not query:
            yield self.create_json_message({"error": "`query` is required"})
            return

        top_k = tool_parameters.get("top_k", 10)
        try:
            top_k_int = int(top_k)
        except Exception:
            top_k_int = 10

        body: dict[str, Any] = {
            "org_id": DEFAULT_ORG_ID,
            "project_id": DEFAULT_PROJECT_ID,
            "query": query,
            "top_k": top_k_int,
        }

        filter_value = tool_parameters.get("filter")
        if filter_value is not None and str(filter_value).strip() != "":
            body["filter"] = str(filter_value)

        types_value = tool_parameters.get("types")
        if isinstance(types_value, list) and types_value:
            # openapi enum is ['semantic', 'episodic']
            body["types"] = [str(t).lower() for t in types_value if str(t).strip()]

        try:
            result = client.post("/memories/search", body)
            yield self.create_json_message(result)
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            yield self.create_json_message(
                {
                    "error": str(e),
                    "status_code": getattr(resp, "status_code", None),
                    "response_text": getattr(resp, "text", None),
                }
            )
        except Exception as e:
            yield self.create_json_message({"error": str(e)})
