from __future__ import annotations

from collections.abc import Generator
from typing import Any

import requests

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from tools._memmachine import DEFAULT_BASE_URL, DEFAULT_ORG_ID, DEFAULT_PROJECT_ID, MemMachineClient


class AddMemoryTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        api_key = (self.runtime.credentials or {}).get("memmachine_api_key")
        base_url = (self.runtime.credentials or {}).get("memmachine_base_url")
        base_url_str = str(base_url).strip() if base_url is not None else ""
        client = MemMachineClient(
            api_key=str(api_key or ""),
            base_url=base_url_str or DEFAULT_BASE_URL,
        )

        content = str(tool_parameters.get("content") or "").strip()
        if not content:
            yield self.create_json_message({"error": "`content` is required"})
            return

        message: dict[str, Any] = {"content": content}
        for key in ("producer", "produced_for", "role", "timestamp"):
            value = tool_parameters.get(key)
            if value is not None and str(value).strip() != "":
                message[key] = value

        metadata = tool_parameters.get("metadata")
        if isinstance(metadata, dict) and metadata:
            # MemMachine expects string->string metadata per OpenAPI.
            message["metadata"] = {str(k): str(v) for k, v in metadata.items() if v is not None}

        body = {
            "org_id": DEFAULT_ORG_ID,
            "project_id": DEFAULT_PROJECT_ID,
            "messages": [message],
        }

        try:
            result = client.post("/memories", body)
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
