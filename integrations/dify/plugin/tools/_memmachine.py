from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_BASE_URL = "https://api.memmachine.ai/v2"
DEFAULT_ORG_ID = "default_org"
DEFAULT_PROJECT_ID = "default_project"


@dataclass(frozen=True)
class MemMachineClient:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: tuple[float, float] = (10.0, 60.0)

    def _auth_header_value(self) -> str:
        value = (self.api_key or "").strip()
        if not value:
            raise ValueError("Missing memmachine api key")
        if value.lower().startswith("bearer "):
            return value
        return f"Bearer {value}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": self._auth_header_value(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def post(self, path: str, json_body: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        resp = requests.post(url, headers=self._headers(), json=json_body, timeout=self.timeout)

        # If API returns non-JSON on error, keep a safe fallback.
        try:
            payload: Any = resp.json() if resp.content else None
        except Exception:
            payload = None

        if not resp.ok:
            raise requests.HTTPError(
                f"MemMachine API error {resp.status_code}",
                response=resp,
            )

        if isinstance(payload, dict):
            return payload
        # normalize non-dict JSON payloads
        return {"data": payload}
