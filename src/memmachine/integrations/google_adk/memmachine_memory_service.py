"""
MemMachine implementation of ADK BaseMemoryService.

This module provides `MemmachineMemoryService`, which ingests ADK sessions into
MemMachine and retrieves memories via semantic search.

User-configurable:
- endpoint: defaults to https://api.memmachine.ai/v2
- api_key: sent via Authorization header (Bearer token)

Per user request:
- All namespace/boundary fields (app_name/user_id/session_id/...) are stored in
    `metadata`.
- Memory messages only include: content, timestamp, metadata.
- org_id/project_id are fixed (default example_org/example_project).
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import requests
from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
)
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session
from google.genai import types


class MemmachineError(RuntimeError):
    """Raised when MemMachine API calls fail or return unexpected data."""


def _join_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def _format_timestamp_utc(timestamp: float) -> str:
    # MemMachine expects timezone-aware RFC3339 timestamps.
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat().replace("+00:00", "Z")


class MemmachineMemoryService(BaseMemoryService):
    """ADK memory service backed by MemMachine."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = "https://api.memmachine.ai/v2",
        org_id: str = "example_org",
        project_id: str = "example_project",
        timeout_s: float = 30.0,
        requests_session: requests.Session | None = None,
    ) -> None:
        """Create a MemMachine-backed ADK memory service."""
        if not api_key:
            raise ValueError("api_key is required")

        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._org_id = org_id
        self._project_id = project_id
        self._timeout_s = timeout_s
        self._session = requests_session or requests.Session()

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = _join_url(self._endpoint, path)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key.strip()}",
        }
        try:
            resp = self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._timeout_s,
            )
        except requests.RequestException as e:
            raise MemmachineError(f"MemMachine request failed: {e}") from e

        try:
            data = resp.json() if resp.text else {}
        except ValueError:
            data = {}

        if 200 <= resp.status_code < 300:
            return data if isinstance(data, dict) else {}

        detail = data.get("detail") if isinstance(data, dict) else None
        raise MemmachineError(
            f"HTTP {resp.status_code} for POST {url}"
            + (f" | detail={detail}" if detail is not None else "")
            + (f" | body={resp.text[:1000]}" if resp.text else "")
        )

    async def add_session_to_memory(self, session: Session) -> None:
        messages: list[dict[str, Any]] = []

        for event in session.events:
            if not event.content or not event.content.parts:
                continue
            text_parts: list[str] = []
            for part in event.content.parts:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
            if not text_parts:
                continue

            # Metadata values are expected to be strings in the MemMachine schema.
            metadata: dict[str, str] = {
                "app_name": str(session.app_name),
                "user_id": str(session.user_id),
                "session_id": str(session.id),
                "event_id": str(event.id),
                "event_author": str(event.author),
            }
            if getattr(event, "invocation_id", ""):
                metadata["invocation_id"] = str(event.invocation_id)
            if getattr(event, "branch", None):
                metadata["branch"] = str(event.branch)

            messages.append(
                {
                    "content": "\n".join(text_parts),
                    "timestamp": _format_timestamp_utc(event.timestamp),
                    "metadata": metadata,
                }
            )

        if not messages:
            return

        payload = {
            "org_id": self._org_id,
            "project_id": self._project_id,
            "messages": messages,
        }

        await asyncio.to_thread(self._post_json, "/memories", payload)

    async def search_memory(  # noqa: C901
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        # Use JSON quoting for filter values to avoid filter parser ambiguity.
        filter_str = (
            "metadata.app_name="
            + json.dumps(app_name)
            + " AND metadata.user_id="
            + json.dumps(user_id)
        )
        payload = {
            "org_id": self._org_id,
            "project_id": self._project_id,
            "query": query,
            "filter": filter_str,
            "top_k": 10,
            "types": ["episodic", "semantic"],
        }
        data = await asyncio.to_thread(self._post_json, "/memories/search", payload)
        content = data.get("content") if isinstance(data, dict) else None
        if not isinstance(content, dict):
            raise MemmachineError(
                "Unexpected MemMachine response format: missing object `content`. "
                f"Got: {json.dumps(data)[:1000]}"
            )

        memories: list[MemoryEntry] = []

        def _add_episode(episode: dict[str, Any], *, bucket: str) -> None:
            text = episode.get("content")
            if not isinstance(text, str) or not text:
                return
            created_at = episode.get("created_at")
            uid = episode.get("uid")
            producer_id = episode.get("producer_id")

            custom_metadata: dict[str, Any] = {
                "memmachine_memory_type": "episodic",
                "memmachine_bucket": bucket,
                "memmachine_uid": uid,
                "memmachine_episode_type": episode.get("episode_type"),
                "memmachine_producer_id": producer_id,
                "memmachine_producer_role": episode.get("producer_role"),
                "memmachine_produced_for_id": episode.get("produced_for_id"),
            }
            # Preserve any metadata MemMachine returns.
            if episode.get("metadata") is not None:
                custom_metadata["memmachine_episode_metadata"] = episode.get("metadata")

            memories.append(
                MemoryEntry(
                    id=str(uid) if uid is not None else None,
                    author=str(producer_id) if producer_id is not None else None,
                    timestamp=str(created_at) if created_at is not None else None,
                    content=types.Content(parts=[types.Part(text=text)], role="user"),
                    custom_metadata=custom_metadata,
                )
            )

        episodic = content.get("episodic_memory")
        if isinstance(episodic, dict):
            long_term = episodic.get("long_term_memory")
            if isinstance(long_term, dict):
                episodes = long_term.get("episodes")
                if isinstance(episodes, list):
                    for ep in episodes:
                        if isinstance(ep, dict):
                            _add_episode(ep, bucket="long_term_memory")

            short_term = episodic.get("short_term_memory")
            if isinstance(short_term, dict):
                episodes = short_term.get("episodes")
                if isinstance(episodes, list):
                    for ep in episodes:
                        if isinstance(ep, dict):
                            _add_episode(ep, bucket="short_term_memory")
                # NOTE: short_term_memory may include `episode_summary` (list[str]).
                # Not included as MemoryEntry by default to avoid changing semantics.

        semantic = content.get("semantic_memory")
        if isinstance(semantic, list):
            for item in semantic:
                if not isinstance(item, dict):
                    continue
                feature_name = item.get("feature_name")
                value = item.get("value")
                if not isinstance(feature_name, str) or not isinstance(value, str):
                    continue

                category = item.get("category")
                tag = item.get("tag")
                text = f"[{category}/{tag}] {feature_name} = {value}"

                meta_obj = item.get("metadata")
                semantic_id = None
                if isinstance(meta_obj, dict) and meta_obj.get("id") is not None:
                    semantic_id = str(meta_obj.get("id"))

                memories.append(
                    MemoryEntry(
                        id=semantic_id,
                        author="memmachine",
                        timestamp=None,
                        content=types.Content(
                            parts=[types.Part(text=text)], role="user"
                        ),
                        custom_metadata={
                            "memmachine_memory_type": "semantic",
                            "memmachine_set_id": item.get("set_id"),
                            "memmachine_category": category,
                            "memmachine_tag": tag,
                            "memmachine_feature_name": feature_name,
                            "memmachine_value": value,
                            "memmachine_metadata": meta_obj,
                        },
                    )
                )

        return SearchMemoryResponse(memories=memories)
