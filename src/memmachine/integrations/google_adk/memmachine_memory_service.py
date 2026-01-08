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

from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
)
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session
from google.genai import types

from memmachine import MemMachineClient


class MemmachineError(RuntimeError):
    """Raised when MemMachine API calls fail or return unexpected data."""


def _format_timestamp_utc(timestamp: float) -> str:
    # MemMachine expects timezone-aware RFC3339 timestamps.
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat().replace("+00:00", "Z")


class MemmachineMemoryService(BaseMemoryService):
    """ADK memory service backed by MemMachine."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = "https://api.memmachine.ai",
        org_id: str = "example_org",
        project_id: str = "example_project",
        timeout_s: float = 30.0,
    ) -> None:
        """
        Create a MemMachine-backed ADK memory service.

        Args:
            api_key: API key for authentication (optional for local development)
            endpoint: MemMachine server URL (default: https://api.memmachine.ai)
            org_id: Organization identifier
            project_id: Project identifier
            timeout_s: Request timeout in seconds

        """
        self._org_id = org_id
        self._project_id = project_id
        self._timeout_s = timeout_s
        is_playground = endpoint == "https://api.memmachine.ai"

        # Initialize the MemMachine Python SDK client
        self._client = MemMachineClient(
            api_key=api_key,
            base_url=endpoint,
            timeout=int(timeout_s),
            is_playground=is_playground,
        )

        # Get or create the project
        try:
            self._project = self._client.get_or_create_project(
                org_id=org_id,
                project_id=project_id,
            )
        except Exception as e:
            raise MemmachineError("Failed to get or create MemMachine project.") from e

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

        # Create a memory instance for this session with combined metadata
        memory = self._project.memory(metadata={})

        # Add each message using the SDK
        for msg in messages:
            await asyncio.to_thread(
                memory.add,
                content=msg["content"],
                role=msg.get("metadata", {}).get("event_author", "user"),
                metadata=msg.get("metadata", {}),
            )

    async def search_memory(  # noqa: C901
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        # Create a memory instance with filter metadata
        memory = self._project.memory(metadata={})

        # Build filter dictionary for app_name and user_id
        filter_dict = {
            "metadata.app_name": app_name,
            "metadata.user_id": user_id,
        }

        # Search using the SDK
        search_result = await asyncio.to_thread(
            memory.search,
            query=query,
            limit=10,
            filter_dict=filter_dict,
        )

        # Extract content from SearchResult
        content = search_result.content if hasattr(search_result, "content") else None
        if not isinstance(content, dict):
            raise MemmachineError(
                "Unexpected MemMachine response format: missing object `content`. "
                f"Got: {json.dumps(search_result.model_dump() if hasattr(search_result, 'model_dump') else str(search_result))[:1000]}"
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
