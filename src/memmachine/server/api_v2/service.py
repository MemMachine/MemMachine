"""API v2 service implementations."""

import asyncio
from dataclasses import dataclass
from typing import cast

from fastapi import Request
from pydantic import JsonValue

from memmachine import MemMachine
from memmachine.common.api import MemoryType as MemoryTypeE
from memmachine.common.api.spec import (
    AddMemoriesSpec,
    AddMemoryResult,
    DeleteMemoriesSpec,
    Episode,
    EpisodicSearchResult,
    ListMemoriesSpec,
    ListResult,
    ListResultContent,
    SearchMemoriesSpec,
    SearchResult,
    SearchResultContent,
    SemanticFeature,
)
from memmachine.common.episode_store.episode_model import EpisodeEntry


# Placeholder dependency injection function
async def get_memmachine(request: Request) -> MemMachine:
    """Get session data manager instance."""
    return request.app.state.mem_machine


@dataclass
class _SessionData:
    org_id: str
    project_id: str

    @property
    def session_key(self) -> str:
        return f"{self.org_id}/{self.project_id}"

    @property
    def user_profile_id(self) -> str | None:  # pragma: no cover - simple proxy
        return None

    @property
    def role_profile_id(self) -> str | None:  # pragma: no cover - simple proxy
        return None

    @property
    def session_id(self) -> str | None:  # pragma: no cover - simple proxy
        return self.session_key


async def _add_messages_to(
    target_memories: list[MemoryTypeE],
    spec: AddMemoriesSpec,
    memmachine: MemMachine,
) -> list[AddMemoryResult]:
    episodes: list[EpisodeEntry] = [
        EpisodeEntry(
            content=message.content,
            producer_id=message.producer,
            produced_for_id=message.produced_for,
            producer_role=message.role,
            created_at=message.timestamp,
            metadata=cast(dict[str, JsonValue], message.metadata),
            episode_type=message.episode_type,
        )
        for message in spec.messages
    ]

    episode_ids = await memmachine.add_episodes(
        session_data=_SessionData(
            org_id=spec.org_id,
            project_id=spec.project_id,
        ),
        episode_entries=episodes,
        target_memories=target_memories,
    )
    return [AddMemoryResult(uid=e_id) for e_id in episode_ids]


async def _delete_memories(
    spec: DeleteMemoriesSpec,
    memmachine: MemMachine,
) -> None:
    delete_episodes = memmachine.delete_episodes(
        session_data=_SessionData(
            org_id=spec.org_id,
            project_id=spec.project_id,
        ),
        episode_ids=spec.episodic_memory_uids,
    )
    delete_semantics = memmachine.delete_features(
        feature_ids=spec.semantic_memory_uids,
    )
    await asyncio.gather(delete_episodes, delete_semantics)


async def _search_target_memories(
    target_memories: list[MemoryTypeE],
    spec: SearchMemoriesSpec,
    memmachine: MemMachine,
) -> SearchResult:
    results = await memmachine.query_search(
        session_data=_SessionData(
            org_id=spec.org_id,
            project_id=spec.project_id,
        ),
        query=spec.query,
        target_memories=target_memories,
        search_filter=spec.filter,
        limit=spec.top_k,
        expand_context=spec.expand_context,
        score_threshold=spec.score_threshold
        if spec.score_threshold is not None
        else -float("inf"),
    )
    content = SearchResultContent(
        episodic_memory=None,
        semantic_memory=None,
    )
    if results.episodic_memory is not None:
        content.episodic_memory = EpisodicSearchResult(
            **results.episodic_memory.model_dump(mode="json")
        )
    if results.semantic_memory is not None:
        content.semantic_memory = [
            SemanticFeature(**f.model_dump(mode="json"))
            for f in results.semantic_memory
        ]
    return SearchResult(
        status=0,
        content=content,
    )


async def _list_target_memories(
    target_memories: list[MemoryTypeE],
    spec: ListMemoriesSpec,
    memmachine: MemMachine,
) -> ListResult:
    results = await memmachine.list_search(
        session_data=_SessionData(
            org_id=spec.org_id,
            project_id=spec.project_id,
        ),
        target_memories=target_memories,
        search_filter=spec.filter,
        page_size=spec.page_size,
        page_num=spec.page_num,
    )

    content = ListResultContent(
        episodic_memory=None,
        semantic_memory=None,
    )
    if results.episodic_memory is not None:
        content.episodic_memory = [
            Episode(**e.model_dump(mode="json")) for e in results.episodic_memory
        ]
    if results.semantic_memory is not None:
        content.semantic_memory = [
            SemanticFeature(**f.model_dump(mode="json"))
            for f in results.semantic_memory
        ]

    return ListResult(
        status=0,
        content=content,
    )
