"""API v2 service implementations."""

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, cast

from babel import Locale, UnknownLocaleError
from fastapi import Query, Request
from memmachine_common.api import MemoryType as MemoryTypeE
from memmachine_common.api.spec import (
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
from pydantic import JsonValue

from memmachine_server import MemMachine
from memmachine_server.common.episode_store.episode_model import EpisodeEntry
from memmachine_server.common.request_context import (
    DEFAULT_LOCALE,
    reset_request_locale,
    set_request_locale,
)

logger = logging.getLogger(__name__)


# Placeholder dependency injection function
async def get_memmachine(request: Request) -> MemMachine:
    """Get session data manager instance."""
    return request.app.state.mem_machine


def _parse_locale(value: str | None) -> Locale:
    """Resolve a BCP-47 locale string to a ``Locale``.

    An absent or unrecognized value falls back to ``DEFAULT_LOCALE`` so a bad
    locale hint never fails the request it annotates.
    """
    if not value:
        return DEFAULT_LOCALE
    try:
        return Locale.parse(value, sep="-")
    except (UnknownLocaleError, ValueError):
        return DEFAULT_LOCALE


async def provide_request_locale(
    locale: Annotated[
        str | None,
        Query(
            description="Client locale as a BCP-47 tag (e.g. en-US) for "
            "locale-sensitive date parsing; defaults to en-US.",
        ),
    ] = None,
) -> AsyncIterator[None]:
    """Bind the request's locale (from the ``locale`` query param) for its lifetime."""
    token = set_request_locale(_parse_locale(locale))
    try:
        yield
    finally:
        reset_request_locale(token)


@dataclass(frozen=True)
class _SessionData:
    org_id: str
    project_id: str

    @property
    def session_key(self) -> str:
        return f"{self.org_id}/{self.project_id}"


def _session_key_to_session_data(session_key: str) -> _SessionData:
    """Convert session key to session data."""
    org_id, project_id = session_key.split("/", 1)
    return _SessionData(org_id=org_id, project_id=project_id)


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
    logger.debug("Service received search: query=%s", spec.query)
    results = await memmachine.query_search(
        session_data=_SessionData(
            org_id=spec.org_id,
            project_id=spec.project_id,
        ),
        query=spec.query,
        target_memories=target_memories,
        set_metadata=spec.set_metadata,
        search_filter=spec.filter,
        limit=spec.top_k,
        expand_context=spec.expand_context,
        score_threshold=spec.score_threshold
        if spec.score_threshold is not None
        else -float("inf"),
        agent_mode=spec.agent_mode,
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
        set_metadata=spec.set_metadata,
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
