"""Unit and integration tests for delete_session batched episode deletion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import numpy as np
import pytest

from memmachine_server.common.episode_store import EpisodeEntry
from memmachine_server.main.memmachine import EPISODE_DELETE_BATCH_SIZE, MemMachine
from memmachine_server.semantic_memory.storage.storage_base import SemanticStorage

pytestmark = pytest.mark.integration


@dataclass
class _SessionData:
    session_key: str
    org_id: str
    project_id: str


async def _wait_for_history(
    semantic_storage: SemanticStorage,
    episode_id: str,
    *,
    timeout_seconds: float = 5.0,
) -> None:
    interval = 0.05
    attempts = max(int(timeout_seconds / interval), 1)
    for _ in range(attempts):
        history = [h async for h in semantic_storage.get_history_messages(set_ids=None)]
        if episode_id in history:
            return
        await asyncio.sleep(interval)
    pytest.fail("Episode history was not recorded in semantic storage")


@pytest.mark.asyncio
async def test_delete_session_clears_semantic_history_and_citations(
    memmachine: MemMachine,
) -> None:
    session_key = f"session-{uuid4()}"
    session_data = _SessionData(
        session_key=session_key,
        org_id=f"org-{session_key}",
        project_id=f"project-{session_key}",
    )

    semantic_manager = await memmachine._resources.get_semantic_manager()
    semantic_storage = await semantic_manager.get_semantic_storage()

    deleted = False
    try:
        await memmachine.create_session(session_key)

        episode_ids = await memmachine.add_episodes(
            session_data,
            [
                EpisodeEntry(
                    content="cleanup semantic references",
                    producer_id="tester",
                    producer_role="user",
                )
            ],
        )
        episode_id = episode_ids[0]

        await _wait_for_history(semantic_storage, episode_id)

        feature_id = await semantic_storage.add_feature(
            set_id="other-set",
            category_name="profile",
            feature="topic",
            value="pizza",
            tag="facts",
            embedding=np.array([1.0, 1.0]),
        )
        await semantic_storage.add_citations(feature_id, [episode_id])

        before_feature = await semantic_storage.get_feature(
            feature_id,
            load_citations=True,
        )
        assert before_feature is not None
        before_citations = before_feature.metadata.citations or []
        assert episode_id in before_citations

        await memmachine.delete_session(session_data)
        await memmachine._deletion_queue.join()
        deleted = True

        remaining_history = [
            h async for h in semantic_storage.get_history_messages(set_ids=None)
        ]
        assert episode_id not in remaining_history

        after_feature = await semantic_storage.get_feature(
            feature_id,
            load_citations=True,
        )
        assert after_feature is not None
        after_citations = after_feature.metadata.citations or []
        assert episode_id not in after_citations
    finally:
        if not deleted:
            remaining = await memmachine.get_session(session_key)
            if remaining is not None:
                await memmachine.delete_session(session_data)


@pytest.mark.asyncio
async def test_delete_episode_store_processes_in_batches() -> None:
    """_delete_episode_store fetches and deletes episodes in batches until empty."""

    @dataclass
    class _SD:
        session_key: str = "test-session"
        org_id: str = "org-1"
        project_id: str = "proj-1"

    batch_size = EPISODE_DELETE_BATCH_SIZE
    total = batch_size * 2 + 1

    # IDs are plain strings now — no Episode objects needed
    all_ids = [str(i) for i in range(total)]
    batches = [
        all_ids[:batch_size],
        all_ids[batch_size : batch_size * 2],
        all_ids[batch_size * 2 :],
        [],  # signals end of loop
    ]
    get_episode_ids_mock = AsyncMock(side_effect=batches)
    delete_episodes_mock = AsyncMock()
    cleanup_mock = AsyncMock()

    episode_store = MagicMock()
    episode_store.get_episode_ids = get_episode_ids_mock
    episode_store.delete_episodes = delete_episodes_mock

    conf = MagicMock()
    conf.episodic_memory.enabled = False
    conf.semantic_memory.enabled = False

    resources = MagicMock()
    resources.close = AsyncMock()
    session_manager = MagicMock()
    session_manager.get_session_info = AsyncMock(
        return_value=MagicMock(status="active")
    )
    session_manager.update_session_status = AsyncMock()
    session_manager.delete_session = AsyncMock()
    resources.get_episode_storage = AsyncMock(return_value=episode_store)
    resources.get_session_data_manager = AsyncMock(return_value=session_manager)

    mm = MemMachine(conf=conf, resources=resources)
    mm._cleanup_semantic_history = cleanup_mock  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

    await mm.start()
    await mm.delete_session(_SD())
    await mm._deletion_queue.join()
    await mm.stop()

    # get_episode_ids called 4 times (3 non-empty + 1 empty sentinel)
    assert get_episode_ids_mock.call_count == 4
    for call in get_episode_ids_mock.call_args_list:
        assert call.kwargs["page_size"] == batch_size

    # delete_episodes called once per non-empty batch
    assert delete_episodes_mock.call_count == 3
    # cleanup called once per non-empty batch with the right IDs
    assert cleanup_mock.call_count == 3
    assert cleanup_mock.call_args_list[0].args[0] == all_ids[:batch_size]
