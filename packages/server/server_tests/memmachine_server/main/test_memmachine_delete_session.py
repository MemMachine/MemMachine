"""Integration tests of deleting a session and purging its partition."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID, uuid4

import numpy as np
import pytest

from memmachine_server.common.episode_store import EpisodeEntry
from memmachine_server.main.memmachine import MemMachine
from memmachine_server.main.partitioned_stores import EpisodePartitionedStore
from memmachine_server.semantic_memory.semantic_session_manager import (
    SemanticSessionManager,
)
from memmachine_server.semantic_memory.storage.storage_base import SemanticStorage

pytestmark = pytest.mark.integration


@dataclass
class _SessionData:
    session_key: str
    org_id: str
    project_id: str


async def _wait_for_history(
    semantic_storage: SemanticStorage,
    episode_id: UUID,
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
@pytest.mark.parametrize("semantic_enabled", [True, False])
async def test_delete_session_clears_semantic_history_and_citations(
    memmachine: MemMachine,
    semantic_enabled: bool,
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

        project_set_id = SemanticSessionManager.project_deletion_set_ids(
            org_id=session_data.org_id, project_id=session_data.project_id
        )[-1]
        project_feature_id = await semantic_storage.add_feature(
            set_id=project_set_id,
            category_name="profile",
            feature="topic",
            value="pasta",
            tag="facts",
            embedding=np.array([1.0, 1.0]),
        )
        await semantic_storage.add_citations(project_feature_id, [episode_id])

        before_feature = await semantic_storage.get_feature(
            feature_id,
            load_citations=True,
        )
        assert before_feature is not None
        before_citations = before_feature.metadata.citations or []
        assert episode_id in before_citations

        partition_manager = await memmachine._partition_manager()
        registration = await partition_manager.resolve_partition(session_key)
        assert registration is not None

        memmachine._conf.semantic_memory.enabled = semantic_enabled
        await memmachine.delete_session(session_data)
        deleted = True

        # The session is unreachable and its project's features are deleted
        # when the deletion returns.
        assert await memmachine.get_session(session_key) is None
        assert await semantic_storage.get_feature(project_feature_id) is None

        # The partition's purge deletes its episodes with their history and
        # citations; the purge rounds start it once the retention has passed.
        episode_store = EpisodePartitionedStore(memmachine._resources)
        while await episode_store.purge_partition(registration.partition_uuid):
            pass

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
        memmachine._conf.semantic_memory.enabled = True
        if not deleted:
            remaining = await memmachine.get_session(session_key)
            if remaining is not None:
                await memmachine.delete_session(session_data)
