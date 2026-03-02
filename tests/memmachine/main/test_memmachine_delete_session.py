"""Unit and integration tests for delete_session batched episode deletion."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.main.memmachine import EPISODE_DELETE_BATCH_SIZE


@pytest.mark.asyncio
async def test_delete_episode_store_processes_in_batches() -> None:
    """_delete_episode_store fetches and deletes episodes in batches until empty."""
    from dataclasses import dataclass

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
    resources.get_episode_storage = AsyncMock(return_value=episode_store)
    resources.get_session_data_manager = AsyncMock(
        return_value=MagicMock(get_session_info=AsyncMock(return_value=MagicMock()))
    )

    from memmachine.main.memmachine import MemMachine

    mm = MemMachine(conf=conf, resources=resources)
    mm._cleanup_semantic_history = cleanup_mock

    await mm.delete_session(_SD())

    # get_episode_ids called 4 times (3 non-empty + 1 empty sentinel)
    assert get_episode_ids_mock.call_count == 4
    for call in get_episode_ids_mock.call_args_list:
        assert call.kwargs["page_size"] == batch_size

    # delete_episodes called once per non-empty batch
    assert delete_episodes_mock.call_count == 3
    # cleanup called once per non-empty batch with the right IDs
    assert cleanup_mock.call_count == 3
    assert cleanup_mock.call_args_list[0].args[0] == all_ids[:batch_size]
