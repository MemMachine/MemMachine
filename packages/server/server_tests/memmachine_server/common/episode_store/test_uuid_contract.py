"""Episode identifiers stay UUID values through model and store boundaries."""

from uuid import UUID

import pytest
from memmachine_common.api.spec import AddMemoryResult, DeleteMemoriesSpec
from sqlalchemy import Uuid

from memmachine_server.common.episode_store import EpisodeEntry, EpisodeStorage
from memmachine_server.common.episode_store.episode_sqlalchemy_store import Episode


def test_episode_entry_and_api_models_keep_uuid_values():
    entry = EpisodeEntry(content="hello", producer_id="user", producer_role="user")

    assert isinstance(entry.uid, UUID)
    assert entry.uid.version == 4
    assert AddMemoryResult(uid=entry.uid).uid == entry.uid
    assert isinstance(AddMemoryResult(uid=entry.uid).uid, UUID)
    deletion = DeleteMemoriesSpec(
        org_id="default", project_id="default", episodic_memory_uids=[entry.uid]
    )
    assert deletion.episodic_memory_uids == [entry.uid]
    assert deletion.model_dump(mode="json")["episodic_memory_uids"] == [str(entry.uid)]


def test_episode_primary_key_uses_uuid_column():
    assert isinstance(Episode.__table__.c.id.type, Uuid)


@pytest.mark.asyncio
async def test_episode_store_round_trips_uuid(episode_storage: EpisodeStorage):
    entry = EpisodeEntry(content="hello", producer_id="user", producer_role="user")
    stored = (await episode_storage.add_episodes("uuid-session", [entry]))[0]

    try:
        assert isinstance(stored.uid, UUID)
        assert stored.uid == entry.uid
        fetched = await episode_storage.get_episode(entry.uid)
        assert fetched is not None
        assert fetched.uid == entry.uid
        assert entry.uid in await episode_storage.get_episode_ids(page_size=1000)
    finally:
        await episode_storage.delete_episodes([entry.uid])
