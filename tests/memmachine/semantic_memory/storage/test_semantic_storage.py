from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import pytest

from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase


@pytest.fixture(
    params=[pytest.param("postgres", marks=pytest.mark.integration), "inmemory"]
)
def storage(request):
    match request.param:
        case "postgres":
            return request.getfixturevalue("asyncpg_profile_storage")
        case "inmemory":
            return request.getfixturevalue("in_memory_profile_storage")
        case _:
            raise ValueError(f"Unknown storage type: {request.param}")


@pytest.mark.asyncio
async def test_empty_storage(storage: SemanticStorageBase):
    assert await storage.get_set_features(set_id="user") == {}


@pytest.mark.asyncio
async def test_multiple_features(
    storage: SemanticStorageBase,
    with_multiple_features,
):
    # Given a storage with two features
    # When we retrieve the profile
    profile = await storage.get_set_features(set_id="user")

    # Then the profile should contain both features
    assert profile == with_multiple_features


@pytest.mark.asyncio
async def test_delete_feature(storage: SemanticStorageBase):
    idx_a = await storage.add_feature(
        set_id="user",
        semantic_type_id="default",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )

    # Given a storage with a single feature
    features = await storage.get_set_features(set_id="user")
    assert features == {
        "food": {
            "likes": {
                "value": "pizza",
            }
        }
    }

    # When we delete the feature
    await storage.delete_features([idx_a])

    features = await storage.get_set_features(set_id="user")

    # Then the feature should no longer exist
    assert features == {}


@pytest.mark.asyncio
async def test_delete_feature_set(
    storage: SemanticStorageBase,
    with_multiple_sets,
):
    # Given a storage with two sets
    set_a = await storage.get_set_features(set_id="user1")
    set_b = await storage.get_set_features(set_id="user2")

    assert set_a == with_multiple_sets["user1"]
    assert set_b == with_multiple_sets["user2"]

    # When we delete the first set
    await storage.delete_feature_set(set_id="user1")

    # Then the first set should be empty
    set_a = await storage.get_set_features(set_id="user1")
    assert set_a == {}

    # And the second set should still exist
    set_b = await storage.get_set_features(set_id="user2")
    assert set_b == with_multiple_sets["user2"]


@pytest.mark.asyncio
async def test_feature_lifecycle(storage: SemanticStorageBase):
    embed = np.array([1.0] * 1536, dtype=float)

    await storage.add_feature(
        set_id="user",
        semantic_type_id="default",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=embed,
    )
    await storage.add_feature(
        set_id="user",
        semantic_type_id="default",
        feature="likes",
        value="sushi",
        tag="food",
        embedding=embed,
    )
    await storage.add_feature(
        set_id="user",
        semantic_type_id="tenant_A",
        feature="color",
        value="blue",
        tag="prefs",
        embedding=embed,
    )

    profile_default = await storage.get_set_features(set_id="user")
    assert "food" in profile_default
    likes_entries = profile_default["food"]["likes"]
    if not isinstance(likes_entries, list):
        likes_entries = [likes_entries]
    assert {item["value"] for item in likes_entries} == {"pizza", "sushi"}

    tenant_profile = await storage.get_set_features(
        set_id="user",
        semantic_type_id="tenant_A",
    )
    assert tenant_profile["prefs"]["color"]["value"] == "blue"

    await storage.delete_feature_with_filter(
        set_id="user",
        semantic_type_id="default",
        feature="likes",
        tag="food",
    )
    after_delete = await storage.get_set_features(set_id="user")
    assert "food" not in after_delete

    await storage.delete_feature_set(set_id="user", semantic_type_id="tenant_A")
    tenant_only = await storage.get_set_features(
        set_id="user",
        semantic_type_id="tenant_A",
    )
    assert tenant_only == {}

    await storage.delete_feature_set(set_id="user")
    assert await storage.get_set_features(set_id="user") == {}


@pytest.mark.asyncio
async def test_semantic_search_and_citations(storage: SemanticStorageBase):
    history = await storage.add_history(
        set_id="user",
        content="context note",
        metadata={"source": "chat"},
    )

    await storage.add_feature(
        set_id="user",
        semantic_type_id="default",
        feature="topic",
        value="ai",
        tag="facts",
        embedding=np.array([1.0, 0.0] + [0.0] * 1534),
        citations=[history["id"]],
    )
    await storage.add_feature(
        set_id="user",
        semantic_type_id="default",
        feature="topic",
        value="music",
        tag="facts",
        embedding=np.array([0.0, 1.0] + [0.0] * 1534),
    )

    results = await storage.semantic_search(
        set_id="user",
        qemb=np.array([1.0, 0.1] + [0.0] * 1534),
        k=10,
        min_cos=-1.0,
        include_citations=True,
    )
    assert [entry["value"] for entry in results] == ["ai", "music"]
    assert (
        results[0]["metadata"]["similarity_score"]
        > results[1]["metadata"]["similarity_score"]
    )
    assert results[0]["metadata"]["citations"] == ["context note"]

    filtered = await storage.semantic_search(
        set_id="user",
        qemb=np.array([1.0, 0.0] + [0.0] * 1534),
        k=1,
        min_cos=0.5,
        include_citations=False,
    )
    assert len(filtered) == 1
    assert filtered[0]["value"] == "ai"

    feature_ids = [entry["metadata"]["id"] for entry in results]
    citation_map = [cid for cid in await storage.get_all_citations_for_ids(feature_ids)]
    assert citation_map == [history["id"]]

    await storage.delete_features(feature_ids[:1])
    remaining = await storage.get_set_features(set_id="user")
    assert remaining["facts"]["topic"]["value"] == "music"


@pytest.mark.asyncio
async def test_history_workflow(storage: SemanticStorageBase):
    h1 = await storage.add_history(
        set_id="user",
        content="first",
        metadata={},
    )
    await asyncio.sleep(0.01)
    h2 = await storage.add_history(
        set_id="user",
        content="second",
        metadata={},
    )
    await asyncio.sleep(0.01)
    cutoff = datetime.now()
    await asyncio.sleep(0.01)
    h3 = await storage.add_history(
        set_id="user",
        content="third",
        metadata={},
    )

    all_messages = await storage.get_history_message(
        set_id="user", end_time=datetime.now()
    )
    assert all_messages == ["first", "second", "third"]

    latest_uningested = await storage.get_history_messages_by_ingestion_status(
        set_id="user",
        k=2,
        is_ingested=False,
    )
    assert [entry["content"] for entry in latest_uningested] == ["first", "second"]

    assert await storage.get_uningested_history_messages_count() == 3
    await storage.mark_messages_ingested(ids=[h1["id"], h2["id"]])
    assert await storage.get_uningested_history_messages_count() == 1

    ingested = await storage.get_history_messages_by_ingestion_status(
        set_id="user",
        is_ingested=True,
    )
    assert {entry["id"] for entry in ingested} == {h1["id"], h2["id"]}

    uningested = await storage.get_history_messages_by_ingestion_status(
        set_id="user",
        is_ingested=False,
    )
    assert {entry["id"] for entry in uningested} == {h3["id"]}

    window = await storage.get_history_message(
        set_id="user",
        end_time=cutoff,
    )
    assert window == ["first", "second"]

    await storage.delete_history(
        set_id="user",
        end_time=cutoff,
    )
    remaining = await storage.get_history_message(set_id="user")
    assert remaining == ["third"]

    await asyncio.sleep(0.01)
    await storage.delete_history(
        set_id="user",
    )
    assert await storage.get_history_message(set_id="user") == []
