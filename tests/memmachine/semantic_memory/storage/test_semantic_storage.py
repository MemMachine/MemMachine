from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio

from memmachine.episode_store.episode_model import EpisodeIdT
from memmachine.semantic_memory.semantic_model import FeatureIdT, SemanticFeature
from memmachine.semantic_memory.storage.storage_base import SemanticStorage


@pytest.mark.asyncio
async def test_empty_storage(semantic_storage: SemanticStorage):
    assert await semantic_storage.get_feature_set(set_ids=["user"]) == []


@pytest.mark.asyncio
async def test_multiple_features(
    semantic_storage: SemanticStorage,
    with_multiple_features,
):
    # Given a storage with two features
    # When we retrieve the profile
    profile_result = await semantic_storage.get_feature_set(set_ids=["user"])
    grouped_profile = SemanticFeature.group_features(profile_result)

    assert len(grouped_profile) == 1

    key, expected_profile = with_multiple_features

    test_user_profile = grouped_profile[key]
    expected_test_user_profile = expected_profile[key]

    # Then the profile should contain both features
    assert len(test_user_profile) == 2
    for i in range(len(test_user_profile)):
        assert test_user_profile[i].value == expected_test_user_profile[i]["value"]


@pytest.mark.asyncio
async def test_delete_feature(semantic_storage: SemanticStorage):
    idx_a = await semantic_storage.add_feature(
        set_id="user",
        category_name="default",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )

    # Given a storage with a single feature
    features = await semantic_storage.get_feature_set(set_ids=["user"])
    assert len(features) == 1
    assert features[0].value == "pizza"

    # When we delete the feature
    await semantic_storage.delete_features([idx_a])

    features = await semantic_storage.get_feature_set(set_ids=["user"])

    # Then the feature should no longer exist
    assert features == []


@pytest.mark.asyncio
async def test_delete_feature_set_by_set_id(
    semantic_storage: SemanticStorage,
    with_multiple_sets,
):
    # Given a storage with two sets
    res_a = await semantic_storage.get_feature_set(set_ids=["user1"])
    grouped_a = SemanticFeature.group_features(res_a)

    res_b = await semantic_storage.get_feature_set(set_ids=["user2"])
    grouped_b = SemanticFeature.group_features(res_b)

    key, expected = with_multiple_sets

    set_a = [{"value": f.value} for f in grouped_a[key]]
    set_b = [{"value": f.value} for f in grouped_b[key]]

    assert set_a == expected["user1"]
    assert set_b == expected["user2"]

    # When we delete the first set
    await semantic_storage.delete_feature_set(set_ids=["user1"])

    # Then the first set should be empty
    res_delete_a = await semantic_storage.get_feature_set(set_ids=["user1"])
    assert res_delete_a == []

    # And the second set should still exist
    res_delete_b = await semantic_storage.get_feature_set(set_ids=["user2"])
    grouped_delete_b = SemanticFeature.group_features(res_delete_b)
    set_delete_b = [{"value": f.value} for f in grouped_delete_b[key]]
    assert set_delete_b == expected["user2"]


@pytest_asyncio.fixture
async def oposite_vector_features(semantic_storage: SemanticStorage):
    embed_a = np.array([1.0], dtype=float)
    value_a = "pizza"

    embed_b = np.array([0.0], dtype=float)
    value_b = "sushi"

    id_a = await semantic_storage.add_feature(
        set_id="user",
        category_name="default",
        tag="food",
        feature="likes",
        value=value_a,
        embedding=embed_a,
    )
    id_b = await semantic_storage.add_feature(
        set_id="user",
        category_name="default",
        tag="food",
        feature="likes",
        value=value_b,
        embedding=embed_b,
    )

    yield [
        (embed_a, value_a),
        (embed_b, value_b),
    ]

    await semantic_storage.delete_features([id_a, id_b])


@pytest.mark.asyncio
async def test_get_feature_set_basic_vector_search(
    semantic_storage: SemanticStorage,
    oposite_vector_features,
):
    if getattr(semantic_storage, "backend_name", None) == "neo4j":
        pytest.skip("Neo4j aproximate kNN limitation")

    # Given a storage with fully distinct features
    embed_a, value_a = oposite_vector_features[0]
    _embed_b, value_b = oposite_vector_features[1]

    # When doing a vector search
    results = await semantic_storage.get_feature_set(
        set_ids=["user"],
        limit=10,
        vector_search_opts=SemanticStorage.VectorSearchOpts(
            query_embedding=embed_a,
            min_distance=None,
        ),
    )

    # Then the results should be the two distinct features
    # With value_a being the first and value_b being the second
    result_values = [f.value for f in results]
    assert result_values == [value_a, value_b]


@pytest.mark.asyncio
async def test_get_feature_set_min_cos_vector_search(
    semantic_storage: SemanticStorage,
    oposite_vector_features,
):
    # Given a storage with fully distinct features
    embed_a, value_a = oposite_vector_features[0]

    # When doing a vector search with a min_cos threshold
    results = await semantic_storage.get_feature_set(
        set_ids=["user"],
        limit=10,
        vector_search_opts=SemanticStorage.VectorSearchOpts(
            query_embedding=embed_a,
            min_distance=0.5,
        ),
    )

    # Then the results should be the single closest distinct feature
    result_values = [f.value for f in results]
    assert result_values == [value_a]


@pytest.mark.asyncio
async def test_set_embedding_length_fixed_per_set(
    semantic_storage: SemanticStorage,
):
    if getattr(semantic_storage, "backend_name", None) != "neo4j":
        pytest.skip("Neo4j-specific validation")

    await semantic_storage.add_feature(
        set_id="user",
        category_name="default",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=np.ones(2, dtype=float),
    )

    with pytest.raises(ValueError, match="Embedding"):
        await semantic_storage.add_feature(
            set_id="user",
            category_name="default",
            feature="likes",
            value="sushi",
            tag="food",
            embedding=np.ones(3, dtype=float),
        )


@pytest.mark.asyncio
async def test_update_feature_respects_set_embedding_length(
    semantic_storage: SemanticStorage,
):
    if getattr(semantic_storage, "backend_name", None) != "neo4j":
        pytest.skip("Neo4j-specific validation")

    feature_id = await semantic_storage.add_feature(
        set_id="user1",
        category_name="default",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=np.ones(2, dtype=float),
    )
    await semantic_storage.add_feature(
        set_id="user2",
        category_name="default",
        feature="likes",
        value="sushi",
        tag="food",
        embedding=np.ones(3, dtype=float),
    )

    with pytest.raises(ValueError, match="Embedding"):
        await semantic_storage.update_feature(
            feature_id,
            set_id="user2",
        )

    with pytest.raises(ValueError, match="Embedding"):
        await semantic_storage.update_feature(
            feature_id,
            embedding=np.ones(3, dtype=float),
        )


@pytest_asyncio.fixture
async def feature_and_citations(
    semantic_storage: SemanticStorage,
    episode_storage,
):
    h1_id = await episode_storage.add_history(
        content="first",
        session_key="session_id",
        producer_id="profile_id",
        producer_role="dev",
    )
    await semantic_storage.add_history_to_set(
        set_id="user",
        history_id=h1_id,
    )
    h2_id = await episode_storage.add_history(
        content="second",
        session_key="session_id",
        producer_id="profile_id",
        producer_role="dev",
    )
    await semantic_storage.add_history_to_set(
        set_id="user",
        history_id=h2_id,
    )

    feature_id = await semantic_storage.add_feature(
        set_id="user",
        category_name="default",
        feature="topic",
        value="ai",
        tag="facts",
        embedding=np.array([1.0, 0.0]),
    )

    yield feature_id, {h1_id, h2_id}

    await semantic_storage.delete_features([feature_id])
    await episode_storage.delete_history([h1_id, h2_id])


@pytest.mark.asyncio
async def test_add_feature_with_citations(
    semantic_storage: SemanticStorage,
    feature_and_citations: tuple[FeatureIdT, set[EpisodeIdT]],
):
    feature_id, citations = feature_and_citations

    before_citations_features = await semantic_storage.get_feature(
        feature_id=feature_id,
        load_citations=True,
    )

    assert before_citations_features is not None
    assert before_citations_features.metadata.citations == []

    await semantic_storage.add_citations(feature_id, list(citations))

    after_citations_features = await semantic_storage.get_feature(
        feature_id=feature_id,
        load_citations=True,
    )
    assert after_citations_features.metadata.citations is not None
    assert all(
        c_id in citations for c_id in after_citations_features.metadata.citations
    )


@pytest.mark.asyncio
async def test_get_feature_without_citations(
    semantic_storage: SemanticStorage,
    feature_and_citations,
):
    feature_id, citations = feature_and_citations
    await semantic_storage.add_citations(feature_id, list(citations))

    without_citations = await semantic_storage.get_feature(
        feature_id=feature_id,
        load_citations=False,
    )
    assert without_citations.metadata.citations is None

    with_citations = await semantic_storage.get_feature(
        feature_id=feature_id,
        load_citations=True,
    )
    assert len(with_citations.metadata.citations) == len(citations)


@pytest.mark.asyncio
async def test_delete_feature_with_citations(
    semantic_storage: SemanticStorage,
    feature_and_citations,
):
    feature_id, citations = feature_and_citations
    await semantic_storage.add_citations(feature_id, list(citations))

    await semantic_storage.delete_features([feature_id])

    after_delete = await semantic_storage.get_feature(
        feature_id=feature_id,
        load_citations=True,
    )
    assert after_delete is None


@pytest.mark.asyncio
async def test_history_message_counts_by_set(
    semantic_storage: SemanticStorage,
    episode_storage,
):
    h1_id = await episode_storage.add_history(
        content="first",
        session_key="session_id",
        producer_id="profile_id",
        producer_role="dev",
    )
    h2_id = await episode_storage.add_history(
        content="second",
        session_key="session_id",
        producer_id="profile_id",
        producer_role="dev",
    )
    h3_id = await episode_storage.add_history(
        content="third",
        session_key="session_id",
        producer_id="profile_id",
        producer_role="dev",
    )

    await semantic_storage.add_history_to_set(set_id="only 1", history_id=h1_id)
    await semantic_storage.add_history_to_set(set_id="has 2", history_id=h1_id)
    await semantic_storage.add_history_to_set(set_id="has 2", history_id=h2_id)
    await semantic_storage.add_history_to_set(set_id="unused", history_id=h3_id)

    assert await semantic_storage.get_history_messages_count(set_ids=None) == 4
    assert await semantic_storage.get_history_messages_count(set_ids=["only 1"]) == 1
    assert await semantic_storage.get_history_messages_count(set_ids=["has 2"]) == 2
    assert await semantic_storage.get_history_messages_count(set_ids=["missing"]) == 0


@pytest.mark.asyncio
async def test_complex_feature_lifecycle(semantic_storage: SemanticStorage):
    embed = np.array([1.0] * 1536, dtype=float)

    await semantic_storage.add_feature(
        set_id="user",
        category_name="default",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=embed,
    )
    await semantic_storage.add_feature(
        set_id="user",
        category_name="default",
        feature="likes",
        value="sushi",
        tag="food",
        embedding=embed,
    )
    await semantic_storage.add_feature(
        set_id="user",
        category_name="tenant_A",
        feature="color",
        value="blue",
        tag="prefs",
        embedding=embed,
    )

    profile_default = await semantic_storage.get_feature_set(set_ids=["user"])
    grouped_default = SemanticFeature.group_features(profile_default)
    assert ("default", "food", "likes") in grouped_default

    likes_entries = grouped_default[("default", "food", "likes")]
    if not isinstance(likes_entries, list):
        likes_entries = [likes_entries]
    assert {item.value for item in likes_entries} == {"pizza", "sushi"}

    tenant_profile = await semantic_storage.get_feature_set(
        set_ids=["user"],
        category_names=["tenant_A"],
    )
    grouped_tenant = SemanticFeature.group_features(tenant_profile)
    assert grouped_tenant[("tenant_A", "prefs", "color")][0].value == "blue"

    await semantic_storage.delete_feature_set(
        set_ids=["user"],
        category_names=["default"],
        feature_names=["likes"],
        tags=["food"],
    )

    after_delete = await semantic_storage.get_feature_set(set_ids=["user"])
    grouped_after_delete = SemanticFeature.group_features(after_delete)
    assert ("default", "food", "likes") not in grouped_after_delete

    await semantic_storage.delete_feature_set(
        set_ids=["user"],
        category_names=["tenant_A"],
    )
    tenant_only = await semantic_storage.get_feature_set(
        set_ids=["user"],
        category_names=["tenant_A"],
    )
    assert tenant_only == []

    await semantic_storage.delete_feature_set(set_ids=["user"])
    assert await semantic_storage.get_feature_set(set_ids=["user"]) == []


@pytest.mark.asyncio
async def test_complex_semantic_search_and_citations(
    semantic_storage: SemanticStorage,
    episode_storage,
):
    history_id = await episode_storage.add_history(
        content="context note",
        metadata={"source": "chat"},
        session_key="session_key",
        producer_id="profile_id",
        producer_role="dev",
    )
    await semantic_storage.add_history_to_set(
        set_id="user",
        history_id=history_id,
    )

    f_id = await semantic_storage.add_feature(
        set_id="user",
        category_name="default",
        feature="topic",
        value="ai",
        tag="facts",
        embedding=np.array([1.0, 0.0]),
    )
    await semantic_storage.add_citations(f_id, [history_id])
    await semantic_storage.add_feature(
        set_id="user",
        category_name="default",
        feature="topic",
        value="music",
        tag="facts",
        embedding=np.array([0.0, 1.0]),
    )

    results = await semantic_storage.get_feature_set(
        set_ids=["user"],
        limit=10,
        vector_search_opts=SemanticStorage.VectorSearchOpts(
            query_embedding=np.array([1.0, 0.0]),
            min_distance=0.0,
        ),
        load_citations=True,
    )

    assert results is not None
    assert [entry.value for entry in results] == ["ai", "music"]

    assert results[0].metadata.citations is not None
    assert results[0].metadata.citations[0] == history_id

    filtered = await semantic_storage.get_feature_set(
        set_ids=["user"],
        limit=1,
        vector_search_opts=SemanticStorage.VectorSearchOpts(
            query_embedding=np.array([1.0, 0.0]),
            min_distance=0.5,
        ),
        # include_citations=False,
    )
    assert len(filtered) == 1
    assert filtered[0].value == "ai"

    history_id_set: set[int] = set()
    for entry in results:
        if entry.metadata.citations is not None:
            for citation in entry.metadata.citations:
                history_id_set.add(citation)

    assert history_id_set == {history_id}

    feature_ids = [
        entry.metadata.id for entry in results if entry.metadata.id is not None
    ]
    await semantic_storage.delete_features(feature_ids[:1])
    remaining = await semantic_storage.get_feature_set(
        set_ids=["user"],
        category_names=["default"],
        tags=["facts"],
        feature_names=["topic"],
    )
    assert len(remaining) == 1
    assert remaining[0].value == "music"


@pytest.mark.asyncio
async def test_history_ingestion_tracking(
    semantic_storage: SemanticStorage,
    episode_storage,
):
    history_ids = [
        await episode_storage.add_history(
            content=f"message-{idx}",
            session_key="session_id",
            producer_id="profile_id",
            producer_role="dev",
        )
        for idx in range(3)
    ]

    for h_id in history_ids:
        await semantic_storage.add_history_to_set(set_id="user", history_id=h_id)

    assert await semantic_storage.get_history_messages_count(
        set_ids=["user"],
        is_ingested=False,
    ) == len(history_ids)

    await semantic_storage.mark_messages_ingested(
        set_id="user",
        history_ids=history_ids[:2],
    )

    assert (
        await semantic_storage.get_history_messages_count(
            set_ids=["user"],
            is_ingested=False,
        )
        == 1
    )
    assert (
        await semantic_storage.get_history_messages_count(
            set_ids=["user"],
            is_ingested=True,
        )
        == 2
    )


@pytest.mark.asyncio
async def test_get_set_ids(
    semantic_storage: SemanticStorage,
):
    await semantic_storage.add_history_to_set(
        set_id="user_a",
        history_id="fake_a",
    )
    await semantic_storage.add_history_to_set(
        set_id="user_b",
        history_id="fake_b",
    )
    await semantic_storage.add_history_to_set(
        set_id="user_c",
        history_id="fake_c",
    )

    set_ids = await semantic_storage.get_history_set_ids()
    set_ids.sort()

    assert set_ids == ["user_a", "user_b", "user_c"]


@pytest.mark.asyncio
async def test_get_set_ids_with_min_uningested(
    semantic_storage: SemanticStorage,
):
    await semantic_storage.add_history_to_set(
        set_id="user_a",
        history_id="fake_a",
    )
    await semantic_storage.add_history_to_set(
        set_id="user_a",
        history_id="fake_b",
    )
    await semantic_storage.add_history_to_set(
        set_id="user_b",
        history_id="fake_c",
    )
    await semantic_storage.add_history_to_set(
        set_id="user_c",
        history_id="fake_d",
    )
    await semantic_storage.add_history_to_set(
        set_id="user_c",
        history_id="fake_a",
    )

    await semantic_storage.mark_messages_ingested(
        set_id="user_a",
        history_ids=["fake_a"],
    )
    await semantic_storage.mark_messages_ingested(
        set_id="user_b",
        history_ids=["fake_c"],
    )

    set_ids = await semantic_storage.get_history_set_ids(
        min_uningested_messages=0,
    )
    set_ids.sort()

    assert set_ids == ["user_a", "user_b", "user_c"]

    set_ids = await semantic_storage.get_history_set_ids(
        min_uningested_messages=1,
    )
    set_ids.sort()
    assert set_ids == ["user_a", "user_c"]

    set_ids = await semantic_storage.get_history_set_ids(
        min_uningested_messages=2,
    )
    assert set_ids == ["user_c"]
