from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import pytest_asyncio

from memmachine.semantic_memory.semantic_model import SemanticFeature
from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase


@pytest.fixture(
    params=[
        pytest.param("postgres", marks=pytest.mark.integration),
        "inmemory",
    ]
)
def storage(request):
    match request.param:
        case "postgres":
            return request.getfixturevalue("sqlalchemy_profile_storage")
        case "inmemory":
            return request.getfixturevalue("in_memory_profile_storage")
        case _:
            raise ValueError(f"Unknown storage type: {request.param}")


@pytest.mark.asyncio
async def test_empty_storage(storage: SemanticStorageBase):
    assert await storage.get_feature_set(set_ids=["user"]) == []


@pytest.mark.asyncio
async def test_multiple_features(
    storage: SemanticStorageBase,
    with_multiple_features,
):
    # Given a storage with two features
    # When we retrieve the profile
    profile_result = await storage.get_feature_set(set_ids=["user"])
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
async def test_delete_feature(storage: SemanticStorageBase):
    idx_a = await storage.add_feature(
        set_id="user",
        type_name="default",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )

    # Given a storage with a single feature
    features = await storage.get_feature_set(set_ids=["user"])
    assert len(features) == 1
    assert features[0].value == "pizza"

    # When we delete the feature
    await storage.delete_features([idx_a])

    features = await storage.get_feature_set(set_ids=["user"])

    # Then the feature should no longer exist
    assert features == []


@pytest.mark.asyncio
async def test_delete_feature_set_by_set_id(
    storage: SemanticStorageBase,
    with_multiple_sets,
):
    # Given a storage with two sets
    res_a = await storage.get_feature_set(set_ids=["user1"])
    grouped_a = SemanticFeature.group_features(res_a)

    res_b = await storage.get_feature_set(set_ids=["user2"])
    grouped_b = SemanticFeature.group_features(res_b)

    key, expected = with_multiple_sets

    set_a = [{"value": f.value} for f in grouped_a[key]]
    set_b = [{"value": f.value} for f in grouped_b[key]]

    assert set_a == expected["user1"]
    assert set_b == expected["user2"]

    # When we delete the first set
    await storage.delete_feature_set(set_ids=["user1"])

    # Then the first set should be empty
    res_delete_a = await storage.get_feature_set(set_ids=["user1"])
    assert res_delete_a == []

    # And the second set should still exist
    res_delete_b = await storage.get_feature_set(set_ids=["user2"])
    grouped_delete_b = SemanticFeature.group_features(res_delete_b)
    set_delete_b = [{"value": f.value} for f in grouped_delete_b[key]]
    assert set_delete_b == expected["user2"]


@pytest_asyncio.fixture
async def oposite_vector_features(storage: SemanticStorageBase):
    embed_a = np.array([1.0], dtype=float)
    value_a = "pizza"

    embed_b = np.array([0.0], dtype=float)
    value_b = "sushi"

    id_a = await storage.add_feature(
        set_id="user",
        type_name="default",
        tag="food",
        feature="likes",
        value=value_a,
        embedding=embed_a,
    )
    id_b = await storage.add_feature(
        set_id="user",
        type_name="default",
        tag="food",
        feature="likes",
        value=value_b,
        embedding=embed_b,
    )

    yield [
        (embed_a, value_a),
        (embed_b, value_b),
    ]

    await storage.delete_features([id_a, id_b])


@pytest.mark.asyncio
async def test_get_feature_set_basic_vector_search(
    storage: SemanticStorageBase,
    oposite_vector_features,
):
    # Given a storage with fully distinct features
    embed_a, value_a = oposite_vector_features[0]
    embed_b, value_b = oposite_vector_features[1]

    # When doing a vector search
    results = await storage.get_feature_set(
        set_ids=["user"],
        k=10,
        vector_search_opts=SemanticStorageBase.VectorSearchOpts(
            query_embedding=embed_a,
            min_cos=None,
        ),
    )

    # Then the results should be the two distinct features
    # With value_a being the first and value_b being the second
    result_values = [f.value for f in results]
    assert result_values == [value_a, value_b]


@pytest.mark.asyncio
async def test_get_feature_set_min_cos_vector_search(
    storage: SemanticStorageBase,
    oposite_vector_features,
):
    # Given a storage with fully distinct features
    embed_a, value_a = oposite_vector_features[0]

    # When doing a vector search with a min_cos threshold
    results = await storage.get_feature_set(
        set_ids=["user"],
        k=10,
        vector_search_opts=SemanticStorageBase.VectorSearchOpts(
            query_embedding=embed_a,
            min_cos=0.5,
        ),
    )

    # Then the results should be the single closest distinct feature
    result_values = [f.value for f in results]
    assert result_values == [value_a]


@pytest_asyncio.fixture
async def time_history_message(storage: SemanticStorageBase):
    datetime_at = datetime.now(timezone.utc) - timedelta(days=1)
    datetime_before = datetime_at - timedelta(minutes=1)
    datetime_after = datetime_at + timedelta(minutes=1)

    h_id = await storage.add_history(
        content="first",
        created_at=datetime_at,
    )
    await storage.add_history_to_set(
        set_id="user",
        history_id=h_id,
    )

    yield datetime_at, datetime_before, datetime_after, h_id

    await storage.delete_history([h_id])


@pytest.mark.asyncio
async def test_get_history_message_before_start_time(
    storage: SemanticStorageBase,
    time_history_message,
):
    # Given a storage with a single feature created at time `a`
    datetime_at, datetime_before, datetime_after, h_id = time_history_message

    # When we retrieve the feature with a start time before the feature was created
    before_start_time_result = await storage.get_history_messages(
        start_time=datetime_before,
    )
    count = await storage.get_history_messages_count(start_time=datetime_before)

    # Then the feature should be returned
    assert count == 1
    assert before_start_time_result[0].metadata.id == h_id


@pytest.mark.asyncio
async def test_get_history_message_before_end_time(
    storage: SemanticStorageBase,
    time_history_message,
):
    # Given a storage with a single feature created at time `a`
    datetime_at, datetime_before, datetime_after, h_id = time_history_message

    # When we retrieve the feature with an end time before the feature was created
    before_end_time_result = await storage.get_history_messages(
        end_time=datetime_before,
    )
    count = await storage.get_history_messages_count(end_time=datetime_before)

    # Then the feature should not be returned
    assert count == 0
    assert before_end_time_result == []


@pytest.mark.asyncio
async def test_get_history_message_after_start_time(
    storage: SemanticStorageBase,
    time_history_message,
):
    # Given a storage with a single feature created at time `a`
    datetime_at, datetime_before, datetime_after, h_id = time_history_message

    # When we retrieve the feature with a start time after the feature was created
    after_start_time_result = await storage.get_history_messages(
        start_time=datetime_after,
    )
    count = await storage.get_history_messages_count(
        start_time=datetime_after,
    )

    # Then the feature should not be returned
    assert count == 0
    assert after_start_time_result == []


@pytest.mark.asyncio
async def test_get_history_message_after_end_time(
    storage: SemanticStorageBase,
    time_history_message,
):
    # Given a storage with a single feature created at time `a`
    datetime_at, datetime_before, datetime_after, h_id = time_history_message

    # When we retrieve the feature with an end time after the feature was created
    after_end_time_result = await storage.get_history_messages(
        end_time=datetime_after,
    )
    count = await storage.get_history_messages_count(
        end_time=datetime_after,
    )

    # Then the feature should be returned
    assert count == 1
    assert after_end_time_result[0].metadata.id == h_id


@pytest.mark.asyncio
async def test_get_history_message_between_start_and_end_time(
    storage: SemanticStorageBase,
    time_history_message,
):
    # Given a storage with a single feature created at time `a`
    datetime_at, datetime_before, datetime_after, h_id = time_history_message

    # When we retrieve the feature between before and after time
    range_result = await storage.get_history_messages(
        start_time=datetime_before,
        end_time=datetime_after,
    )
    count = await storage.get_history_messages_count(
        start_time=datetime_before,
        end_time=datetime_after,
    )

    # Then the feature should be returned
    assert count == 1
    assert range_result[0].metadata.id == h_id


@pytest.mark.asyncio
async def test_get_history_message_at_start_time(
    storage: SemanticStorageBase,
    time_history_message,
):
    # Given a storage with a single feature created at time `a`
    datetime_at, datetime_before, datetime_after, h_id = time_history_message

    # When we retrieve the feature at the start time
    at_result = await storage.get_history_messages(
        start_time=datetime_at,
    )
    count = await storage.get_history_messages_count(
        start_time=datetime_at,
    )

    # Then the feature should be returned
    assert count == 1
    assert at_result[0].metadata.id == h_id


@pytest.mark.asyncio
async def test_get_history_message_at_end_time(
    storage: SemanticStorageBase,
    time_history_message,
):
    # Given a storage with a single feature created at time `a`
    datetime_at, datetime_before, datetime_after, h_id = time_history_message

    # When we retrieve the feature at the end time
    at_result = await storage.get_history_messages(
        end_time=datetime_at,
    )
    count = await storage.get_history_messages_count(
        end_time=datetime_at,
    )

    # Then the feature should be returned
    assert count == 1
    assert at_result[0].metadata.id == h_id


@pytest.mark.asyncio
async def test_get_history_messaage_at_range_time(
    storage: SemanticStorageBase,
    time_history_message,
):
    # Given a storage with a single feature created at time `a`
    datetime_at, datetime_before, datetime_after, h_id = time_history_message

    # When we retrieve the feature between the at time
    at_range_result = await storage.get_history_messages(
        start_time=datetime_at,
        end_time=datetime_at,
    )
    count = await storage.get_history_messages_count(
        start_time=datetime_at,
        end_time=datetime_at,
    )

    # Then the feature should be returned
    assert count == 1
    assert at_range_result[0].metadata.id == h_id


@pytest_asyncio.fixture
async def feature_and_citations(storage: SemanticStorageBase):
    h1_id = await storage.add_history(
        content="first",
    )
    await storage.add_history_to_set(
        set_id="user",
        history_id=h1_id,
    )
    h2_id = await storage.add_history(
        content="second",
    )
    await storage.add_history_to_set(
        set_id="user",
        history_id=h2_id,
    )

    feature_id = await storage.add_feature(
        set_id="user",
        type_name="default",
        feature="topic",
        value="ai",
        tag="facts",
        embedding=np.array([1.0, 0.0]),
    )

    yield feature_id, {h1_id, h2_id}

    await storage.delete_features([feature_id])
    await storage.delete_history([h1_id, h2_id])


@pytest.mark.asyncio
async def test_add_feature_with_citations(
    storage: SemanticStorageBase, feature_and_citations
):
    feature_id, citations = feature_and_citations

    before_citations_features = await storage.get_feature(
        feature_id=feature_id,
        load_citations=True,
    )

    assert before_citations_features is not None
    assert before_citations_features.metadata.citations == []

    await storage.add_citations(feature_id, list(citations))

    after_citations_features = await storage.get_feature(
        feature_id=feature_id,
        load_citations=True,
    )
    assert after_citations_features.metadata.citations is not None
    assert all(
        c.metadata.id in citations for c in after_citations_features.metadata.citations
    )


@pytest.mark.asyncio
async def test_get_feature_without_citations(
    storage: SemanticStorageBase, feature_and_citations
):
    feature_id, citations = feature_and_citations
    await storage.add_citations(feature_id, list(citations))

    without_citations = await storage.get_feature(
        feature_id=feature_id, load_citations=False
    )
    assert without_citations.metadata.citations is None

    with_citations = await storage.get_feature(
        feature_id=feature_id, load_citations=True
    )
    assert len(with_citations.metadata.citations) == len(citations)


@pytest.mark.asyncio
async def test_delete_feature_with_citations(
    storage: SemanticStorageBase,
    feature_and_citations,
):
    feature_id, citations = feature_and_citations
    await storage.add_citations(feature_id, list(citations))

    await storage.delete_features([feature_id])

    after_delete = await storage.get_feature(feature_id=feature_id, load_citations=True)
    assert after_delete is None


@pytest.mark.asyncio
async def test_delete_history_with_citations(
    storage: SemanticStorageBase,
    feature_and_citations,
):
    feature_id, citations = feature_and_citations
    await storage.add_citations(feature_id, list(citations))

    h_id = list(citations)[0]

    await storage.delete_history([h_id])

    after_delete = await storage.get_history(h_id)
    assert after_delete is None

    feature = await storage.get_feature(feature_id=feature_id, load_citations=True)
    assert len(feature.metadata.citations) == len(citations) - 1


@pytest.mark.asyncio
async def test_get_history_messsage_count_set_id(
    storage: SemanticStorageBase,
):
    h1_id = await storage.add_history(
        content="first",
    )
    h2_id = await storage.add_history(
        content="second",
    )
    _ = await storage.add_history(
        content="third",
    )

    await storage.add_history_to_set(
        set_id="only 1",
        history_id=h1_id,
    )

    await storage.add_history_to_set(
        set_id="has 2",
        history_id=h1_id,
    )
    await storage.add_history_to_set(
        set_id="has 2",
        history_id=h2_id,
    )

    all_count = await storage.get_history_messages_count()
    all_history = await storage.get_history_messages()

    assert len(all_history) == 3
    assert all_count == 3

    only_1_count = await storage.get_history_messages_count(set_id="only 1")
    only_1_history = await storage.get_history_messages(set_id="only 1")

    assert len(only_1_history) == 1
    assert only_1_count == 1

    has_2_count = await storage.get_history_messages_count(set_id="has 2")
    has_2_history = await storage.get_history_messages(set_id="has 2")

    assert len(has_2_history) == 2
    assert has_2_count == 2

    no_count = await storage.get_history_messages_count(set_id="no")
    no_history = await storage.get_history_messages(set_id="no")

    assert len(no_history) == 0
    assert no_count == 0


@pytest.mark.asyncio
async def test_complex_feature_lifecycle(storage: SemanticStorageBase):
    embed = np.array([1.0] * 1536, dtype=float)

    await storage.add_feature(
        set_id="user",
        type_name="default",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=embed,
    )
    await storage.add_feature(
        set_id="user",
        type_name="default",
        feature="likes",
        value="sushi",
        tag="food",
        embedding=embed,
    )
    await storage.add_feature(
        set_id="user",
        type_name="tenant_A",
        feature="color",
        value="blue",
        tag="prefs",
        embedding=embed,
    )

    profile_default = await storage.get_feature_set(set_ids=["user"])
    grouped_default = SemanticFeature.group_features(profile_default)
    assert ("default", "food", "likes") in grouped_default

    likes_entries = grouped_default[("default", "food", "likes")]
    if not isinstance(likes_entries, list):
        likes_entries = [likes_entries]
    assert {item.value for item in likes_entries} == {"pizza", "sushi"}

    tenant_profile = await storage.get_feature_set(
        set_ids=["user"],
        type_names=["tenant_A"],
    )
    grouped_tenant = SemanticFeature.group_features(tenant_profile)
    assert grouped_tenant[("tenant_A", "prefs", "color")][0].value == "blue"

    await storage.delete_feature_set(
        set_ids=["user"],
        type_names=["default"],
        feature_names=["likes"],
        tags=["food"],
    )

    after_delete = await storage.get_feature_set(set_ids=["user"])
    grouped_after_delete = SemanticFeature.group_features(after_delete)
    assert ("default", "food", "likes") not in grouped_after_delete

    await storage.delete_feature_set(set_ids=["user"], type_names=["tenant_A"])
    tenant_only = await storage.get_feature_set(
        set_ids=["user"],
        type_names=["tenant_A"],
    )
    assert tenant_only == []

    await storage.delete_feature_set(set_ids=["user"])
    assert await storage.get_feature_set(set_ids=["user"]) == []


@pytest.mark.asyncio
async def test_complex_semantic_search_and_citations(storage: SemanticStorageBase):
    history_id = await storage.add_history(
        content="context note",
        metadata={"source": "chat"},
    )
    await storage.add_history_to_set(
        set_id="user",
        history_id=history_id,
    )

    f_id = await storage.add_feature(
        set_id="user",
        type_name="default",
        feature="topic",
        value="ai",
        tag="facts",
        embedding=np.array([1.0, 0.0]),
    )
    await storage.add_citations(f_id, [history_id])
    await storage.add_feature(
        set_id="user",
        type_name="default",
        feature="topic",
        value="music",
        tag="facts",
        embedding=np.array([0.0, 1.0]),
    )

    results = await storage.get_feature_set(
        set_ids=["user"],
        k=10,
        vector_search_opts=SemanticStorageBase.VectorSearchOpts(
            query_embedding=np.array([1.0, 0.0]),
            min_cos=0.0,
        ),
        load_citations=True,
    )

    assert results is not None
    assert [entry.value for entry in results] == ["ai", "music"]

    assert results[0].metadata.citations is not None
    assert results[0].metadata.citations[0].metadata.id == history_id

    filtered = await storage.get_feature_set(
        set_ids=["user"],
        k=1,
        vector_search_opts=SemanticStorageBase.VectorSearchOpts(
            query_embedding=np.array([1.0, 0.0]),
            min_cos=0.5,
        ),
        # include_citations=False,
    )
    assert len(filtered) == 1
    assert filtered[0].value == "ai"

    history_id_set: set[int] = set()
    for entry in results:
        if entry.metadata.citations is not None:
            for citation in entry.metadata.citations:
                history_id_set.add(citation.metadata.id)

    assert history_id_set == {history_id}

    feature_ids = [
        entry.metadata.id for entry in results if entry.metadata.id is not None
    ]
    await storage.delete_features(feature_ids[:1])
    remaining = await storage.get_feature_set(
        set_ids=["user"],
        type_names=["default"],
        tags=["facts"],
        feature_names=["topic"],
    )
    assert len(remaining) == 1
    assert remaining[0].value == "music"


@pytest.mark.asyncio
async def test_complex_history_workflow(storage: SemanticStorageBase):
    h1_id = await storage.add_history(
        content="first",
        metadata={},
    )
    await storage.add_history_to_set(
        set_id="user",
        history_id=h1_id,
    )
    h2_id = await storage.add_history(
        content="second",
        metadata={},
    )
    await storage.add_history_to_set(
        set_id="user",
        history_id=h2_id,
    )
    await asyncio.sleep(0.5)
    cutoff = datetime.now(timezone.utc)
    await asyncio.sleep(0.5)
    h3_id = await storage.add_history(
        content="third",
        metadata={},
    )
    await storage.add_history_to_set(
        set_id="user",
        history_id=h3_id,
    )

    all_messages = await storage.get_history_messages(set_id="user")
    message_contents = [m.content for m in all_messages]
    assert message_contents == ["first", "second", "third"]

    latest_uningested = await storage.get_history_messages(
        set_id="user",
        k=2,
        is_ingested=False,
    )
    assert [entry.content for entry in latest_uningested] == ["first", "second"]

    assert await storage.get_history_messages_count(is_ingested=False) == 3
    await storage.mark_messages_ingested(
        set_id="user",
        ids=[h1_id, h2_id],
    )
    assert await storage.get_history_messages_count(is_ingested=False) == 1
    ingested = await storage.get_history_messages(
        set_id="user",
        is_ingested=True,
    )
    assert {entry.metadata.id for entry in ingested} == {h1_id, h2_id}

    uningested = await storage.get_history_messages(
        set_id="user",
        is_ingested=False,
    )
    assert {entry.metadata.id for entry in uningested} == {h3_id}

    window = await storage.get_history_messages(
        set_id="user",
        end_time=cutoff,
    )
    assert [h.content for h in window] == ["first", "second"]

    await storage.delete_history_messages(
        end_time=cutoff,
    )
    remaining = await storage.get_history_messages(set_id="user")
    assert remaining[0].content == "third"
    assert len(remaining) == 1

    await asyncio.sleep(0.01)
    await storage.delete_history_messages()
    assert await storage.get_history_messages(set_id="user") == []
