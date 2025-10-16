import numpy as np
import pytest
import pytest_asyncio

from tests.memmachine.profile_memory.storage.in_memory_profile_storage import (
    InMemoryProfileStorage,
)


@pytest_asyncio.fixture
async def storage():
    store = InMemoryProfileStorage()
    await store.startup()
    yield store
    await store.delete_all()
    await store.cleanup()


@pytest.mark.asyncio
async def test_add_get_and_delete_profile_entries(storage: InMemoryProfileStorage):
    embed = np.array([1.0, 0.0], dtype=float)

    await storage.add_profile_feature(
        user_id="user",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=embed,
    )
    await storage.add_profile_feature(
        user_id="user",
        feature="likes",
        value="sushi",
        tag="food",
        embedding=embed,
    )
    await storage.add_profile_feature(
        user_id="user",
        feature="color",
        value="blue",
        tag="prefs",
        embedding=embed,
        isolations={"tenant": "A"},
    )

    profile_default = await storage.get_profile("user", {})
    assert profile_default["food"]["likes"][0]["value"] == "pizza"
    assert {item["value"] for item in profile_default["food"]["likes"]} == {
        "pizza",
        "sushi",
    }

    tenant_profile = await storage.get_profile("user", {"tenant": "A"})
    assert tenant_profile == {
        "prefs": {"color": {"value": "blue"}},
    }

    await storage.delete_profile_feature(
        user_id="user",
        feature="likes",
        tag="food",
        value="pizza",
        isolations={},
    )
    after_delete = await storage.get_profile("user", {})
    assert after_delete["food"]["likes"]["value"] == "sushi"

    await storage.delete_profile("user", {"tenant": "A"})
    assert await storage.get_profile("user", {"tenant": "A"}) == {}

    await storage.delete_profile("user", {})
    assert await storage.get_profile("user", {}) == {}


@pytest.mark.asyncio
async def test_citation_helpers(storage: InMemoryProfileStorage):
    history = await storage.add_history(
        user_id="user",
        content="first interaction",
        metadata={"source": "chat"},
        isolations={"tenant": "A"},
    )

    await storage.add_profile_feature(
        user_id="user",
        feature="greeting",
        value="hello",
        tag="chat",
        embedding=np.array([0.1, 0.9]),
        isolations={"tenant": "A"},
        citations=[history["id"]],
    )

    citations = await storage.get_citation_list(
        user_id="user",
        feature="greeting",
        value="hello",
        tag="chat",
        isolations={"tenant": "A"},
    )
    assert citations == [history["id"]]

    sections = await storage.get_large_profile_sections(
        user_id="user",
        thresh=1,
        isolations={"tenant": "A"},
    )
    profile_id = sections[0][0]["metadata"]["id"]

    all_citations = await storage.get_all_citations_for_ids([profile_id])
    assert all_citations == [(history["id"], {"tenant": "A"})]

    await storage.delete_profile_feature_by_id(profile_id)
    assert await storage.get_profile("user", {"tenant": "A"}) == {}


@pytest.mark.asyncio
async def test_semantic_search_sorting_and_citations(storage: InMemoryProfileStorage):
    history = await storage.add_history(
        user_id="user",
        content="context note",
        metadata={},
        isolations={},
    )

    await storage.add_profile_feature(
        user_id="user",
        feature="topic",
        value="ai",
        tag="facts",
        embedding=np.array([1.0, 0.0]),
        citations=[history["id"]],
    )
    await storage.add_profile_feature(
        user_id="user",
        feature="topic",
        value="music",
        tag="facts",
        embedding=np.array([0.0, 1.0]),
    )

    results = await storage.semantic_search(
        user_id="user",
        qemb=np.array([1.0, 0.1]),
        k=10,
        min_cos=-1.0,
        isolations={},
        include_citations=True,
    )
    assert [entry["value"] for entry in results] == ["ai", "music"]
    assert (
        results[0]["metadata"]["similarity_score"]
        > results[1]["metadata"]["similarity_score"]
    )
    assert results[0]["metadata"]["citations"] == ["context note"]

    filtered = await storage.semantic_search(
        user_id="user",
        qemb=np.array([1.0, 0.0]),
        k=1,
        min_cos=0.5,
        isolations={},
        include_citations=False,
    )
    assert len(filtered) == 1
    assert filtered[0]["value"] == "ai"


@pytest.mark.asyncio
async def test_history_management(storage: InMemoryProfileStorage):
    h1 = await storage.add_history(
        user_id="user",
        content="first",
        metadata={},
        isolations={},
    )
    h2 = await storage.add_history(
        user_id="user",
        content="second",
        metadata={},
        isolations={},
    )
    h3 = await storage.add_history(
        user_id="user",
        content="third",
        metadata={},
        isolations={},
    )

    storage._history_by_id[h1["id"]].timestamp = 100.0
    storage._history_by_id[h2["id"]].timestamp = 200.0
    storage._history_by_id[h3["id"]].timestamp = 300.0

    last_two = await storage.get_ingested_history_messages("user", k=2, isolations={})
    assert [entry["id"] for entry in last_two] == [h3["id"], h2["id"]]

    window = await storage.get_history_message(
        user_id="user",
        start_time=150,
        end_time=250,
        isolations={},
    )
    assert window == ["second"]

    await storage.delete_history(
        user_id="user",
        start_time=0,
        end_time=250,
        isolations={},
    )
    remaining = await storage.get_ingested_history_messages("user", k=5, isolations={})
    assert [entry["content"] for entry in remaining] == ["third"]

    tenant_history = await storage.add_history(
        user_id="user",
        content="tenant specific",
        metadata={},
        isolations={"tenant": "A"},
    )
    storage._history_by_id[tenant_history["id"]].timestamp = 400.0
    tenant_messages = await storage.get_ingested_history_messages(
        "user", k=5, isolations={"tenant": "A"}
    )
    assert [entry["id"] for entry in tenant_messages] == [tenant_history["id"]]

    await storage.purge_history(
        user_id="user",
        start_time=450,
        isolations={"tenant": "A"},
    )
    assert (
        await storage.get_ingested_history_messages(
            "user", k=5, isolations={"tenant": "A"}
        )
        == []
    )

    history_all = await storage.get_history_message(
        user_id="user",
        start_time=0,
        end_time=1000,
        isolations={},
    )
    assert history_all == ["third"]
