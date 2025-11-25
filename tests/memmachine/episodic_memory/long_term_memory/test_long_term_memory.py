from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from sentence_transformers import CrossEncoder, SentenceTransformer
from testcontainers.neo4j import Neo4jContainer

from memmachine.common.embedder.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    SentenceTransformerEmbedderParams,
)
from memmachine.common.episode_store import Episode
from memmachine.common.reranker.cross_encoder_reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankerParams,
)
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine.episodic_memory.long_term_memory import (
    LongTermMemory,
    LongTermMemoryParams,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder():
    return SentenceTransformerEmbedder(
        SentenceTransformerEmbedderParams(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            sentence_transformer=SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
        ),
    )


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker(
        CrossEncoderRerankerParams(
            model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
            cross_encoder=CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L6-v2",
            ),
        ),
    )


@pytest.fixture(scope="module")
def neo4j_connection_info():
    neo4j_username = "neo4j"
    neo4j_password = "password"

    with Neo4jContainer(
        image="neo4j:latest",
        username=neo4j_username,
        password=neo4j_password,
    ) as neo4j:
        yield {
            "uri": neo4j.get_connection_url(),
            "username": neo4j_username,
            "password": neo4j_password,
        }


@pytest_asyncio.fixture(scope="module")
async def neo4j_driver(neo4j_connection_info):
    driver = AsyncGraphDatabase.driver(
        neo4j_connection_info["uri"],
        auth=(
            neo4j_connection_info["username"],
            neo4j_connection_info["password"],
        ),
    )
    yield driver
    await driver.close()


@pytest.fixture(scope="module")
def vector_graph_store(neo4j_driver):
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            force_exact_similarity_search=True,
        ),
    )


@pytest.fixture(scope="module")
def long_term_memory(embedder, reranker, vector_graph_store):
    return LongTermMemory(
        LongTermMemoryParams(
            session_id="test_session",
            embedder=embedder,
            reranker=reranker,
            vector_graph_store=vector_graph_store,
        ),
    )


@pytest.fixture(autouse=True)
def setup_nltk_data():
    import nltk

    nltk.download("punkt_tab")


@pytest_asyncio.fixture(autouse=True)
async def clear_long_term_memory(long_term_memory):
    await long_term_memory.delete_matching_episodes()
    all_episodes = await long_term_memory.get_matching_episodes()
    assert len(all_episodes) == 0
    yield


@pytest.mark.asyncio
async def test_add_episodes(long_term_memory):
    all_episodes = await long_term_memory.get_matching_episodes()
    assert len(all_episodes) == 0

    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="episode1",
            content="The mitochondria is the powerhouse of the cell.",
            session_key="session1",
            created_at=now,
            producer_id="biology textbook",
            producer_role="document",
            sequence_num=123,
            filterable_metadata={"project": "science", "length": "short"},
            metadata={"chapter": 5, "page": 42},
        ),
        Episode(
            uid="episode2",
            content="Who was the first president of the United States?",
            session_key="session2",
            created_at=now,
            producer_id="Alice",
            producer_role="user",
            sequence_num=0,
            filterable_metadata={"project": "history", "category": "question"},
        ),
        Episode(
            uid="episode3",
            content="George Washington was the first president of the United States.",
            session_key="session2",
            created_at=now + timedelta(seconds=10),
            producer_id="LLM",
            producer_role="assistant",
            produced_for_id="Alice",
            filterable_metadata={"project": "history", "length": "short"},
        ),
    ]

    await long_term_memory.add_episodes(episodes)

    all_episodes = await long_term_memory.get_matching_episodes()
    assert len(all_episodes) == len(episodes)
    assert set(all_episodes) == set(episodes)


@pytest.mark.asyncio
async def test_search(long_term_memory):
    pass


@pytest.mark.asyncio
async def test_get_episodes(long_term_memory):
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="episode1",
            content="The mitochondria is the powerhouse of the cell.",
            session_key="session1",
            created_at=now,
            producer_id="biology textbook",
            producer_role="document",
            sequence_num=123,
            filterable_metadata={"project": "science", "length": "short"},
            metadata={"chapter": 5, "page": 42},
        ),
        Episode(
            uid="episode2",
            content="Who was the first president of the United States?",
            session_key="session2",
            created_at=now,
            producer_id="Alice",
            producer_role="user",
            sequence_num=0,
            filterable_metadata={"project": "history", "category": "question"},
        ),
        Episode(
            uid="episode3",
            content="George Washington was the first president of the United States.",
            session_key="session2",
            created_at=now + timedelta(seconds=10),
            producer_id="LLM",
            producer_role="assistant",
            produced_for_id="Alice",
            filterable_properties={"project": "history", "length": "short"},
        ),
    ]

    await long_term_memory.add_episodes(episodes)

    results = await long_term_memory.get_episodes(["episode1", "episode3"])
    assert len(results) == 2
    assert set(results) == {episodes[0], episodes[2]}


@pytest.mark.asyncio
async def test_get_matching_episodes(long_term_memory):
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="episode1",
            content="The mitochondria is the powerhouse of the cell.",
            session_key="session1",
            created_at=now,
            producer_id="biology textbook",
            producer_role="document",
            sequence_num=123,
            filterable_metadata={"project": "science", "length": "short"},
            metadata={"chapter": 5, "page": 42},
        ),
        Episode(
            uid="episode2",
            content="Who was the first president of the United States?",
            session_key="session2",
            created_at=now,
            producer_id="Alice",
            producer_role="user",
            sequence_num=0,
            filterable_metadata={"project": "history", "category": "question"},
        ),
        Episode(
            uid="episode3",
            content="George Washington was the first president of the United States.",
            session_key="session2",
            created_at=now + timedelta(seconds=10),
            producer_id="LLM",
            producer_role="assistant",
            produced_for_id="Alice",
            filterable_metadata={"project": "history", "length": "short"},
        ),
    ]

    await long_term_memory.add_episodes(episodes)

    results = await long_term_memory.get_matching_episodes(
        property_filter={"metadata.length": "short"},
    )
    assert len(results) == 2
    assert set(results) == {episodes[0], episodes[2]}

    results = await long_term_memory.get_matching_episodes(
        property_filter={"m.length": None},
    )
    assert len(results) == 1
    assert set(results) == {episodes[1]}

    results = await long_term_memory.get_matching_episodes(
        property_filter={"m.project": "science", "metadata.length": "short"},
    )
    assert len(results) == 1
    assert set(results) == {episodes[0]}

    results = await long_term_memory.get_matching_episodes(
        property_filter={"m.project": None},
    )
    assert len(results) == 0

    results = await long_term_memory.get_matching_episodes(
        property_filter={"created_at": now},
    )
    assert len(results) == 2
    assert set(results) == {episodes[0], episodes[1]}


@pytest.mark.asyncio
async def test_delete_episodes(long_term_memory):
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="episode1",
            content="The mitochondria is the powerhouse of the cell.",
            session_key="session1",
            created_at=now,
            producer_id="biology textbook",
            producer_role="document",
            sequence_num=123,
            filterable_metadata={"project": "science", "length": "short"},
            metadata={"chapter": 5, "page": 42},
        ),
        Episode(
            uid="episode2",
            content="Who was the first president of the United States?",
            session_key="session2",
            created_at=now,
            producer_id="Alice",
            producer_role="user",
            sequence_num=0,
            filterable_metadata={"project": "history", "category": "question"},
        ),
        Episode(
            uid="episode3",
            content="George Washington was the first president of the United States.",
            session_key="session2",
            created_at=now + timedelta(seconds=10),
            producer_id="LLM",
            producer_role="assistant",
            produced_for_id="Alice",
            filterable_metadata={"project": "history", "length": "short"},
        ),
    ]

    await long_term_memory.add_episodes(episodes)

    await long_term_memory.delete_episodes(
        ["episode1", "episode3", "nonexistent_episode"]
    )
    all_episodes = await long_term_memory.get_matching_episodes()
    assert len(all_episodes) == 1
    assert set(all_episodes) == {episodes[1]}


@pytest.mark.asyncio
async def test_delete_matching_episodes(long_term_memory):
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="episode1",
            content="The mitochondria is the powerhouse of the cell.",
            session_key="session1",
            created_at=now,
            producer_id="biology textbook",
            producer_role="document",
            sequence_num=123,
            filterable_metadata={"project": "science", "length": "short"},
            metadata={"chapter": 5, "page": 42},
        ),
        Episode(
            uid="episode2",
            content="Who was the first president of the United States?",
            session_key="session2",
            created_at=now,
            producer_id="Alice",
            producer_role="user",
            sequence_num=0,
            filterable_metadata={"project": "history", "category": "question"},
        ),
        Episode(
            uid="episode3",
            content="George Washington was the first president of the United States.",
            session_key="session2",
            created_at=now + timedelta(seconds=10),
            producer_id="LLM",
            producer_role="assistant",
            produced_for_id="Alice",
            filterable_metadata={"project": "history", "length": "short"},
        ),
    ]

    await long_term_memory.add_episodes(episodes)

    await long_term_memory.delete_matching_episodes(
        property_filter={"m.length": None},
    )
    all_episodes = await long_term_memory.get_matching_episodes()
    assert len(all_episodes) == 2
    assert set(all_episodes) == {episodes[0], episodes[2]}

    await long_term_memory.delete_matching_episodes(
        property_filter={"metadata.length": "medium"},
    )
    all_episodes = await long_term_memory.get_matching_episodes()
    assert len(all_episodes) == 2
    assert set(all_episodes) == {episodes[0], episodes[2]}

    await long_term_memory.delete_matching_episodes(
        property_filter={"m.project": "history"},
    )
    all_episodes = await long_term_memory.get_matching_episodes()
    assert len(all_episodes) == 1
    assert set(all_episodes) == {episodes[0]}
