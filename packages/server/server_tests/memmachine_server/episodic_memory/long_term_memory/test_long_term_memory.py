"""Integration tests for the declarative-backed LongTermMemory.

These tests require a live Neo4j or NebulaGraph instance and are marked
`integration`; they're deselected from the default pytest run.
"""

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer

from memmachine_server.common.episode_store import Episode
from memmachine_server.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine_server.episodic_memory.long_term_memory import (
    DeclarativeBackendParams,
    LongTermMemory,
)
from server_tests.memmachine_server.conftest import (
    is_docker_available,
    requires_sentence_transformers,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder():
    from sentence_transformers import SentenceTransformer

    from memmachine_server.common.embedder.sentence_transformer_embedder import (
        SentenceTransformerEmbedder,
        SentenceTransformerEmbedderParams,
    )

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
    from sentence_transformers import CrossEncoder

    from memmachine_server.common.reranker.cross_encoder_reranker import (
        CrossEncoderReranker,
        CrossEncoderRerankerParams,
    )

    return CrossEncoderReranker(
        CrossEncoderRerankerParams(
            cross_encoder=CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L6-v2",
            ),
        ),
    )


@pytest.fixture(scope="module")
def neo4j_connection_info():
    if not is_docker_available():
        pytest.skip("Docker is not available")

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
def neo4j_vector_graph_store(neo4j_driver):
    """Neo4j vector graph store for testing."""
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            force_exact_similarity_search=True,
        ),
    )


# --- NebulaGraph Fixtures ---


@pytest.fixture(scope="module")
def nebula_connection_info(nebula_connection_info_factory):
    """NebulaGraph connection info for long-term memory tests."""
    return nebula_connection_info_factory(
        schema_name="/test_long_term_schema",
        graph_name="test_long_term_graph",
    )


@pytest_asyncio.fixture(scope="module")
async def nebula_client(nebula_client_factory, nebula_connection_info):
    """Create NebulaGraph client for long-term memory tests."""
    return await nebula_client_factory(nebula_connection_info)


@pytest.fixture(scope="module")
def nebula_vector_graph_store(nebula_client, nebula_connection_info):
    """NebulaGraph vector graph store for testing."""
    from server_tests.memmachine_server.episodic_memory.conftest import (
        create_nebula_vector_graph_store,
    )

    return create_nebula_vector_graph_store(nebula_client, nebula_connection_info)


# --- Parameterized Fixture for Both Backends ---


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("neo4j_vector_graph_store", id="neo4j"),
        pytest.param("nebula_vector_graph_store", id="nebula"),
    ],
)
def vector_graph_store(request):
    """Parameterized fixture that tests both Neo4j and NebulaGraph."""
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="module")
def long_term_memory(embedder, reranker, vector_graph_store):
    return LongTermMemory(
        DeclarativeBackendParams(
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
    await long_term_memory.drop_session_partition()
    yield


@requires_sentence_transformers
@pytest.mark.asyncio
async def test_add_episodes(long_term_memory):
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

    scored = await long_term_memory.search_scored(
        "first president of the United States",
        num_episodes_limit=10,
    )
    returned_uids = {episode.uid for _, episode in scored}
    assert "episode2" in returned_uids or "episode3" in returned_uids


@requires_sentence_transformers
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
        ["episode1", "episode3", "nonexistent_episode"],
    )

    scored = await long_term_memory.search_scored(
        "first president of the United States",
        num_episodes_limit=10,
    )
    returned_uids = {episode.uid for _, episode in scored}
    assert "episode1" not in returned_uids
    assert "episode3" not in returned_uids


@requires_sentence_transformers
@pytest.mark.asyncio
async def test_drop_session_partition(long_term_memory):
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
            filterable_metadata={"project": "science"},
        ),
        Episode(
            uid="episode2",
            content="Who was the first president of the United States?",
            session_key="session2",
            created_at=now,
            producer_id="Alice",
            producer_role="user",
            sequence_num=0,
            filterable_metadata={"project": "history"},
        ),
    ]

    await long_term_memory.add_episodes(episodes)
    await long_term_memory.drop_session_partition()

    scored = await long_term_memory.search_scored(
        "first president",
        num_episodes_limit=10,
    )
    assert scored == []
