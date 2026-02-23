from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer

from memmachine.common.episode_store import Episode
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine.episodic_memory.long_term_memory import (
    LongTermMemory,
    LongTermMemoryParams,
)
from tests.memmachine.conftest import (
    is_docker_available,
    requires_sentence_transformers,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder():
    from sentence_transformers import SentenceTransformer

    from memmachine.common.embedder.sentence_transformer_embedder import (
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

    from memmachine.common.reranker.cross_encoder_reranker import (
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
    from tests.memmachine.episodic_memory.conftest import (
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


@requires_sentence_transformers
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


@requires_sentence_transformers
@pytest.mark.asyncio
async def test_search(long_term_memory):
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid=str(uuid4()),
            session_key="search_session",
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            created_at=now - i * timedelta(seconds=1),
            producer_id="filler",
            producer_role="more_filler",
            filterable_metadata={"project": "testing", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    episodes += [
        Episode(
            uid=str(uuid4()),
            session_key="search_session",
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            created_at=now - i * timedelta(seconds=1),
            producer_id="filler",
            producer_role="more_filler",
            filterable_metadata={"project": "memmachine", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    episodes += [
        Episode(
            uid="episode1",
            session_key="search_session",
            content="This test is broken. Who wrote this test?",
            created_at=now,
            producer_id="Alice",
            producer_role="user",
            filterable_metadata={"project": "memmachine", "length": "short"},
            metadata={"some_key": "some_value"},
        ),
        Episode(
            uid="episode2",
            session_key="search_session",
            content="Charlie.",
            created_at=now + timedelta(seconds=10),
            producer_id="Bob",
            producer_role="user",
            filterable_metadata={"project": "other", "length": "short"},
            metadata={"some_other_key": "some_other_value"},
        ),
        Episode(
            uid="episode3",
            session_key="search_session",
            content="The mitochondria is the powerhouse of the cell.",
            created_at=now + timedelta(seconds=20),
            producer_id="textbook",
            producer_role="document",
        ),
        Episode(
            uid="episode4",
            session_key="search_session",
            content="",
            created_at=now + timedelta(seconds=30),
            producer_id="pet rock",
            producer_role="pet",
        ),
        Episode(
            uid="episode5",
            session_key="search_session",
            content="Edwin Yu: https://github.com/edwinyyyu\n",
            created_at=now + timedelta(seconds=40),
            producer_id="Charlie",
            producer_role="user",
            filterable_metadata={"project": "memmachine"},
        ),
        Episode(
            uid="episode6",
            session_key="search_session",
            content="I wrote this test.",
            created_at=now + timedelta(seconds=50),
            producer_id="Edwin",
            producer_role="user",
            filterable_metadata={"project": "memmachine", "length": "short"},
        ),
    ]
    episodes += [
        Episode(
            uid=str(uuid4()),
            session_key="search_session",
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            created_at=now + i * timedelta(seconds=1),
            producer_id="filler",
            producer_role="more_filler",
            filterable_metadata={"project": "testing", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    episodes += [
        Episode(
            uid=str(uuid4()),
            session_key="search_session",
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            created_at=now + i * timedelta(seconds=100),
            producer_id="filler",
            producer_role="more_filler",
            filterable_metadata={"project": "memmachine", "length": "medium"},
        )
        for i in range(1, 11)
    ]

    await long_term_memory.add_episodes(episodes)

    results = await long_term_memory.search(
        query="Who wrote the test?",
        num_episodes_limit=1,
    )

    assert len(results) == 1
    assert results[0].uid == "episode1" or results[0].uid == "episode6"

    results = await long_term_memory.search(
        query="Who wrote the test?",
        num_episodes_limit=4,
        score_threshold=-float("inf"),
    )

    assert len(results) == 4
    # Most relevant.
    assert "episode1" in [result.uid for result in results]
    assert "episode6" in [result.uid for result in results]

    results = await long_term_memory.search(
        query="Who wrote the test?",
        num_episodes_limit=4,
        expand_context=0,
        score_threshold=-float("inf"),
    )

    assert len(results) == 4
    # Most relevant.
    assert "episode1" in [result.uid for result in results]
    assert "episode6" in [result.uid for result in results]

    results = await long_term_memory.search(
        query="Who wrote the test?",
        num_episodes_limit=4,
        expand_context=3,
        score_threshold=-float("inf"),
    )

    assert len(results) == 4
    # Most relevant.
    assert "episode1" in [result.uid for result in results] or "episode6" in [
        result.uid for result in results
    ]
    # Relevant but first result consumes entire budget.
    assert "episode1" not in [result.uid for result in results] or "episode6" not in [
        result.uid for result in results
    ]

    results = await long_term_memory.search(
        query="Who wrote the test?",
        num_episodes_limit=4,
        score_threshold=float("inf"),
    )

    assert len(results) == 0

    results = await long_term_memory.search(
        query="Who wrote the test?",
        num_episodes_limit=10,
        property_filter=FilterComparison(
            field="m.project",
            op="=",
            value="memmachine",
        ),
    )
    assert len(results) == 10
    assert "episode1" in [result.uid for result in results]
    assert "episode5" in [result.uid for result in results]

    results = await long_term_memory.search(
        query="Who wrote the test?",
        num_episodes_limit=4,
        property_filter=FilterComparison(
            field="metadata.length",
            op="=",
            value="short",
        ),
    )

    assert len(results) == 3
    assert "episode1" in [result.uid for result in results]
    assert "episode2" in [result.uid for result in results]
    assert "episode6" in [result.uid for result in results]


@requires_sentence_transformers
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
            filterable_metadata={"project": "history", "length": "short"},
        ),
    ]

    await long_term_memory.add_episodes(episodes)

    results = await long_term_memory.get_episodes(["episode1", "episode3"])
    assert len(results) == 2
    assert set(results) == {episodes[0], episodes[2]}


@requires_sentence_transformers
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
        property_filter=FilterComparison(
            field="producer_id",
            op="=",
            value="Alice",
        )
    )
    assert len(results) == 1
    assert set(results) == {episodes[1]}

    results = await long_term_memory.get_matching_episodes(
        property_filter=FilterComparison(
            field="producer_role",
            op="=",
            value="assistant",
        )
    )
    assert len(results) == 1
    assert set(results) == {episodes[2]}

    results = await long_term_memory.get_matching_episodes(
        property_filter=FilterComparison(
            field="produced_for_id",
            op="=",
            value="Alice",
        )
    )
    assert len(results) == 1
    assert set(results) == {episodes[2]}

    results = await long_term_memory.get_matching_episodes(
        property_filter=FilterComparison(
            field="m.length",
            op="=",
            value="short",
        ),
    )
    assert len(results) == 2
    assert set(results) == {episodes[0], episodes[2]}

    results = await long_term_memory.get_matching_episodes(
        property_filter=FilterComparison(
            field="m.length",
            op="is_null",
            value=None,
        )
    )
    assert len(results) == 1
    assert set(results) == {episodes[1]}

    results = await long_term_memory.get_matching_episodes(
        property_filter=FilterAnd(
            left=FilterComparison(
                field="m.project",
                op="=",
                value="science",
            ),
            right=FilterComparison(
                field="metadata.length",
                op="=",
                value="short",
            ),
        )
    )
    assert len(results) == 1
    assert set(results) == {episodes[0]}

    results = await long_term_memory.get_matching_episodes(
        property_filter=FilterOr(
            left=FilterComparison(
                field="m.project",
                op="=",
                value="history",
            ),
            right=FilterComparison(
                field="metadata.project",
                op="=",
                value="science",
            ),
        )
    )
    assert len(results) == 3
    assert set(results) == {episodes[0], episodes[1], episodes[2]}

    results = await long_term_memory.get_matching_episodes(
        property_filter=FilterComparison(
            field="m.project",
            op="is_null",
            value=None,
        )
    )
    assert len(results) == 0

    results = await long_term_memory.get_matching_episodes(
        property_filter=FilterComparison(
            field="created_at",
            op="=",
            value=now,
        )
    )
    assert len(results) == 2
    assert set(results) == {episodes[0], episodes[1]}


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
        ["episode1", "episode3", "nonexistent_episode"]
    )
    all_episodes = await long_term_memory.get_matching_episodes()
    assert len(all_episodes) == 1
    assert set(all_episodes) == {episodes[1]}


@requires_sentence_transformers
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
        property_filter=FilterComparison(
            field="m.length",
            op="is_null",
            value=None,
        )
    )
    all_episodes = await long_term_memory.get_matching_episodes()
    assert len(all_episodes) == 2
    assert set(all_episodes) == {episodes[0], episodes[2]}

    await long_term_memory.delete_matching_episodes(
        property_filter=FilterComparison(
            field="metadata.length",
            op="=",
            value="medium",
        )
    )
    all_episodes = await long_term_memory.get_matching_episodes()
    assert len(all_episodes) == 2
    assert set(all_episodes) == {episodes[0], episodes[2]}

    await long_term_memory.delete_matching_episodes(
        property_filter=FilterComparison(
            field="m.project",
            op="=",
            value="history",
        )
    )
    all_episodes = await long_term_memory.get_matching_episodes()
    assert len(all_episodes) == 1
    assert set(all_episodes) == {episodes[0]}
