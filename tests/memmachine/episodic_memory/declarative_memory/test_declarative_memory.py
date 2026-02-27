from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer

from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    In as FilterIn,
)
from memmachine.common.filter.filter_parser import (
    IsNull as FilterIsNull,
)
from memmachine.common.filter.filter_parser import (
    Not as FilterNot,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine.episodic_memory.declarative_memory import (
    ContentType,
    DeclarativeMemory,
    DeclarativeMemoryParams,
    Episode,
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
def vector_graph_store(neo4j_driver):
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            force_exact_similarity_search=True,
        ),
    )


@pytest.fixture(scope="module")
def declarative_memory(embedder, reranker, vector_graph_store):
    return DeclarativeMemory(
        DeclarativeMemoryParams(
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
async def clear_declarative_memory(declarative_memory):
    all_episodes = await declarative_memory.get_matching_episodes()
    await declarative_memory.delete_episodes([episode.uid for episode in all_episodes])
    yield


@requires_sentence_transformers
@pytest.mark.asyncio
async def test_add_episodes(declarative_memory):
    all_episodes = await declarative_memory.get_matching_episodes()
    assert len(all_episodes) == 0

    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="episode1",
            timestamp=now,
            source="Alice",
            content_type=ContentType.MESSAGE,
            content="This test is broken. Who wrote this test?",
            filterable_properties={"project": "memmachine", "length": "short"},
            user_metadata={"some_key": "some_value"},
        ),
        Episode(
            uid="episode2",
            timestamp=now + timedelta(seconds=10),
            source="Bob",
            content_type=ContentType.MESSAGE,
            content=("Edwin Yu: https://github.com/edwinyyyu\n"),
            user_metadata={"some_other_key": "some_other_value"},
        ),
        Episode(
            uid="episode3",
            timestamp=now + timedelta(seconds=20),
            source="textbook",
            content_type=ContentType.TEXT,
            content="The mitochondria is the powerhouse of the cell.",
            filterable_properties={"project": "other", "length": "short"},
        ),
        Episode(
            uid="episode4",
            timestamp=now + timedelta(seconds=30),
            source="pet rock",
            content_type=ContentType.MESSAGE,
            content="",
        ),
    ]

    await declarative_memory.add_episodes(episodes)

    all_episodes = await declarative_memory.get_matching_episodes()
    assert len(all_episodes) == len(episodes)


@requires_sentence_transformers
@pytest.mark.asyncio
async def test_search(declarative_memory):
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid=str(uuid4()),
            timestamp=now - i * timedelta(seconds=1),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "testing", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    episodes += [
        Episode(
            uid=str(uuid4()),
            timestamp=now - i * timedelta(seconds=1),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "memmachine", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    episodes += [
        Episode(
            uid="episode1",
            timestamp=now,
            source="Alice",
            content_type=ContentType.MESSAGE,
            content="This test is broken. Who wrote this test?",
            filterable_properties={"project": "memmachine", "length": "short"},
            user_metadata={"some_key": "some_value"},
        ),
        Episode(
            uid="episode2",
            timestamp=now + timedelta(seconds=10),
            source="Bob",
            content_type=ContentType.MESSAGE,
            content="Charlie.",
            filterable_properties={"project": "other", "length": "short"},
            user_metadata={"some_other_key": "some_other_value"},
        ),
        Episode(
            uid="episode3",
            timestamp=now + timedelta(seconds=20),
            source="textbook",
            content_type=ContentType.TEXT,
            content="The mitochondria is the powerhouse of the cell.",
        ),
        Episode(
            uid="episode4",
            timestamp=now + timedelta(seconds=30),
            source="pet rock",
            content_type=ContentType.MESSAGE,
            content="",
        ),
        Episode(
            uid="episode5",
            timestamp=now + timedelta(seconds=40),
            source="Charlie",
            content_type=ContentType.MESSAGE,
            content="Edwin Yu: https://github.com/edwinyyyu\n",
            filterable_properties={"project": "memmachine"},
        ),
        Episode(
            uid="episode6",
            timestamp=now + timedelta(seconds=50),
            source="Edwin",
            content_type=ContentType.MESSAGE,
            content="I wrote this test.",
            filterable_properties={"project": "memmachine", "length": "short"},
        ),
    ]
    episodes += [
        Episode(
            uid=str(uuid4()),
            timestamp=now + i * timedelta(seconds=1),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "testing", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    episodes += [
        Episode(
            uid=str(uuid4()),
            timestamp=now + i * timedelta(seconds=100),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "memmachine", "length": "medium"},
        )
        for i in range(1, 11)
    ]

    await declarative_memory.add_episodes(episodes)

    results = await declarative_memory.search(
        query="Who wrote the test?",
        max_num_episodes=1,
    )

    assert len(results) == 1
    assert results[0].uid == "episode1" or results[0].uid == "episode6"

    results = await declarative_memory.search(
        query="Who wrote the test?",
        max_num_episodes=4,
        expand_context=0,
    )

    assert len(results) == 4
    # Most relevant.
    assert "episode1" in [result.uid for result in results]
    assert "episode6" in [result.uid for result in results]

    results = await declarative_memory.search(
        query="Who wrote the test?",
        max_num_episodes=4,
        expand_context=3,
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

    results = await declarative_memory.search(
        query="Who wrote the test?",
        max_num_episodes=10,
        property_filter=FilterComparison(
            field="project",
            op="=",
            value="memmachine",
        ),
    )
    assert len(results) == 10
    assert "episode1" in [result.uid for result in results]
    assert "episode5" in [result.uid for result in results]

    results = await declarative_memory.search(
        query="Who wrote the test?",
        max_num_episodes=4,
        property_filter=FilterComparison(
            field="length",
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
async def test_get_episodes(declarative_memory):
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid=str(uuid4()),
            timestamp=now - i * timedelta(seconds=1),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "testing", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    episodes += [
        Episode(
            uid=str(uuid4()),
            timestamp=now - i * timedelta(seconds=1),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "memmachine", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    special_episodes = [
        Episode(
            uid="episode1",
            timestamp=now,
            source="Alice",
            content_type=ContentType.MESSAGE,
            content="This test is broken. Who wrote this test?",
            filterable_properties={"project": "memmachine", "length": "short"},
            user_metadata={"some_key": "some_value"},
        ),
        Episode(
            uid="episode2",
            timestamp=now + timedelta(seconds=10),
            source="Bob",
            content_type=ContentType.MESSAGE,
            content="Charlie.",
            filterable_properties={"project": "other", "length": "short"},
            user_metadata={"some_other_key": "some_other_value"},
        ),
        Episode(
            uid="episode3",
            timestamp=now + timedelta(seconds=20),
            source="textbook",
            content_type=ContentType.TEXT,
            content="The mitochondria is the powerhouse of the cell.",
        ),
        Episode(
            uid="episode4",
            timestamp=now + timedelta(seconds=30),
            source="pet rock",
            content_type=ContentType.MESSAGE,
            content="",
        ),
        Episode(
            uid="episode5",
            timestamp=now + timedelta(seconds=40),
            source="Charlie",
            content_type=ContentType.MESSAGE,
            content="Edwin Yu: https://github.com/edwinyyyu\n",
            filterable_properties={"project": "memmachine"},
        ),
    ]
    episodes += special_episodes
    episodes += [
        Episode(
            uid=str(uuid4()),
            timestamp=now + i * timedelta(seconds=1),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "testing", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    episodes += [
        Episode(
            uid=str(uuid4()),
            timestamp=now + i * timedelta(seconds=100),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "memmachine", "length": "medium"},
        )
        for i in range(1, 11)
    ]

    await declarative_memory.add_episodes(episodes)

    results = await declarative_memory.get_episodes(
        [
            "episode1",
            "episode2",
            "episode3",
            "episode4",
            "episode5",
            "nonexistent_episode",
        ],
    )
    assert len(results) == 5
    assert set(results) == set(special_episodes)


@requires_sentence_transformers
@pytest.mark.asyncio
async def test_get_matching_episodes(declarative_memory):
    now = datetime.now(tz=UTC)

    episodes = [
        Episode(
            uid=str(uuid4()),
            timestamp=now - i * timedelta(seconds=1),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "testing", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    episodes += [
        Episode(
            uid=str(uuid4()),
            timestamp=now - i * timedelta(seconds=1),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "memmachine", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    episodes += [
        Episode(
            uid="episode1",
            timestamp=now,
            source="Alice",
            content_type=ContentType.MESSAGE,
            content="This test is broken. Who wrote this test?",
            filterable_properties={"project": "memmachine", "length": "short"},
            user_metadata={"some_key": "some_value"},
        ),
        Episode(
            uid="episode2",
            timestamp=now + timedelta(seconds=10),
            source="Bob",
            content_type=ContentType.MESSAGE,
            content="Charlie.",
            filterable_properties={"project": "other", "length": "short"},
            user_metadata={"some_other_key": "some_other_value"},
        ),
        Episode(
            uid="episode3",
            timestamp=now + timedelta(seconds=20),
            source="textbook",
            content_type=ContentType.TEXT,
            content="The mitochondria is the powerhouse of the cell.",
        ),
        Episode(
            uid="episode4",
            timestamp=now + timedelta(seconds=30),
            source="pet rock",
            content_type=ContentType.MESSAGE,
            content="",
        ),
        Episode(
            uid="episode5",
            timestamp=now + timedelta(seconds=40),
            source="Charlie",
            content_type=ContentType.MESSAGE,
            content="Edwin Yu: https://github.com/edwinyyyu\n",
            filterable_properties={"project": "memmachine"},
        ),
    ]
    episodes += [
        Episode(
            uid=str(uuid4()),
            timestamp=now + i * timedelta(seconds=1),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "testing", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    episodes += [
        Episode(
            uid=str(uuid4()),
            timestamp=now + i * timedelta(seconds=100),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "memmachine", "length": "medium"},
        )
        for i in range(1, 11)
    ]

    await declarative_memory.add_episodes(episodes)

    results = await declarative_memory.get_matching_episodes(
        property_filter=FilterComparison(
            field="project",
            op="=",
            value="memmachine",
        ),
    )
    assert len(results) == 22

    results = await declarative_memory.get_matching_episodes(
        property_filter=FilterAnd(
            left=FilterComparison(
                field="project",
                op="=",
                value="memmachine",
            ),
            right=FilterIsNull(
                field="length",
            ),
        ),
    )
    assert len(results) == 1

    results = await declarative_memory.get_matching_episodes(
        property_filter=FilterComparison(
            field="length",
            op="=",
            value="short",
        )
    )
    assert len(results) == 2

    results = await declarative_memory.get_matching_episodes(
        property_filter=FilterAnd(
            left=FilterComparison(
                field="project",
                op="=",
                value="memmachine",
            ),
            right=FilterComparison(
                field="length",
                op="=",
                value="short",
            ),
        ),
    )
    assert len(results) == 1

    results = await declarative_memory.get_matching_episodes()
    assert len(results) == 45


@requires_sentence_transformers
@pytest.mark.asyncio
async def test_delete_episodes(declarative_memory):
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid=str(uuid4()),
            timestamp=now - i * timedelta(seconds=1),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "testing", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    episodes += [
        Episode(
            uid=str(uuid4()),
            timestamp=now - i * timedelta(seconds=1),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "memmachine", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    special_episodes = [
        Episode(
            uid="episode1",
            timestamp=now,
            source="Alice",
            content_type=ContentType.MESSAGE,
            content="This test is broken. Who wrote this test?",
            filterable_properties={"project": "memmachine", "length": "short"},
            user_metadata={"some_key": "some_value"},
        ),
        Episode(
            uid="episode2",
            timestamp=now + timedelta(seconds=10),
            source="Bob",
            content_type=ContentType.MESSAGE,
            content="Charlie.",
            filterable_properties={"project": "other", "length": "short"},
            user_metadata={"some_other_key": "some_other_value"},
        ),
        Episode(
            uid="episode3",
            timestamp=now + timedelta(seconds=20),
            source="textbook",
            content_type=ContentType.TEXT,
            content="The mitochondria is the powerhouse of the cell.",
        ),
        Episode(
            uid="episode4",
            timestamp=now + timedelta(seconds=30),
            source="pet rock",
            content_type=ContentType.MESSAGE,
            content="",
        ),
        Episode(
            uid="episode5",
            timestamp=now + timedelta(seconds=40),
            source="Charlie",
            content_type=ContentType.MESSAGE,
            content="Edwin Yu: https://github.com/edwinyyyu\n",
            filterable_properties={"project": "memmachine"},
        ),
    ]
    episodes += special_episodes
    episodes += [
        Episode(
            uid=str(uuid4()),
            timestamp=now + i * timedelta(seconds=1),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "testing", "length": "medium"},
        )
        for i in range(1, 11)
    ]
    episodes += [
        Episode(
            uid=str(uuid4()),
            timestamp=now + i * timedelta(seconds=100),
            source="filler",
            content_type=ContentType.MESSAGE,
            content=str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4())
            + str(uuid4()),
            filterable_properties={"project": "memmachine", "length": "medium"},
        )
        for i in range(1, 11)
    ]

    await declarative_memory.add_episodes(episodes)

    all_episodes = await declarative_memory.get_matching_episodes()
    assert len(all_episodes) == 45

    await declarative_memory.delete_episodes(
        [episode.uid for episode in special_episodes],
    )

    all_episodes = await declarative_memory.get_matching_episodes()
    assert len(all_episodes) == 40
    assert all(episode not in all_episodes for episode in special_episodes)


@requires_sentence_transformers
def test_string_from_episode_context():
    now = datetime.now(tz=UTC)
    episode1 = Episode(
        uid="episode1",
        timestamp=now,
        source="Alice",
        content_type=ContentType.MESSAGE,
        content="This test is broken. Who wrote this test?",
    )
    episode2 = Episode(
        uid="episode2",
        timestamp=now + timedelta(seconds=10),
        source="Bob",
        content_type=ContentType.MESSAGE,
        content="Edwin.",
    )
    episode3 = Episode(
        uid="episode3",
        timestamp=now + timedelta(seconds=20),
        source="textbook",
        content_type=ContentType.TEXT,
        content="The mitochondria is the powerhouse of the cell.",
    )

    context_string = DeclarativeMemory.string_from_episode_context(
        [episode1, episode2, episode3],
    )

    assert episode1.content in context_string
    assert episode2.content in context_string
    assert episode3.content in context_string


@requires_sentence_transformers
@pytest.mark.asyncio
async def test_get_matching_episodes_extended_filters(declarative_memory):
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="episode1",
            timestamp=now,
            source="Alice",
            content_type=ContentType.MESSAGE,
            content="This test is broken. Who wrote this test?",
            filterable_properties={"project": "memmachine", "length": "short"},
            user_metadata={"some_key": "some_value"},
        ),
        Episode(
            uid="episode2",
            timestamp=now + timedelta(seconds=10),
            source="Bob",
            content_type=ContentType.MESSAGE,
            content="Charlie.",
            filterable_properties={"project": "other", "length": "short"},
            user_metadata={"some_other_key": "some_other_value"},
        ),
        Episode(
            uid="episode3",
            timestamp=now + timedelta(seconds=20),
            source="textbook",
            content_type=ContentType.TEXT,
            content="The mitochondria is the powerhouse of the cell.",
            filterable_properties={"project": "testing", "length": "long"},
        ),
        Episode(
            uid="episode4",
            timestamp=now + timedelta(seconds=30),
            source="pet rock",
            content_type=ContentType.MESSAGE,
            content="",
        ),
        Episode(
            uid="episode5",
            timestamp=now + timedelta(seconds=40),
            source="Charlie",
            content_type=ContentType.MESSAGE,
            content="Edwin Yu: https://github.com/edwinyyyu\n",
            filterable_properties={"project": "memmachine"},
        ),
    ]

    await declarative_memory.add_episodes(episodes)

    # != on project: != 'memmachine' → episodes with project=other, testing
    results = await declarative_memory.get_matching_episodes(
        property_filter=FilterComparison(
            field="project",
            op="!=",
            value="memmachine",
        ),
    )
    result_uids = {r.uid for r in results}
    assert "episode2" in result_uids
    assert "episode3" in result_uids
    assert "episode1" not in result_uids
    assert "episode5" not in result_uids

    # In on project
    results = await declarative_memory.get_matching_episodes(
        property_filter=FilterIn(
            field="project",
            values=["memmachine", "other"],
        ),
    )
    result_uids = {r.uid for r in results}
    assert "episode1" in result_uids
    assert "episode2" in result_uids
    assert "episode5" in result_uids

    # Not: NOT length = 'short'
    results = await declarative_memory.get_matching_episodes(
        property_filter=FilterNot(
            expr=FilterComparison(
                field="length",
                op="=",
                value="short",
            )
        ),
    )
    result_uids = {r.uid for r in results}
    assert "episode1" not in result_uids
    assert "episode2" not in result_uids
    assert "episode3" in result_uids

    # Or: project = 'other' OR length = 'short'
    results = await declarative_memory.get_matching_episodes(
        property_filter=FilterOr(
            left=FilterComparison(
                field="project",
                op="=",
                value="other",
            ),
            right=FilterComparison(
                field="length",
                op="=",
                value="short",
            ),
        ),
    )
    result_uids = {r.uid for r in results}
    assert "episode1" in result_uids
    assert "episode2" in result_uids


# =====================================================================
# PRODUCED_BY relationship (Episode -> Producer)
# =====================================================================


@requires_sentence_transformers
@pytest.mark.asyncio
async def test_produced_by_relationship_created(declarative_memory, neo4j_driver):
    """Task 4.2: Episode gets a PRODUCED_BY relationship to a Producer node."""
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="ep-prod-1",
            timestamp=now,
            source="testuser",
            content_type=ContentType.MESSAGE,
            content="Hello from testuser.",
        ),
    ]
    await declarative_memory.add_episodes(episodes)

    records, _, _ = await neo4j_driver.execute_query(
        """
        MATCH (e {uid: $uid})-[:PRODUCED_BY]->(p)
        WHERE any(label IN labels(p) WHERE label STARTS WITH 'SANITIZED_Producer')
        RETURN p.producer_id AS producer_id
        """,
        uid="ep-prod-1",
    )

    assert len(records) == 1
    assert records[0]["producer_id"] == "testuser"


@requires_sentence_transformers
@pytest.mark.asyncio
async def test_multiple_episodes_same_producer_share_node(
    declarative_memory, neo4j_driver
):
    """Task 4.3: Multiple episodes from same producer share one Producer node."""
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="ep-shared-1",
            timestamp=now,
            source="alice",
            content_type=ContentType.MESSAGE,
            content="First message from alice.",
        ),
        Episode(
            uid="ep-shared-2",
            timestamp=now + timedelta(seconds=1),
            source="alice",
            content_type=ContentType.MESSAGE,
            content="Second message from alice.",
        ),
        Episode(
            uid="ep-shared-3",
            timestamp=now + timedelta(seconds=2),
            source="alice",
            content_type=ContentType.MESSAGE,
            content="Third message from alice.",
        ),
    ]
    await declarative_memory.add_episodes(episodes)

    records, _, _ = await neo4j_driver.execute_query(
        """
        MATCH (e)-[:PRODUCED_BY]->(p {producer_id: $pid})
        WHERE any(label IN labels(p) WHERE label STARTS WITH 'SANITIZED_Producer')
        RETURN count(e) AS episode_count, count(DISTINCT p) AS producer_count
        """,
        pid="alice",
    )

    assert records[0]["episode_count"] == 3
    assert records[0]["producer_count"] == 1


@requires_sentence_transformers
@pytest.mark.asyncio
async def test_different_producers_separate_nodes(declarative_memory, neo4j_driver):
    """Task 4.4: Episodes from different producers create separate Producer nodes."""
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="ep-diff-1",
            timestamp=now,
            source="alice",
            content_type=ContentType.MESSAGE,
            content="Message from alice.",
        ),
        Episode(
            uid="ep-diff-2",
            timestamp=now + timedelta(seconds=1),
            source="bob",
            content_type=ContentType.MESSAGE,
            content="Message from bob.",
        ),
    ]
    await declarative_memory.add_episodes(episodes)

    records, _, _ = await neo4j_driver.execute_query(
        """
        MATCH (p)
        WHERE any(label IN labels(p) WHERE label STARTS WITH 'SANITIZED_Producer')
          AND p.producer_id IN $pids
        RETURN p.producer_id AS pid
        ORDER BY pid
        """,
        pids=["alice", "bob"],
    )

    assert len(records) == 2
    assert records[0]["pid"] == "alice"
    assert records[1]["pid"] == "bob"
