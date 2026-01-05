from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from sentence_transformers import CrossEncoder, SentenceTransformer
from testcontainers.neo4j import Neo4jContainer

from memmachine.common.embedder.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    SentenceTransformerEmbedderParams,
)
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.reranker.cross_encoder_reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankerParams,
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
    assert results[0].uid == "episode1"

    results = await declarative_memory.search(
        query="Who wrote the test?",
        max_num_episodes=4,
    )

    assert len(results) == 4
    # Most relevant.
    assert "episode1" in [result.uid for result in results]

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

    assert len(results) == 2
    assert "episode1" in [result.uid for result in results]
    assert "episode2" in [result.uid for result in results]


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
            right=FilterComparison(
                field="length",
                op="is_null",
                value=None,
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
