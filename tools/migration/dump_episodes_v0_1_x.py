import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import cast
from uuid import UUID

from memmachine.episodic_memory.data_types import ContentType, Episode
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel, JsonValue

from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
)
from memmachine.episodic_memory.declarative_memory import DeclarativeMemory


class EpisodeModel(BaseModel):
    uuid: UUID
    episode_type: str
    content_type: str
    content: str
    timestamp: datetime
    group_id: str
    session_id: str
    producer_id: str
    produced_for_id: str
    user_metadata: JsonValue


def write_json_batch(episodes: list[Episode], batch_index: int, output_dir: str):
    output_path = os.path.join(output_dir, f"episodes_batch_{batch_index}.json")

    episode_models = [
        EpisodeModel(
            uuid=episode.uuid,
            episode_type=episode.episode_type,
            content_type=episode.content_type.value,
            content=episode.content,
            timestamp=episode.timestamp,
            group_id=episode.group_id,
            session_id=episode.session_id,
            producer_id=episode.producer_id,
            produced_for_id=episode.produced_for_id,
            user_metadata=episode.user_metadata,
        )
        for episode in episodes
    ]

    data = [episode_model.model_dump() for episode_model in episode_models]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str, indent=2)


async def dump_episodes(
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    output_dir: str,
    batch_size: int,
):
    first_query = """
    MATCH (n:Episode)
    RETURN n
    ORDER BY elementId(n)
    LIMIT $batch_size
    """

    query = """
    MATCH (n:Episode)
    WHERE elementId(n) > $last_id
    RETURN n
    ORDER BY elementId(n)
    LIMIT $batch_size
    """

    neo4j_driver = AsyncGraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_username, neo4j_password),
    )

    batch_index = 0
    records, _, _ = await neo4j_driver.execute_query(first_query, batch_size=batch_size)
    while True:
        neo4j_nodes = [record["n"] for record in records]
        nodes = Neo4jVectorGraphStore._nodes_from_neo4j_nodes(neo4j_nodes)  # noqa: SLF001
        declarative_memory_episodes = DeclarativeMemory._episodes_from_episode_nodes(  # noqa: SLF001
            nodes
        )
        long_term_memory_episodes = [
            Episode(
                uuid=declarative_memory_episode.uuid,
                episode_type=declarative_memory_episode.episode_type,
                content_type=ContentType.STRING,
                content=declarative_memory_episode.content,
                timestamp=declarative_memory_episode.timestamp,
                group_id=cast(
                    str,
                    declarative_memory_episode.filterable_properties.get(
                        "group_id", ""
                    ),
                ),
                session_id=cast(
                    str,
                    declarative_memory_episode.filterable_properties.get(
                        "session_id", ""
                    ),
                ),
                producer_id=cast(
                    str,
                    declarative_memory_episode.filterable_properties.get(
                        "producer_id", ""
                    ),
                ),
                produced_for_id=cast(
                    str,
                    declarative_memory_episode.filterable_properties.get(
                        "produced_for_id", ""
                    ),
                ),
                user_metadata=declarative_memory_episode.user_metadata,
            )
            for declarative_memory_episode in declarative_memory_episodes
        ]

        write_json_batch(long_term_memory_episodes, batch_index, output_dir)

        if len(neo4j_nodes) < batch_size:
            break

        batch_index += 1
        last_id = neo4j_nodes[-1].element_id
        records, _, _ = await neo4j_driver.execute_query(
            query,
            last_id=last_id,
            batch_size=batch_size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch episodes from Neo4j")
    parser.add_argument("--uri", type=str, required=True, help="Neo4j connection URI")
    parser.add_argument("--username", type=str, required=True, help="Neo4j username")
    parser.add_argument("--password", type=str, required=True, help="Neo4j password")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for episodes JSON files",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Number of episodes per batch"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    asyncio.run(
        dump_episodes(
            neo4j_uri=args.uri,
            neo4j_username=args.username,
            neo4j_password=args.password,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    )
