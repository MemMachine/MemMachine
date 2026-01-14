import argparse
import asyncio
import os
from datetime import datetime
from uuid import uuid4

import boto3
import neo4j
import openai
from dotenv import load_dotenv
from longmemeval_models import (
    LongMemEvalItem,
    load_longmemeval_dataset,
)

from memmachine.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
from memmachine.common.utils import async_with
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


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")

    args = parser.parse_args()

    data_path = args.data_path

    all_questions = load_longmemeval_dataset(data_path)
    num_questions = len(all_questions)
    print(f"{num_questions} total questions")

    neo4j_driver = neo4j.AsyncGraphDatabase.driver(
        uri=os.getenv("NEO4J_URI"),
        auth=(
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
        ),
    )

    vector_graph_store = Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            range_index_creation_threshold=10000,
            vector_index_creation_threshold=10000,
        )
    )

    openai_client = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
        )
    )

    region = "us-west-2"
    aws_client = boto3.client(
        "bedrock-agent-runtime",
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    reranker = AmazonBedrockReranker(
        AmazonBedrockRerankerParams(
            client=aws_client,
            region=region,
            model_id="cohere.rerank-v3-5:0",
        )
    )

    async def process_conversation(question: LongMemEvalItem):
        group_id = question.question_id
        session_ids = list(question.session_id_map.keys())

        memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=group_id,
                vector_graph_store=vector_graph_store,
                embedder=embedder,
                reranker=reranker,
            )
        )

        session_tasks = []
        for session_id in session_ids:
            session = question.get_session(session_id)

            episodes = []
            for turn in session:
                timestamp = datetime.fromisoformat(turn.timestamp)
                episodes.append(
                    Episode(
                        uid=str(uuid4()),
                        timestamp=timestamp,
                        source="Assistant" if turn.role == "assistant" else "User",
                        content_type=ContentType.MESSAGE,
                        content=turn.content.strip(),
                        user_metadata={
                            "longmemeval_session_id": session_id,
                        },
                    )
                )

            session_tasks.append(memory.add_episodes(episodes=episodes))

        await asyncio.gather(*session_tasks)

    semaphore = asyncio.Semaphore(1)
    tasks = [
        async_with(semaphore, process_conversation(question))
        for question in all_questions
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
