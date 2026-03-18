import argparse
import asyncio
import json
import os
import time

import boto3
import neo4j
from dotenv import load_dotenv
from longmemeval_models import (
    LongMemEvalItem,
    get_datetime_from_timestamp,
    load_longmemeval_dataset,
)
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine_server.episodic_memory.declarative_memory import (
    DeclarativeMemory,
    DeclarativeMemoryParams,
)
from openai import AsyncOpenAI

# Parts of prompt borrowed from Mastra's OM.
# https://github.com/mastra-ai/mastra/blob/977b49e23d8b050a2c6a6a91c0aa38b28d6388ee/packages/memory/src/processors/observational-memory/observational-memory.ts#L312-L318
ANSWER_PROMPT = """
You are a helpful assistant with access to extensive conversation history.
When answering questions, carefully review the conversation history to identify and use any relevant user preferences, interests, or specific details they have mentioned.

<history>
{memories}
</history>

IMPORTANT: When responding, reference specific details from these observations. Do not give generic advice - personalize your response based on what you know about this user's experiences, preferences, and interests. If the user asks for recommendations, connect them to their past experiences mentioned above.

KNOWLEDGE UPDATES: When asked about current state (e.g., "where do I currently...", "what is my current..."), always prefer the MOST RECENT information. Observations include dates - if you see conflicting information, the newer observation supersedes the older one. Look for phrases like "will start", "is switching", "changed to", "moved to" as indicators that previous information has been updated.

PLANNED ACTIONS: If the user stated they planned to do something (e.g., "I'm going to...", "I'm looking forward to...", "I will...") and the date they planned to do it is now in the past (check the relative time like "3 weeks ago"), assume they completed the action unless there's evidence they didn't. For example, if someone said "I'll start my new diet on Monday" and that was 2 weeks ago, assume they started the diet.

Current date: {question_timestamp}
Question: {question}
"""


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )

    args = parser.parse_args()

    data_path = args.data_path
    target_path = args.target_path

    all_questions = load_longmemeval_dataset(data_path)

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
            max_concurrent_transactions=1000,
        )
    )

    openai_client = AsyncOpenAI(
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

    async def qa_eval(
        memories,
        question_timestamp,
        question: str,
        model: str = "gpt-5-mini",
    ):
        start_time = time.monotonic()
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(
                        memories=memories,
                        question_timestamp=question_timestamp,
                        question=question,
                    ),
                },
            ],
        )
        end_time = time.monotonic()

        latency = end_time - start_time

        return {
            "response": response.choices[0].message.content.strip(),
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "latency": latency,
        }

    async def process_question(
        question: LongMemEvalItem,
    ):
        group_id = question.question_id

        memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=group_id,
                vector_graph_store=vector_graph_store,
                embedder=embedder,
                reranker=reranker,
            )
        )

        search_query = f"User: {question.question}"

        total_start = time.monotonic()
        memory_start = time.monotonic()
        chunks = await memory.search(
            query=search_query, max_num_episodes=100, expand_context=0
        )
        memory_end = time.monotonic()
        memory_latency = memory_end - memory_start

        formatted_context = memory.string_from_episode_context(chunks)

        response = await qa_eval(
            formatted_context,
            get_datetime_from_timestamp(question.question_date).strftime(
                "%A, %B %d, %Y at %I:%M %p"
            ),
            question.question,
        )
        total_end = time.monotonic()
        total_latency = total_end - total_start

        print(
            f"Question ID: {question.question_id}\n"
            f"Question: {question.question}\n"
            f"Question Date: {question.question_date}\n"
            f"Question Type: {question.question_type}\n"
            f"Answer: {question.answer}\n"
            f"Response: {response['response']}\n"
            f"Memory retrieval time: {memory_latency:.2f} seconds\n"
            f"LLM response time: {response['latency']:.2f} seconds\n"
            f"Total processing time: {total_latency:.2f} seconds\n"
            f"MEMORIES_START\n{formatted_context}MEMORIES_END\n"
        )

        return {
            "question_id": question.question_id,
            "question_date": question.question_date,
            "question": question.question,
            "answer": question.answer,
            "response": response["response"],
            "question_type": question.question_type.value,
            "abstention": question.abstention_question,
            "total_latency": total_latency,
            "memory_latency": memory_latency,
            "llm_latency": response["latency"],
            "episodes_text": formatted_context,
        }

    semaphore = asyncio.Semaphore(5)
    tasks = [
        async_with(
            semaphore,
            process_question(question),
        )
        for question in all_questions
    ]
    results = await asyncio.gather(*tasks)

    with open(target_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
