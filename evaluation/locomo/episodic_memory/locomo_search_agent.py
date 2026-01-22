import argparse
import asyncio
import json
import os
import re
import time
import traceback
from typing import Any

import boto3
import neo4j
import openai
from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
    trace,
)
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pydantic import BaseModel

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
    DeclarativeMemory,
    DeclarativeMemoryParams,
)

language = "english"
stop_words = stopwords.words(language)


def default_tokenize(text: str) -> list[str]:
    """
    Preprocess the input text
    by removing non-alphanumeric characters,
    converting to lowercase,
    word-tokenizing,
    and removing stop words.

    Args:
        text (str): The input text to preprocess.

    Returns:
        list[str]: A list of tokens for use in BM25 scoring.
    """
    alphanumeric_text = re.sub(r"\W+", " ", text)
    lower_text = alphanumeric_text.lower()
    words = word_tokenize(lower_text, language)
    tokens = [word for word in words if word and word not in stop_words]
    return tokens


def convert_for_json(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable format"""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    if hasattr(obj, "__dict__"):
        return {key: convert_for_json(value) for key, value in obj.__dict__.items()}
    if isinstance(obj, str):
        try:
            return convert_for_json(json.loads(obj))
        except Exception:
            return obj
    else:
        # For non-serializable types, convert to string
        return str(obj)


LOCOMO_INSTRUCTIONS = """
You are asked to answer a question based on your memories of a conversation.

<procedure>
1. First, the question has been used directly as a contextual cue to retrieve some relevant base memories.
2. Reason about the base memories to break down the question into sub-questions or identify specific details that need to be recalled.
3. Use these sub-questions and details to come up with new cues and follow-up questions to retrieve more memories using the retrieve_memories tool.
4. You may use the retrieve_memories tool as many times as necessary to retrieve all relevant memories.
5. Finally, synthesize all the retrieved memories to formulate a concise and accurate answer to the original question.
</procedure>

<guidelines>
1. Prioritize memories that answer the question directly. Be meticulous about identifying details.
2. When there may be multiple answers to the question, use effort to retrieve memories to list all possible answers. Do not become satisfied with just the first few answers retrieved.
3. When asked about time intervals or to count items, do not rush to answer immediately. Instead, carefully compute the answer.
4. Your memories are episodic, meaning that they consist of only your raw observations of what was said. You may need to reason about or guess what the memories imply in order to answer the question.
5. The question may contain typos or be based on the asker's own unreliable memories. Do your best to answer the question using the most relevant information in your memories.
6. Your memories may include small or large jumps in time or context. Do not assume that the information is complete.
7. Your memories are ordered from earliest to latest for each time you retrieve more memories using the retrieve_memories tool.
8. Your final response to the question should be no more than a couple of sentences.
</guidelines>

<base>
{memories}
</base>

<question>
{question}
</question>
"""


async def process_question(
    memory: DeclarativeMemory,
    question,
    answer,
    category,
    evidence,
    adversarial_answer,
):
    @function_tool(name_override="retrieve_memories")
    async def retrieve_memories(cue: str) -> str:
        """
        Retrieve relevant memories based on the provided cue.
        The cue should be a complete question or phrase to help retrieve relevant memories.

        Args:
            cue (str): A cue used to retrieve relevant memories.
        """
        episodes = await memory.search(
            query=cue,
            max_num_episodes=30,
        )
        formatted_context = memory.string_from_episode_context(episodes)
        return formatted_context

    episodes = await memory.search(
        query=question,
        max_num_episodes=30,
    )
    prefetched_context = memory.string_from_episode_context(episodes)

    locomo_agent = Agent(
        name="agent",
        instructions=LOCOMO_INSTRUCTIONS.format(
            memories=prefetched_context, question=question
        ),
        model="gpt-4.1-mini",
        model_settings=ModelSettings(max_tokens=2000, temperature=0.2, store=False),
        tools=[retrieve_memories],
    )

    agent_start = time.monotonic()

    with trace("locomo"):
        try:
            run_result = await Runner.run(
                locomo_agent,
                input=question,
                max_turns=20,
            )

            agent_trace = [
                {str(type(item).__name__): convert_for_json(item.raw_item)}
                for item in run_result.new_items
            ]

            results = {
                "response": run_result.final_output.strip(),
                "trace": agent_trace,
            }
        except Exception:
            traceback.print_exc()
            results = {"response": "Error", "trace": "None"}

    agent_end = time.monotonic()

    print(
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Response: {results['response']}\n"
        f"Agent time: {agent_end - agent_start:.2f} seconds\n"
    )
    return {
        "question": question,
        "locomo_answer": answer,
        "model_answer": results["response"],
        "category": category,
        "evidence": evidence,
        "adversarial_answer": adversarial_answer,
        "agent_time": agent_end - agent_start,
        "agent_trace": results["trace"],
    }


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

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

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

    results: dict[str, Any] = {}
    for idx, item in enumerate(locomo_data):
        if "conversation" not in item:
            continue

        qa_list = item["qa"]

        print(f"Processing questions for group {idx}...")

        group_id = f"group_{idx}"

        memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=group_id,
                vector_graph_store=vector_graph_store,
                embedder=embedder,
                reranker=reranker,
            )
        )

        async def respond_question(qa):
            question = qa["question"]
            answer = qa.get("answer", "")
            category = qa["category"]
            evidence = qa["evidence"]

            adversarial_answer = qa.get("adversarial_answer", "")

            question_response = await process_question(
                memory,
                question,
                answer,
                category,
                evidence,
                adversarial_answer,
            )
            return (
                category,
                question_response,
            )

        semaphore = asyncio.Semaphore(30)
        response_tasks = [
            async_with(
                semaphore,
                respond_question(qa),
            )
            for qa in qa_list
        ]

        responses = await asyncio.gather(*response_tasks)

        for category, response in responses:
            category_result = results.get(category, [])
            category_result.append(response)
            results[category] = category_result

    with open(target_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
