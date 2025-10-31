import argparse
import asyncio
import json
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import cast
import time
import os

from dotenv import load_dotenv

from memmachine.episodic_memory.data_types import ContentType
from memmachine.episodic_memory.episodic_memory import EpisodicMemory
from memmachine.episodic_memory.episodic_memory_manager import (
    EpisodicMemoryManager,
)
from memmachine.knowledge_graph.re_gpt4_1 import KnowledgeGraph
from memmachine.common.vector_graph_store import Node

def load_data(
    start_line: int = 1,
    num_cases: int = 100,
    randomize: bool = True,
):
    # dataset = "/home/tomz/MemMachine/evaluation/locomo/wiki-filter-gpt-4o-mini.json"
    dataset = "/home/tomz/MemMachine/evaluation/locomo/wikimultihop.json"
    print(f"Loading data from line {start_line} to {num_cases}, randomize={randomize}")
    contexts = []
    questions = []
    answers = []
    types = []
    i = 1
    with open(dataset, "r", encoding="utf-8") as f:
        for line in f:
            if i < start_line:
                i += 1
                continue
            if i > num_cases:
                break

            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["context"] = json.loads(obj["context"])
            c_list = []
            for key, sentences in obj["context"]:
                for s in sentences:
                    c = f"{key}: {s}"
                    if randomize:
                        insert_index = random.randrange(len(contexts) + 1)  # 0..len inclusive
                        c_list.insert(insert_index, c)
                    else:
                        c_list.append(c)

            contexts.append(c_list)
            questions.append(obj["question"])
            answers.append(obj["answer"])
            types.append(obj["type"])
            i += 1
    return contexts, questions, answers, types

async def ingest_wikimultihop():
    memory_manager = EpisodicMemoryManager.create_episodic_memory_manager(
        "locomo_config.yaml"
    )

    memory = cast(
        EpisodicMemory,
        await memory_manager.get_episodic_memory_instance(
            group_id="1",
            session_id="1",
            user_id=["user"],
        ),
    )

    contexts, _, _, _ = load_data(start_line=1, num_cases=305, randomize=False)

    print("Loaded", len(contexts), "contexts, start ingestion...")
    
    num_batch = 50
    em_tasks = []
    episodes = []
    added_contexts = set()
    num_added = 0
    t1 = datetime.now(timezone.utc)
    for c_list in contexts:
        for c in c_list:
            if c not in added_contexts:
                added_contexts.add(c)
                num_added += 1
                # if num_added <= 7885:
                #     continue

                cur_uuid = uuid.uuid4()
                ts = t1 + timedelta(seconds=len(added_contexts))
                episodes.append(Node(
                    uuid=cur_uuid,
                    labels={"Episode"},
                    # Make timestamp different for each episode
                    properties={
                        "content": c,
                        "timestamp": ts,
                        "session_id": 1
                    },
                ))

                producer = c.split(":")[0]
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                em_tasks.append(memory.add_memory_episode(
                        producer="user",
                        produced_for="user",
                        episode_content=c,
                        episode_type="default",
                        content_type=ContentType.STRING,
                        timestamp=ts_str,
                        metadata={
                            "source_timestamp": ts_str,
                            "source_speaker": "user",
                        },
                        uuid=cur_uuid,
                    )
                )

                if len(added_contexts) % num_batch == 0 or (c_list == contexts[-1] and c == c_list[-1]):
                    t = time.perf_counter()
                    await add_episode_bulk(episodes)
                    print(f"Gathered and added {len(episodes)} episodes to KG in {(time.perf_counter() - t):.3f}s")
                    episodes = []

                    t = time.perf_counter()
                    await asyncio.gather(*em_tasks)
                    print(f"Added {len(em_tasks)} episodes to EM in {(time.perf_counter() - t):.3f}s")
                    em_tasks = []

                    print(f"Total added episodes: {len(added_contexts)}")
    
    print(f"Completed WIKI-Multihop ingestion, added {len(added_contexts)} episodes.")

async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")

    args = parser.parse_args()

    data_path = args.data_path

    memory_manager = EpisodicMemoryManager.create_episodic_memory_manager(
        "locomo_config.yaml"
    )

    from openai import AsyncOpenAI
    model = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=600,
    )

    from neo4j import AsyncGraphDatabase
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:9999",
        auth=(
            "neo4j",
            "password",
        ),
        connection_timeout=3600,                    # seconds
        # how long to wait for a pooled connection
        connection_acquisition_timeout=3600,        # seconds
        # built-in retry window for transient failures
        max_transaction_retry_time=3600,
        max_connection_lifetime=3600,
        keep_alive=True,
    )

    from memmachine.common.vector_graph_store.neo4j_vector_graph_store import Neo4jVectorGraphStore, Neo4jVectorGraphStoreParams
    store = Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=driver,
            max_concurrent_transactions=200,
            force_exact_similarity_search=False,
        )
    )

    await store.create_fulltext_index()

    from memmachine.common.embedder.openai_embedder import OpenAIEmbedder
    embedder = OpenAIEmbedder(
        {
            "model": "text-embedding-3-small",
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    )

    # await ingest_wikimultihop()
    # return

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

    async def process_conversation(idx, item, memory_manager: EpisodicMemoryManager):
        if "conversation" not in item:
            return
        
        nonlocal model, store, embedder

        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        print(
            f"Processing conversation for group {idx} with speakers {speaker_a} and {speaker_b}..."
        )

        group_id = f"group_{idx}"

        model_name="gpt-4.1-mini"
        print(f"Creating Knowledge Graph with model {model_name}...")
        kg = KnowledgeGraph(
            model_name=model_name,
            model=model,
            embedder=embedder,
            store=store,
        )

        memory = cast(
            EpisodicMemory,
            await memory_manager.get_episodic_memory_instance(
                group_id=group_id,
                session_id=group_id,
                user_id=[speaker_a, speaker_b],
            ),
        )

        num_batch = 50
        kg_batch = []
        session_idx = 0
        num_msg = 0
        while True:
            session_idx += 1
            session_id = f"session_{session_idx}"

            if session_id not in conversation:
                break

            session = conversation[session_id]
            session_date_time = conversation[f"{session_id}_date_time"]

            context_messages: deque[str] = deque(maxlen=5)
            for message in session:
                speaker = message["speaker"]
                blip_caption = message.get("blip_caption")
                message_text = message["text"]

                context_messages.append(
                    f"[{session_date_time}] {speaker}: {message_text}"
                )
                
                id = uuid.uuid4()
                ts = datetime.now()

                await memory.add_memory_episode(
                    producer=speaker,
                    produced_for=speaker,
                    episode_content=message_text,
                    episode_type="default",
                    content_type=ContentType.STRING,
                    timestamp=ts,
                    metadata={
                        "source_timestamp": session_date_time,
                        "source_speaker": speaker,
                        "blip_caption": blip_caption,
                    },
                    uuid=id,
                )

                fmt = "%I:%M %p on %d %B, %Y"
                ts = datetime.strptime(session_date_time, fmt)
                kg_content = f"{speaker}: {message_text}"
                if blip_caption:
                    kg_content += f" [Image Caption: {blip_caption}]"
                kg_batch.append(Node(
                    uuid=id,
                    labels={"Episode"},
                    # Make timestamp different for each episode
                    properties={
                        "content": kg_content,
                        "timestamp": ts + timedelta(seconds=num_msg),
                        "session_id": memory.group_id(),
                    },
                ))
                num_msg += 1

                if len(kg_batch) >= num_batch:
                    t = time.perf_counter()
                    await kg.add_episode_bulk(kg_batch)
                    print(f"Added batch of {len(kg_batch)} episodes to KG in {(time.perf_counter() - t):.3f}s")
                    kg_batch = []
        if len(kg_batch) > 0:
            t = time.perf_counter()
            await kg.add_episode_bulk(kg_batch, True)
            print(f"Added final batch of {len(kg_batch)} episodes to KG in {(time.perf_counter() - t):.3f}s")
            kg_batch = []
        try:
            kg.print_ingest_perf_matrix()
        except Exception as e:
            print(f"Error printing KG ingest perf matrix: {e}")
        await memory.close()

    tasks = [
        process_conversation(idx, item, memory_manager)
        for idx, item in enumerate(locomo_data)
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
