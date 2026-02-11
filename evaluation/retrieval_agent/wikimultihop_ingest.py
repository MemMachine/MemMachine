import argparse
import asyncio
import json
import random
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.utils import agent_utils  # noqa: E402
from memmachine.episodic_memory.declarative_memory import (  # noqa: E402
    ContentType,
    Episode,
)


def load_data(  # noqa: C901
    data_path: str,
    start_line: int = 1,
    end_line: int = 100,
    randomize: str = "KEYWORD",
):
    print(f"Loading data from {data_path}")
    print(f"Loading data from line {start_line} to {end_line}, randomize={randomize}")
    contexts = []
    questions = []
    answers = []
    types = []
    supporting_facts = []
    i = 1
    with open(data_path, "r", encoding="utf-8") as f:
        keyword_list = []
        for line in f:
            if i < start_line:
                i += 1
                continue
            if i > end_line:
                break
            i += 1

            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["context"] = json.loads(obj["context"])
            cur_ketword_seg = []
            key_to_sentences = {}
            for key, sentences in obj["context"]:
                key_to_sentences[key] = sentences
                for s in sentences:
                    c = f"{key}: {s}"
                    contexts.append(c)
                    cur_ketword_seg.append(c)
            keyword_list.append(cur_ketword_seg)
            questions.append(obj["question"])
            answers.append(obj["answer"])

            type_map = {
                "compositional": 1,
                "comparison": 2,
                "inference": 3,
                "bridge_comparison": 4,
            }
            types.append(type_map[obj["type"]])

            cur_facts = json.loads(obj["supporting_facts"])
            fact_sents = []
            for fact in cur_facts:
                key = fact[0]
                sentence_idx = int(fact[1])
                sent = f"{key}: {key_to_sentences[key][sentence_idx]}"
                fact_sents.append(sent)
            supporting_facts.append(fact_sents)

        # Randomize on sentence level
        if randomize == "SENTENCE":
            random.shuffle(contexts)
        # Randomize on keyword level.
        # Wikimultihop dataset gives context for each question in the format of:
        # [[keyword1, [sent1, sent2]], [keyword2, [sent3, sent4], ...]]
        # Each keyword list is kept in order, but the order of keyword lists are shuffled.
        elif randomize == "KEYWORD":
            random.shuffle(keyword_list)
            contexts = []
            for seg in keyword_list:
                contexts.extend(seg)

    return contexts, questions, answers, types, supporting_facts


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")
    parser.add_argument(
        "--length", type=int, default=500, help="Number of records to ingest"
    )

    args = parser.parse_args()

    data_path = args.data_path

    vector_graph_store = agent_utils.init_vector_graph_store(
        neo4j_uri="bolt://localhost:7687"
    )
    memory, _, _ = await agent_utils.init_memmachine_params(
        vector_graph_store=vector_graph_store,
        session_id="group1",  # Wikimultihop dataset does not have session concept
    )

    contexts, _, _, _, _ = load_data(
        data_path=data_path, start_line=1, end_line=args.length, randomize="SENTENCE"
    )
    print("Loaded", len(contexts), "contexts, start ingestion...")

    num_batch = 1000
    episodes = []
    added_contexts = set()
    t1 = datetime.now(UTC)
    episodes = []
    for c in contexts:
        # Wikimultihop dataset may have duplicate sentences, skip them
        if c in added_contexts:
            continue

        added_contexts.add(c)

        ts = t1 + timedelta(seconds=len(added_contexts))

        source = c.split(":")[0]
        episodes.append(
            Episode(
                uid=str(uuid4()),
                timestamp=ts,
                source=source,
                content_type=ContentType.TEXT,
                content=c,
            )
        )

        if len(added_contexts) % num_batch == 0 or (c == contexts[-1]):
            t = time.perf_counter()
            await memory.add_episodes(episodes=episodes)
            print(
                f"Gathered and added {len(episodes)} episodes in {(time.perf_counter() - t):.3f}s"
            )
            episodes = []

            print(f"Total added episodes: {len(added_contexts)}")

    print(f"Completed WIKI-Multihop ingestion, added {len(added_contexts)} episodes.")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
