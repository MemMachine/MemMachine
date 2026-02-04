# ruff: noqa: N999

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
from memmachine.common.episode_store import Episode  # noqa: E402

# Citation: Luo et al. (2025), "Agent Lightning: Train ANY AI Agents with
# Reinforcement Learning", arXiv:2508.03680.
ANSWER_PROMPT = """You are asked to answer `{question}` using `{memories}` as the only source of knowledge.

<instructions>
1. Normalize inputs before deciding anything:
   - Treat `{memories}` as possibly empty.
   - Normalize entity spellings/case/ordinals/titles and common aliases (e.g., “10Th” → “10th”; honorific variants).
   - If `{question}` is malformed, underspecified, or missing key constraints, ask exactly one concise clarifying question instead of answering.

2. Choose the evidence basis using this strict priority:
   (a) **Memory-explicit**: Use when `{memories}` contain at least one explicit statement that answers the question or provides all necessary facts.
   (b) **Memory-determined inference**: Use when explicit memory facts, taken together, *fully determine* the answer unambiguously (show minimal reasoning).
   (c) **Open-domain fallback**: Use general world knowledge when memories are empty/irrelevant/too vague OR do not fully determine the answer.

3. Uncertainty rule:
   - Do **not** say “unknown/not mentioned” if open-domain knowledge can reasonably answer.
   - If neither memories nor general knowledge allow a confident answer, say “I don’t know” (optionally add a brief reason).

4. Ambiguity handling:
   - If multiple plausible entities/answers remain after normalization, provide the top candidates and note the ambiguity briefly.
   - If multiple valid answers are genuinely possible, enumerate them (comma-separated or short bullets).

5. Computation and counting:
   - For counts or time intervals, compute explicitly (brief enumeration or numeric subtraction) to avoid mistakes.

6. Output requirements (concise, auditable):
   - Provide the **Answer** only, without additional commentary.
   - Keep the total response to **max 2 sentences**, except when enumeration/computation is required; then use **up to 4 short lines** (bullets allowed) while staying as brief as possible.
</instructions>

<memories>
{memories}
</memories>

Question: {question}
"""


async def hotpotqa_ingest(dataset: list[dict[str, any]]):
    t1 = datetime.now(UTC)
    added_content = 0
    per_batch = 1000

    vector_graph_store = agent_utils.init_vector_graph_store(
        neo4j_uri="bolt://localhost:7687"
    )

    # Notice that the index of items must align between ingestion and search
    memory, _, _ = await agent_utils.init_memmachine_params(
        vector_graph_store=vector_graph_store,
        session_id="hotpotqa_group",
    )

    all_content = []
    for data in dataset:
        context = data["context"]
        titles = context["title"]
        sentences = context["sentences"]
        for title, sent_list in zip(titles, sentences, strict=True):
            for sent in sent_list:
                all_content.append(f"{title}: {sent}")

    # Fully randomize contents
    random.shuffle(all_content)
    episodes = []
    for sent in all_content:
        added_content += 1
        ts = t1 + timedelta(minutes=added_content)
        episodes.append(
            Episode(
                uid=str(uuid4()),
                content=sent,
                session_key="hotpotqa_group",
                created_at=ts,
                producer_id="user",
                producer_role="user",
            )
        )
        if added_content % per_batch == 0 or sent == all_content[-1]:
            print(f"Adding batch of {len(episodes)} episodes...")
            t = time.perf_counter()
            await memory.add_memory_episodes(episodes=episodes)
            print(
                f"Gathered and added {len(episodes)} episodes in {(time.perf_counter() - t):.3f}s"
            )
            print(f"Total added episodes: {added_content}")
            print(f"Total episodes processed: {added_content}/{len(all_content)}")
            episodes = []
    print(
        f"Completed HotpotQA ingestion, added {len(dataset)} questions, {added_content} episodes."
    )


async def hotpotqa_search(
    dataset: list[dict[str, any]],
    eval_result_path: str | None = None,
    agent_name: str = "ToolSelectAgent",
    pure_llm: bool = False,
):
    tasks = []
    attribute_matrix = agent_utils.init_attribute_matrix()
    responses: list[tuple[int, dict[str, any]]] = []
    num_searched = 0
    vector_graph_store = agent_utils.init_vector_graph_store(
        neo4j_uri="bolt://localhost:7687"
    )
    memory, model, query_agent = await agent_utils.init_memmachine_params(
        vector_graph_store=vector_graph_store,
        model_name="gpt-5-mini",
        session_id="hotpotqa_group",
        agent_name=agent_name,
    )

    for data in dataset:
        context = data["context"]
        titles = context["title"]
        sentences = context["sentences"]  # List[List[str]]

        # Get supporting facts in string
        supporting_facts = []
        fact_index_dict = data["supporting_facts"]
        for title, sent_id in zip(
            fact_index_dict["title"], fact_index_dict["sent_id"], strict=True
        ):
            sent = sentences[titles.index(title)][sent_id]
            supporting_facts.append(sent)

        full_content = [sent for sent_list in sentences for sent in sent_list]
        full_content_str = "\n".join(full_content)

        tasks.append(
            agent_utils.process_question(
                answer_prompt=ANSWER_PROMPT,
                query_agent=query_agent,
                memory=memory,
                model=model,
                question=data["question"],
                answer=data["answer"],
                category=(data["type"]),
                supporting_facts=supporting_facts,
                search_limit=20,
                model_name="gpt-5-mini",
                full_content=full_content_str if pure_llm else None,
                extra_attributes={"level": data["level"]},
            )
        )

        if len(tasks) % 30 == 0 or data == dataset[-1]:
            responses.extend(await asyncio.gather(*tasks))
            num_searched += len(tasks)
            print(
                f"Completed HotpotQA searching {num_searched}/{len(dataset)} questions..."
            )
            tasks = []

    results: dict[str, any] = {}
    agent_utils.update_results(responses, attribute_matrix, results)
    agent_utils.update_final_attribute_matrix(
        "hotpotqa",
        attribute_matrix,
        results,
    )

    if eval_result_path is not None:
        with open(eval_result_path, "w") as f:
            json.dump(results, f, indent=4)


def load_hotpotqa_dataset(length: int, split: str) -> list[dict[str, any]]:
    from datasets import load_dataset

    dataset = load_dataset("hotpot_qa", "distractor", split=split)

    # To JSON format
    data = dataset.select(range(length)).to_list()
    return data


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-result-path",
        required=False,
        help="Path to save evaluation results",
        default=None,
    )
    parser.add_argument(
        "--run-type",
        required=False,
        help="Type of run: ingest or search",
        default="search",
    )
    parser.add_argument(
        "--length",
        required=False,
        help="Number of records to run on EACH n-needle dataset(total 3x length are testing)",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--split-name",
        required=False,
        help="Dataset split name: train(90.4k questions, 20%% easy, 63%% medium, 17%% hard), validation(7.41k question, all hard)",
        default="validation",
    )
    parser.add_argument(
        "--test-target",
        required=True,
        help="Testing with memmachine(bypass agent), retrieval_agent, or pure llm",
        choices=["memmachine", "retrieval_agent", "llm"],
    )
    args = parser.parse_args()

    dataset = load_hotpotqa_dataset(args.length, args.split_name)

    if args.run_type == "ingest":
        await hotpotqa_ingest(dataset)
    elif args.run_type == "search":
        print("Starting HotpotQA test...")
        print(f"Evaluation result path: {args.eval_result_path}")
        print(f"Length: {args.length}")
        print(f"Dataset split: {args.split_name}")
        print(f"Test target: {args.test_target}")

        agent_name = (
            "MemMachineAgent" if args.test_target == "memmachine" else "ToolSelectAgent"
        )
        await hotpotqa_search(
            dataset, args.eval_result_path, agent_name, args.test_target == "llm"
        )
    else:
        raise ValueError(
            f"Unknown run type: {args.run_type}, please use 'ingest' or 'search'."
        )


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
