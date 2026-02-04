import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
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


def _split_chunks(text: str, max_chars: int = 3000) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0
    text_len = len(normalized)
    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            split_at = normalized.rfind(" ", start, end)
            if split_at > start + (max_chars // 2):
                end = split_at
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _collect_turn_contents(sample: dict[str, Any]) -> list[str]:
    turns: list[str] = []
    for session in sample.get("haystack_sessions", []) or []:
        for turn in session or []:
            content = str(turn.get("content", "")).strip()
            if content:
                turns.extend(_split_chunks(content))
    return turns


def _collect_supporting_facts(sample: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    for session in sample.get("haystack_sessions", []) or []:
        for turn in session or []:
            if turn.get("has_answer"):
                content = str(turn.get("content", "")).strip()
                if content:
                    facts.append(content)
    return facts


def _set_safe_embedder_request_limits(memory: Any) -> None:
    long_term_memory = getattr(memory, "long_term_memory", None)
    declarative_memory = (
        getattr(long_term_memory, "declarative_memory", None)
        if long_term_memory is not None
        else None
    )
    embedder = getattr(declarative_memory, "_embedder", None)
    if embedder is None:
        return
    if hasattr(embedder, "max_total_input_length_per_request"):
        # Keep cluster size below model token ceiling for embedding requests.
        embedder.max_total_input_length_per_request = 30000


async def longmemeval_ingest(dataset: list[dict[str, Any]], session_id: str):
    t1 = datetime.now(UTC)
    added_content = 0
    per_batch = 1000

    vector_graph_store = agent_utils.init_vector_graph_store(
        neo4j_uri="bolt://localhost:7687"
    )

    memory, _, _ = await agent_utils.init_memmachine_params(
        vector_graph_store=vector_graph_store,
        session_id=session_id,
    )
    _set_safe_embedder_request_limits(memory)

    all_content: list[str] = []
    for sample in dataset:
        all_content.extend(_collect_turn_contents(sample))

    episodes: list[Episode] = []
    for content in all_content:
        added_content += 1
        ts = t1 + timedelta(seconds=added_content)
        episodes.append(
            Episode(
                uid=str(uuid4()),
                content=content,
                session_key=session_id,
                created_at=ts,
                producer_id="user",
                producer_role="user",
            )
        )

        if added_content % per_batch == 0 or content == all_content[-1]:
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
        f"Completed LongMemEval ingestion, added {len(dataset)} questions, {added_content} episodes."
    )


async def longmemeval_search(
    dataset: list[dict[str, Any]],
    session_id: str,
    eval_result_path: str | None = None,
    agent_name: str = "ToolSelectAgent",
    pure_llm: bool = False,
):
    tasks = []
    attribute_matrix = agent_utils.init_attribute_matrix()
    responses: list[tuple[str, dict[str, Any]]] = []
    num_searched = 0

    vector_graph_store = agent_utils.init_vector_graph_store(
        neo4j_uri="bolt://localhost:7687"
    )
    memory, model, query_agent = await agent_utils.init_memmachine_params(
        vector_graph_store=vector_graph_store,
        model_name="gpt-5-mini",
        session_id=session_id,
        agent_name=agent_name,
    )
    _set_safe_embedder_request_limits(memory)

    for sample in dataset:
        question = str(sample.get("question", "")).strip()
        answer = str(sample.get("answer", "")).strip()
        if not question:
            continue

        supporting_facts = _collect_supporting_facts(sample)
        all_content = _collect_turn_contents(sample)
        full_content = "\n".join(all_content)

        tasks.append(
            agent_utils.process_question(
                answer_prompt=ANSWER_PROMPT,
                query_agent=query_agent,
                memory=memory,
                model=model,
                question=question,
                answer=answer,
                category=str(sample.get("question_type", "unknown")),
                supporting_facts=supporting_facts,
                search_limit=20,
                model_name="gpt-5-mini",
                full_content=full_content if pure_llm else None,
                extra_attributes={
                    "question_id": sample.get("question_id", ""),
                    "split": sample.get("split", ""),
                },
            )
        )

        if len(tasks) % 30 == 0 or sample == dataset[-1]:
            responses.extend(await asyncio.gather(*tasks))
            num_searched += len(tasks)
            print(
                f"Completed LongMemEval searching {num_searched}/{len(dataset)} questions..."
            )
            tasks = []

    results: dict[str, Any] = {}
    agent_utils.update_results(responses, attribute_matrix, results)
    agent_utils.update_final_attribute_matrix(
        "longmemeval",
        attribute_matrix,
        results,
    )

    if eval_result_path is not None:
        with open(eval_result_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)


def load_longmemeval_dataset(length: int, split: str) -> list[dict[str, Any]]:
    split_file = split if split.endswith(".json") else f"{split}.json"

    records: list[dict[str, Any]] | None = None

    # Primary path: use datasets for consistency with existing benchmark scripts.
    try:
        from datasets import load_dataset

        dataset = load_dataset("xiaowu0162/longmemeval-cleaned", split=split)
        num_rows = min(length, len(dataset))
        records = dataset.select(range(num_rows)).to_list()
    except Exception as err:
        # Fallback path: download split JSON directly when datasets loader
        # hits schema incompatibilities.
        print(f"datasets loader failed ({type(err).__name__}), using JSON fallback...")
        from huggingface_hub import hf_hub_download

        data_path = hf_hub_download(
            repo_id="xiaowu0162/longmemeval-cleaned",
            repo_type="dataset",
            filename=split_file,
        )
        with open(data_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
        if not isinstance(raw_data, list):
            raise TypeError(
                f"Expected list data in {split_file}, got {type(raw_data).__name__}."
            ) from err
        records = raw_data[:length]

    normalized_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue

        normalized_record = dict(record)
        normalized_record["question"] = str(normalized_record.get("question", ""))
        normalized_record["answer"] = str(normalized_record.get("answer", ""))
        normalized_record.setdefault("question_type", "unknown")
        normalized_record.setdefault("haystack_sessions", [])
        normalized_record["split"] = split
        normalized_records.append(normalized_record)

    return normalized_records


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
        help="Number of records to run",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--split-name",
        required=False,
        help="Dataset split name from xiaowu0162/longmemeval-cleaned",
        default="longmemeval_s_cleaned",
    )
    parser.add_argument(
        "--test-target",
        required=True,
        help="Testing with memmachine(bypass agent), retrieval_agent, or pure llm",
        choices=["memmachine", "retrieval_agent", "llm"],
    )
    parser.add_argument(
        "--session-id",
        required=False,
        help="Session id used for both ingestion and retrieval",
        default="longmemeval_group",
    )
    args = parser.parse_args()

    dataset = load_longmemeval_dataset(args.length, args.split_name)

    if args.run_type == "ingest":
        await longmemeval_ingest(dataset, args.session_id)
    elif args.run_type == "search":
        print("Starting LongMemEval test...")
        print(f"Evaluation result path: {args.eval_result_path}")
        print(f"Length: {args.length}")
        print(f"Dataset split: {args.split_name}")
        print(f"Test target: {args.test_target}")

        agent_name = (
            "MemMachineAgent" if args.test_target == "memmachine" else "ToolSelectAgent"
        )
        await longmemeval_search(
            dataset,
            args.session_id,
            args.eval_result_path,
            agent_name,
            args.test_target == "llm",
        )
    else:
        raise ValueError(
            f"Unknown run type: {args.run_type}, please use 'ingest' or 'search'."
        )


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
