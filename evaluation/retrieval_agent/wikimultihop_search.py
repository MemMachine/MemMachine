import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.retrieval_agent.wikimultihop_ingest import load_data  # noqa: E402
from evaluation.utils import agent_utils  # noqa: E402

# Citation: Luo et al. (2025), "Agent Lightning: Train ANY AI Agents with
# Reinforcement Learning", arXiv:2508.03680.
ANSWER_PROMPT = """You are asked to answer `{question}` using `{memories}` as the primary source when they contain sufficient evidence; otherwise use general world knowledge.

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

async def run_wiki(
    dpath: str | None = None,
    epath: str | None = None,
) -> tuple[str, dict[str, Any]]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--eval-result-path", required=True, help="Path to save evaluation results", default=None
    )
    parser.add_argument(
        "--length", type=int, default=500, help="Number of questions to search"
    )
    parser.add_argument(
        "--test-target", required=True, help="Testing with memmachine(bypass agent), retrieval_agent, or pure llm", choices=["memmachine", "retrieval_agent", "llm"]
    )

    args = parser.parse_args()

    print("Starting WikiMultiHop test...")
    print(f"Data path: {args.data_path}")
    print(f"Evaluation result path: {args.eval_result_path}")
    print(f"Length: {args.length}")
    print(f"Test target: {args.test_target}")

    data_path = args.data_path
    eval_result_path = args.eval_result_path

    if dpath:
        data_path = dpath
    if epath:
        eval_result_path = epath

    vector_graph_store = agent_utils.init_vector_graph_store(neo4j_uri="bolt://localhost:7687")
    memory, model, query_agent = await agent_utils.init_memmachine_params(
        vector_graph_store=vector_graph_store,
        session_id="group1", # Wikimultihop dataset does not have session concept
        model_name="gpt-5-mini",
        agent_name="ToolSelectAgent" if args.test_target == "retrieval_agent" else "MemMachineAgent",
    )

    contexts, questions, answers, types, supporting_facts = load_data(data_path=data_path, start_line=1, end_line=args.length, randomize="NONE")
    print(f"Loaded {len(questions)} questions, start querying...")

    tasks = []
    results: dict[str, Any] = {}
    attribute_matrix = agent_utils.init_attribute_matrix()
    full_content = "\n".join(contexts)
    num_processed = 0
    for q, a, t, f_list in zip(
        questions, answers, types, supporting_facts, strict=True
    ):
        t += 5 # Use locomo category 1-5, and wiki 6-9
        tasks.append(
            agent_utils.process_question(
                ANSWER_PROMPT,
                query_agent,
                memory,
                model,
                q,
                a,
                t,
                f_list,
                "",
                full_content=full_content if args.test_target == "llm" else None,
            )
        )

        if len(tasks) % 10 == 0 or (q == questions[-1]):
            responses = await asyncio.gather(*tasks)
            tasks = []
            agent_utils.update_results(responses, attribute_matrix, results)
            num_processed += len(responses)
            print(f"Completed searching {num_processed}/{len(questions)} questions...")

    agent_utils.update_final_attribute_matrix(
        "wiki",
        attribute_matrix,
        results,
    )
    return eval_result_path, results

async def main():
    eval_result_path, results = await run_wiki()
    with open(eval_result_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
