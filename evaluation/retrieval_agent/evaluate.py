# This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/evals.py).
# It is modified to only report LLM judge scores.

import argparse
import concurrent.futures
import json
import sys
import threading
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.retrieval_agent.llm_judge import (  # noqa: E402
    create_judge_fn,
    evaluate_llm_judge,
)


def process_sample(group_key: str, item: dict, call_fn):
    question = str(item["question"])
    locomo_answer = str(item["golden_answer"])
    response = str(item["model_answer"])
    category = str(item["category"])

    # Skip category 5
    if category == "5":
        return group_key, None

    llm_score = evaluate_llm_judge(question, locomo_answer, response, call_fn)

    res = {
        "question": question,
        "answer": locomo_answer,
        "response": response,
        "category": category,
        "llm_score": llm_score,
    }
    for key, val in item.items():
        if key not in [
            "question",
            "golden_answer",
            "model_answer",
            "category",
        ]:
            if type(val) is float:
                val = round(val, 3)
            res[key] = val

    return group_key, res


def main():
    parser = argparse.ArgumentParser(description="Evaluate results")
    parser.add_argument(
        "--data-path",
        type=str,
        default="results/rag_results_500_k1.json",
        help="Path to the input dataset file",
    )
    parser.add_argument(
        "--target-path",
        type=str,
        default="evaluation_metrics.json",
        help="Path to save the evaluation results",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=30,
        help="Maximum number of worker threads",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to configuration.yml (used to select the judge LLM)",
    )

    args = parser.parse_args()

    call_fn = create_judge_fn(args.config_path)

    with open(args.data_path, "r") as f:
        data = json.load(f)

    results = defaultdict(list)
    results_lock = threading.Lock()
    sample_tasks: list[tuple[str, dict]] = [
        (group_key, item) for group_key, items in data.items() for item in items
    ]

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = [
            executor.submit(process_sample, group_key, item, call_fn)
            for group_key, item in sample_tasks
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            group_key, sample_result = future.result()
            if sample_result is None:
                continue
            with results_lock:
                results[group_key].append(sample_result)

            with open(args.target_path, "w") as f:
                json.dump(results, f, indent=4)

    print(f"Results saved to {args.target_path}")


if __name__ == "__main__":
    main()
