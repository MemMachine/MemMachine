# This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/evals.py).
# It is modified to only report LLM judge scores.

import argparse
import concurrent.futures
import json
import logging
import threading
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from llm_judge import evaluate_llm_judge
from tqdm import tqdm

logger = logging.getLogger(__name__)

load_dotenv()


def process_item(
    item_data: tuple[str, list[dict[str, str]]],
) -> defaultdict[str, list[dict[str, str]]]:
    k, v = item_data
    local_results = defaultdict(list)

    for item in tqdm(v, desc=f"Processing {k} sample"):
        question = str(item["question"])
        locomo_answer = str(item["answer"])
        response = str(item["response"])
        category = str(item["category"])

        # Skip category 5
        if category == "5":
            continue

        llm_score = evaluate_llm_judge(question, locomo_answer, response)

        local_results[k].append(
            {
                "question": question,
                "answer": locomo_answer,
                "response": response,
                "category": category,
                "llm_score": llm_score,
            },
        )

    return local_results


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate results")
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("results/rag_results_500_k1.json"),
        help="Path to the input dataset file",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("evaluation_metrics.json"),
        help="Path to save the evaluation results",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Maximum number of worker threads",
    )

    args = parser.parse_args()

    with args.input_file.open("r") as input_file:
        data: dict[str, list[dict[str, str]]] = json.load(input_file)

    results: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    results_lock = threading.Lock()

    # Use ThreadPoolExecutor with specified workers
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers,
    ) as executor:
        futures = [
            executor.submit(process_item, item_data) for item_data in data.items()
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
        ):
            local_results = future.result()
            with results_lock:
                for k, items in local_results.items():
                    results[k].extend(items)

            # Save results to JSON file
            with args.output_file.open("w") as output_file:
                json.dump(results, output_file, indent=4)

    logger.info("Results saved to %s", args.output_file)


if __name__ == "__main__":
    main()
