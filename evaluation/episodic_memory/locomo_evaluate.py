# This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/evals.py).
# It is modified to only report LLM judge scores and to be simpler.

import argparse
import json

from dotenv import load_dotenv
from llm_judge import evaluate_llm_judge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the source data file",
    )
    parser.add_argument(
        "--target-path",
        required=True,
        help="Path to the target data file",
    )
    args = parser.parse_args()
    data_path = args.data_path
    target_path = args.target_path

    # Load environment variables
    load_dotenv()

    with open(data_path, "r") as f:
        test_data = json.load(f)
    results = {}
    for key, value in test_data.items():
        if key == "5":
            continue
        local_result = []
        for item in value:
            question = item["question"]
            locomo_answer = f"{item['locomo_answer']}"
            response = f"{item['model_answer']}"
            llm_score = evaluate_llm_judge(question, locomo_answer, response)
            local_result.append(
                {
                    "question": question,
                    "answer": locomo_answer,
                    "response": response,
                    "category": key,
                    "llm_score": llm_score,
                },
            )
        results[key] = local_result
    with open(target_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
