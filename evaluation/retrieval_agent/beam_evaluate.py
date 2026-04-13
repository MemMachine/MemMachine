import argparse
import concurrent.futures
import json
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Callable

import json_repair
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.retrieval_agent.cli_utils import positive_int  # noqa: E402
from evaluation.retrieval_agent.llm_judge import (  # noqa: E402
    create_judge_fn,
)


# Rubric evaluation prompt - evaluates each criterion individually
RUBRIC_EVALUATION_PROMPT = """You are tasked with evaluating whether a response meets a specific criterion.

Question: {question}
Criterion: {criterion}
Response: {response}

Evaluate if the response adequately addresses this criterion.
Return a JSON object with:
- "score": 1 if the response meets the criterion, 0 otherwise
- "reasoning": brief explanation of your evaluation

Example: {{"score": 1, "reasoning": "The response clearly addresses the criterion by..."}}
"""


def evaluate_rubric_criterion(
    question: str,
    criterion: str,
    response: str,
    call_fn: Callable[[str], str],
) -> tuple[int, str]:
    """Evaluate a single rubric criterion.

    Args:
        question: The probing question.
        criterion: Single rubric criterion to evaluate.
        response: The model's response.
        call_fn: Callable for LLM judge.

    Returns:
        Tuple of (score, reasoning).
    """
    prompt = RUBRIC_EVALUATION_PROMPT.format(
        question=question,
        criterion=criterion,
        response=response,
    )

    raw = call_fn(prompt)
    try:
        result = json_repair.loads(raw)
        score = int(result.get("score", 0))
        reasoning = result.get("reasoning", "")
    except Exception:
        score = 0
        reasoning = "Failed to parse LLM response"

    return score, reasoning


def evaluate_with_rubric(
    question: str,
    response: str,
    rubric: list,
    call_fn: Callable[[str], str],
) -> dict:
    """Evaluate a response against rubric criteria.

    Args:
        question: The probing question.
        response: The model's response.
        rubric: List of evaluation criteria.
        call_fn: Callable for LLM judge.

    Returns:
        Dictionary with rubric_score, individual scores, and reasoning.
    """
    if not rubric:
        return {
            "rubric_score": 0,
            "num_criteria": 0,
            "criterion_scores": [],
            "criterion_reasonings": [],
        }

    criterion_scores = []
    criterion_reasonings = []
    total_score = 0

    for criterion in rubric:
        score, reasoning = evaluate_rubric_criterion(
            question, criterion, response, call_fn
        )
        criterion_scores.append(score)
        criterion_reasonings.append(reasoning)
        total_score += score

    rubric_score = total_score / len(rubric) if rubric else 0

    return {
        "rubric_score": rubric_score,
        "num_criteria": len(rubric),
        "criterion_scores": criterion_scores,
        "criterion_reasonings": criterion_reasonings,
    }


def process_beam_sample(
    group_key: str,
    item: dict,
    call_fn: Callable[[str], str],
) -> tuple[str, dict | None]:
    """Process a single BEAM sample with rubric evaluation.

    Args:
        group_key: Category key (e.g., "abstention", "contradiction_resolution").
        item: Sample data from BEAM results.
        call_fn: Callable for LLM judge.

    Returns:
        Tuple of (group_key, result_dict) or (group_key, None) if skipped.
    """
    question = str(item.get("question", ""))
    ideal_answer = str(item.get("golden_answer", item.get("ideal_answer", "")))
    response = str(item.get("model_answer", item.get("response", "")))
    category = str(item.get("category", ""))
    rubric = item.get("rubric", [])

    # Skip if no rubric (fallback to standard evaluation)
    if not rubric:
        # For samples without rubric, use basic accuracy check
        return group_key, {
            "question": question,
            "ideal_answer": ideal_answer,
            "response": response,
            "category": category,
            "rubric_score": None,
            "note": "No rubric provided",
        }

    # Perform rubric-based evaluation
    rubric_result = evaluate_with_rubric(question, response, rubric, call_fn)

    # Build result dictionary
    rubric_score = round(rubric_result["rubric_score"], 3)
    res = {
        "question": question,
        "ideal_answer": ideal_answer,
        "response": response,
        "category": category,
        "rubric": rubric,
        "rubric_score": rubric_score,
        "llm_score": rubric_score,  # For compatibility with generate_scores.py
        "num_criteria": rubric_result["num_criteria"],
        "criterion_scores": rubric_result["criterion_scores"],
        "criterion_reasonings": rubric_result["criterion_reasonings"],
    }

    # Add extra BEAM-specific attributes if present
    for key in [
        "difficulty",
        "abstention_type",
        "contradiction_type",
        "ordering_type",
        "ideal_response",
    ]:
        if key in item:
            res[key] = item[key]

    return group_key, res


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for BEAM evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate BEAM benchmark results with rubric-based scoring"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the BEAM results JSON file",
    )
    parser.add_argument(
        "--target-path",
        type=str,
        default="beam_evaluation_metrics.json",
        help="Path to save the evaluation results",
    )
    parser.add_argument(
        "--max-workers",
        type=positive_int,
        default=30,
        help="Maximum number of worker threads (default: 30)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to configuration.yml (used to select the judge LLM)",
    )
    return parser


def main():
    """Main entry point for BEAM evaluation."""
    args = build_parser().parse_args()

    print("Starting BEAM rubric-based evaluation...")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.target_path}")
    print(f"Max workers: {args.max_workers}")

    call_fn = create_judge_fn(args.config_path)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = defaultdict(list)
    results_lock = threading.Lock()
    sample_tasks = [
        (group_key, item) for group_key, items in data.items() for item in items
    ]

    # Track category-level metrics
    category_metrics = defaultdict(lambda: {"total": 0, "score_sum": 0.0})

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = [
            executor.submit(process_beam_sample, group_key, item, call_fn)
            for group_key, item in sample_tasks
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Evaluating",
        ):
            group_key, sample_result = future.result()
            if sample_result is None:
                continue

            with results_lock:
                results[group_key].append(sample_result)

                # Update category metrics
                if sample_result.get("rubric_score") is not None:
                    category_metrics[group_key]["total"] += 1
                    category_metrics[group_key]["score_sum"] += sample_result[
                        "rubric_score"
                    ]

            # Save intermediate results
            with open(args.target_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    # Print category-level summary
    print("\n" + "=" * 60)
    print("BEAM Rubric Evaluation Summary")
    print("=" * 60)

    for category, metrics in sorted(category_metrics.items()):
        if metrics["total"] > 0:
            avg_score = metrics["score_sum"] / metrics["total"]
            print(
                f"  {category}: {avg_score:.4f} "
                f"({metrics['total']} samples)"
            )

    # Calculate overall average
    total_samples = sum(m["total"] for m in category_metrics.values())
    total_score_sum = sum(m["score_sum"] for m in category_metrics.values())
    if total_samples > 0:
        overall_avg = total_score_sum / total_samples
        print("-" * 60)
        print(f"  Overall: {overall_avg:.4f} ({total_samples} samples)")

    print("=" * 60)
    print(f"\nResults saved to {args.target_path}")


if __name__ == "__main__":
    main()
