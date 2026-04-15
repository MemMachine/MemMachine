"""BEAM benchmark evaluation script with rubric-based scoring."""

import argparse
import concurrent.futures
import json
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Callable

import json_repair
from scipy.stats import kendalltau
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.retrieval_agent.cli_utils import positive_int  # noqa: E402
from evaluation.retrieval_agent.llm_judge import (  # noqa: E402
    create_judge_fn,
)


# BEAM unified LLM judge prompt
UNIFIED_LLM_JUDGE_PROMPT = """You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): {question}
- RUBRIC CRITERION (what to check): {criterion}
- RESPONSE TO EVALUATE: {response}

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., "$68,000", "68k", "68,000 USD", or "sixty-eight thousand dollars"). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., "itemized list", "no citations", "one sentence").
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.

- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.

- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)

3. **Check Compliance**:
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}}

NOTE: Only output the JSON object, without any explanation before or after it.
"""


def evaluate_rubric_criterion(
    question: str,
    criterion: str,
    response: str,
    call_fn: Callable[[str], str],
) -> tuple[float, str]:
    """Evaluate a single rubric criterion using LLM judge.

    Args:
        question: The probing question.
        criterion: Single rubric criterion to evaluate.
        response: The model's response.
        call_fn: Callable for LLM judge.

    Returns:
        Tuple of (score, reasoning) where score is 0.0, 0.5, or 1.0.
    """
    prompt = UNIFIED_LLM_JUDGE_PROMPT.format(
        question=question,
        criterion=criterion,
        response=response,
    )

    raw = call_fn(prompt)
    try:
        result = json_repair.loads(raw)
        score = float(result.get("score", 0.0))
        # Clamp score to valid range
        score = max(0.0, min(1.0, score))
        reasoning = result.get("reason", result.get("reasoning", ""))
    except Exception:
        score = 0.0
        reasoning = "Failed to parse LLM response"

    return score, reasoning


def extract_facts_from_response(response: str, question: str) -> list[str]:
    """Extract facts from response for event ordering evaluation.

    Args:
        response: The model's response.
        question: The probing question (for context).

    Returns:
        List of extracted facts/statements.
    """
    # Split by newlines and filter empty lines
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    return lines


def llm_equivalence(first_snippet: str, second_snippet: str, call_fn: Callable[[str], str]) -> bool:
    """Check if two snippets describe the same event/fact using LLM.

    Args:
        first_snippet: First fact/snippet.
        second_snippet: Second fact/snippet.
        call_fn: Callable for LLM judge.

    Returns:
        True if the two snippets describe the same event/fact, False otherwise.
    """
    system_msg = "You are a binary classifier. If the TWO snippets describe the SAME event/fact, reply YES. Otherwise reply NO. No extra words. DO NOT provide any explanation."

    user_msg = f"First snippet: {first_snippet}\nSecond snippet: {second_snippet}"

    prompt = f"{system_msg}\n\n{user_msg}"

    response = call_fn(prompt).lower()

    return "yes" in response


def align_with_llm(reference: list[str], system: list[str], call_fn: Callable[[str], str]) -> tuple[list[str], list[str]]:
    """Align system facts with reference facts using LLM equivalence.

    Args:
        reference: Reference ordered list of facts.
        system: System predicted ordered list of facts.
        call_fn: Callable for LLM judge.

    Returns:
        Tuple of (reference_canon, system_canon) after alignment.
    """
    used = set()
    system_out = []

    for s in system:
        matched_index = None
        for index, r in enumerate(reference):
            if index in used:
                continue

            if llm_equivalence(first_snippet=r, second_snippet=s, call_fn=call_fn):
                matched_index = index
                break

        if matched_index is not None:
            system_out.append(reference[matched_index])
            used.add(matched_index)
        else:
            system_out.append(s)

    return reference, system_out


def compute_kendall_tau_normalized(reference_list: list[str], system_list: list[str], call_fn: Callable[[str], str] | None = None) -> dict:
    """Compute event ordering score using Kendall tau-b normalized.

    Args:
        reference_list: Reference ordered list of events/facts.
        system_list: System predicted ordered list of events/facts.
        call_fn: Optional callable for LLM alignment. If provided, uses LLM-based alignment.

    Returns:
        Dictionary with precision, recall, f1, tau_norm, and final_score.
    """
    if not reference_list or not system_list:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tau_norm": 0.0,
            "final_score": 0.0,
        }

    # Apply LLM-based alignment if call_fn is provided
    if call_fn is not None:
        reference_list, system_list = align_with_llm(reference_list, system_list, call_fn)
    union = list(dict.fromkeys(reference_list + system_list))
    tie_rank = len(union) + 1

    def to_rank(seq: list[str]) -> list[int]:
        """Convert sequence to rank list."""
        rank_map = {item: i + 1 for i, item in enumerate(seq)}
        return [rank_map.get(u, tie_rank) for u in union]

    # Compute precision/recall/F1 based on set overlap
    ref_set = set(reference_list)
    sys_set = set(system_list)

    tp = len(ref_set & sys_set)
    fp = len(sys_set - ref_set)
    fn = len(ref_set - sys_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute Kendall tau-b normalized
    ref_ranks = to_rank(reference_list)
    sys_ranks = to_rank(system_list)

    tau_b, _ = kendalltau(ref_ranks, sys_ranks, variant="b", method="auto")
    tau_norm = (tau_b + 1) / 2 if tau_b is not None else 0.0

    # Final score is tau_norm * f1
    final_score = tau_norm * f1

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tau_norm": round(tau_norm, 4),
        "final_score": round(final_score, 4),
    }


def evaluate_with_rubric(
    question: str,
    response: str,
    rubric: list,
    call_fn: Callable[[str], str],
    category: str = "",
) -> dict:
    """Evaluate a response against rubric criteria.

    Args:
        question: The probing question.
        response: The model's response.
        rubric: List of evaluation criteria.
        call_fn: Callable for LLM judge.
        category: BEAM category (for special handling of event_ordering).

    Returns:
        Dictionary with rubric_score, criterion_scores, and evaluation details.
    """
    if not rubric:
        return {
            "rubric_score": 0.0,
            "num_criteria": 0,
            "criterion_scores": [],
            "criterion_reasonings": [],
            "event_ordering": None,
        }

    criterion_scores = []
    criterion_reasonings = []
    total_score = 0.0

    for criterion in rubric:
        score, reasoning = evaluate_rubric_criterion(
            question, criterion, response, call_fn
        )
        criterion_scores.append(score)
        criterion_reasonings.append(reasoning)
        total_score += score

    rubric_score = total_score / len(rubric) if rubric else 0.0

    # Special handling for event_ordering category
    event_ordering_result = None
    if category == "event_ordering":
        # Extract facts from rubric (reference) and response (system)
        reference_list = extract_facts_from_response("\n".join(rubric), question)
        system_list = extract_facts_from_response(response, question)

        event_ordering_result = compute_kendall_tau_normalized(reference_list, system_list, call_fn)

    return {
        "rubric_score": round(rubric_score, 4),
        "num_criteria": len(rubric),
        "criterion_scores": [round(s, 4) for s in criterion_scores],
        "criterion_reasonings": criterion_reasonings,
        "event_ordering": event_ordering_result,
    }


def process_beam_sample(
    group_key: str,
    item: dict,
    call_fn: Callable[[str], str],
) -> tuple[str, dict | None]:
    """Process a single BEAM sample.

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
        return group_key, {
            "question": question,
            "ideal_answer": ideal_answer,
            "response": response,
            "category": category,
            "rubric_score": None,
            "note": "No rubric provided",
        }

    # Perform rubric-based evaluation
    eval_result = evaluate_with_rubric(question, response, rubric, call_fn, category)

    rubric_score = eval_result["rubric_score"]

    # For event_ordering, use tau_norm as the primary metric
    if category == "event_ordering" and eval_result["event_ordering"]:
        llm_score = eval_result["event_ordering"]["tau_norm"]
    else:
        llm_score = rubric_score

    res = {
        "question": question,
        "ideal_answer": ideal_answer,
        "response": response,
        "category": category,
        "rubric": rubric,
        "rubric_score": rubric_score,
        "llm_score": llm_score,  # For compatibility with generate_scores.py
        "num_criteria": eval_result["num_criteria"],
        "criterion_scores": eval_result["criterion_scores"],
        "criterion_reasonings": eval_result["criterion_reasonings"],
        "llm_judge_responses": [
            {"score": s, "reason": r}
            for s, r in zip(eval_result["criterion_scores"], eval_result["criterion_reasonings"])
        ],
    }

    # Add event_ordering specific metrics if applicable
    if eval_result["event_ordering"]:
        res["event_ordering"] = eval_result["event_ordering"]

    # Add extra BEAM-specific attributes if present
    for key in [
        "difficulty",
        "abstention_type",
        "contradiction_type",
        "ordering_type",
        "ideal_response",
        "preference_being_tested",
        "why_unanswerable",
    ]:
        if key in item:
            res[key] = item[key]

    return group_key, res


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for BEAM evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate BEAM benchmark results"
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

    print("Starting BEAM evaluation ...")
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

    # Save results once after all evaluations complete
    with open(args.target_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Print category-level summary
    print("\n" + "=" * 60)
    print("BEAM Evaluation Summary")
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
