import argparse
import asyncio
import json
import os
from collections import defaultdict
from time import time

import numpy as np
from dotenv import load_dotenv
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from openai import AsyncOpenAI
from sklearn.metrics import f1_score
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def get_llm_evaluation(prompt, model="gpt-4o"):
    """
    Get LLM evaluation for a given prompt
    """
    try:
        # Replace with your preferred LLM API call
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        return "error"


def compute_bleu_score(reference, candidate):
    """
    Compute BLEU score between reference and candidate text
    """
    try:
        # Tokenize the texts (simple whitespace tokenization)
        reference_tokens = [reference.split()]
        candidate_tokens = candidate.split()

        # Apply smoothing for cases where there might be zero matches
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            reference_tokens, candidate_tokens, smoothing_function=smoothing
        )
        return score
    except Exception as e:
        print(f"Error computing BLEU score: {e}")
        return 0.0


def compute_iqr(data):
    """Compute the interquartile range (IQR) of a list of numbers."""
    q75, q25 = np.percentile(data, [75, 25])
    return q75 - q25


async def evaluate_responses(responses, llm_model="gpt-4o", exclude_abstention=True):
    """
    Evaluate responses using F1, BLEU, and LLM scores for each question type,
    while storing latency per question and token usage stats.
    """
    results = defaultdict(
        lambda: {
            "f1_scores": [],
            "bleu_scores": [],
            "llm_scores": [],
            "latencies": [],
            "tokens": [],
            "count": 0,
        }
    )

    binary_labels = []
    binary_predictions = []

    for item in tqdm_asyncio(responses, desc="Evaluating", unit="q"):
        qid = item["question_id"]
        abstention = item["abstention"]
        if exclude_abstention and abstention:
            continue
        task = item["question_type"]
        question = item["question"]
        correct_answer = item["answer"]
        model_response = item.get("response", "")

        prompt = get_anscheck_prompt(
            task, question, correct_answer, model_response, abstention
        )

        # Measure latency
        start_time = time()
        llm_response = await get_llm_evaluation(prompt, llm_model)
        print(llm_response)
        latency = time() - start_time

        llm_score = 1 if "yes" in llm_response.lower() else 0
        bleu_score = compute_bleu_score(correct_answer, model_response)

        results[task]["llm_scores"].append(llm_score)
        results[task]["bleu_scores"].append(bleu_score)
        results[task]["latencies"].append(latency)
        results[task]["f1_scores"].append(llm_score)
        results[task]["count"] += 1

        results[task].setdefault("question_llm_map", {})[qid] = llm_score

        binary_labels.append(1)  # adjust if you have true labels
        binary_predictions.append(llm_score)

    # Aggregate metrics
    final_results = {}
    for task, metrics in results.items():
        if metrics["count"] > 0:
            f1 = f1_score([1] * len(metrics["f1_scores"]), metrics["f1_scores"])
            avg_bleu = np.mean(metrics["bleu_scores"])
            avg_llm = np.mean(metrics["llm_scores"])
            avg_latency = np.mean(metrics["latencies"])

            # Token stats
            token_avg = np.mean(metrics["tokens"]) if metrics["tokens"] else None
            token_iqr = compute_iqr(metrics["tokens"]) if metrics["tokens"] else None

            final_results[task] = {
                "f1_score": float(f1),
                "bleu_score": float(avg_bleu),
                "llm_score": float(avg_llm),
                "avg_latency": float(avg_latency),
                "token_avg": float(token_avg) if token_avg is not None else None,
                "token_iqr": float(token_iqr) if token_iqr is not None else None,
                "count": metrics["count"],
                "f1_scores_detail": metrics["f1_scores"],
                "bleu_scores_detail": metrics["bleu_scores"],
                "llm_scores_detail": metrics["llm_scores"],
                "latencies_detail": metrics["latencies"],
                "tokens_detail": metrics["tokens"],
                "question_llm_map": metrics["question_llm_map"],
            }

    # Overall metrics
    all_tokens = [
        t for task_metrics in results.values() for t in task_metrics["tokens"]
    ]
    overall_f1 = f1_score(binary_labels, binary_predictions)
    overall_bleu = np.mean(
        [
            score
            for task_metrics in results.values()
            for score in task_metrics["bleu_scores"]
        ]
    )
    overall_llm = np.mean(binary_predictions)
    overall_latency = np.mean(
        [lat for task_metrics in results.values() for lat in task_metrics["latencies"]]
    )
    overall_token_avg = np.mean(all_tokens) if all_tokens else None
    overall_token_iqr = compute_iqr(all_tokens) if all_tokens else None

    final_results["overall"] = {
        "f1_score": float(overall_f1),
        "bleu_score": float(overall_bleu),
        "llm_score": float(overall_llm),
        "avg_latency": float(overall_latency),
        "token_avg": float(overall_token_avg)
        if overall_token_avg is not None
        else None,
        "token_iqr": float(overall_token_iqr)
        if overall_token_iqr is not None
        else None,
        "total_count": sum(metrics["count"] for metrics in results.values()),
    }

    return final_results


def load_dataset(file_path):
    """
    Load your dataset from JSON file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results, output_path):
    """
    Save evaluation results to JSON file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# Your existing function (included for completeness)
def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "temporal-reasoning":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "knowledge-update":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "single-session-preference":
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


async def main():
    """
    Main function to run the evaluation
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )

    args = parser.parse_args()

    data_path = args.data_path
    target_path = args.target_path

    # Load your dataset
    responses = load_dataset(data_path)

    print(f"Evaluating {len(responses)} questions...")

    # Evaluate responses
    results = await evaluate_responses(
        responses, llm_model="gpt-4o", exclude_abstention=False
    )
    # Save results
    save_results(results, output_path=target_path)


if __name__ == "__main__":
    asyncio.run(main())
