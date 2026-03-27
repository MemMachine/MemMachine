# This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/metrics/llm_judge.py).
# It is modified to remove dependency on the Mem0 library and formatted.

import argparse
import json
import os
from collections import defaultdict

import json_repair
import numpy as np
import openai

ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

DEFAULT_JUDGE_MODEL = "gpt-5-mini"

_openai_client = None


def _get_openai_client() -> openai.OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY must be set when no eval_llm client is provided."
            )
        _openai_client = openai.OpenAI(api_key=api_key)
    return _openai_client


def evaluate_llm_judge(
    question: str,
    gold_answer: str,
    generated_answer: str,
    model_name: str = DEFAULT_JUDGE_MODEL,
    eval_llm: object | None = None,
) -> int:
    """Evaluate a generated answer against the gold answer using an LLM judge.

    When *eval_llm* (a
    :class:`~evaluation.retrieval_skill.benchmark_config.LLMClient`) is
    provided it is used for generation; otherwise falls back to the OpenAI
    Responses API via ``OPENAI_API_KEY``.

    Args:
        question: The question being evaluated.
        gold_answer: The ground-truth answer.
        generated_answer: The model-produced answer.
        model_name: OpenAI model to use when *eval_llm* is ``None``.
        eval_llm: Optional :class:`LLMClient` from benchmark config.

    Returns:
        1 if the answer is CORRECT, 0 if WRONG.
    """
    prompt = ACCURACY_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )
    if eval_llm is not None:
        result = eval_llm.generate(prompt, json_mode=True)
        raw = result.text or ""
    else:
        client = _get_openai_client()
        rsp = client.responses.create(
            model=model_name,
            input=prompt,
            text={"format": {"type": "json_object"}},
        )
        raw = rsp.output_text or ""
    label = json_repair.loads(raw)["label"]
    return 1 if label == "CORRECT" else 0


def main():
    """Main function to evaluate RAG results using LLM judge."""
    parser = argparse.ArgumentParser(description="Evaluate RAG results using LLM judge")
    parser.add_argument(
        "--input_file",
        type=str,
        default="results/default_run_v4_k30_new_graph.json",
        help="Path to the input dataset file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"OpenAI model name for LLM judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to benchmark_config.yml for evaluation model",
    )

    args = parser.parse_args()

    eval_llm = None
    if args.config:
        from evaluation.retrieval_skill.benchmark_config import (
            LLMClient,
            load_benchmark_config,
        )

        cfg = load_benchmark_config(args.config)
        eval_llm = LLMClient(cfg.evaluation_model)

    dataset_path = args.input_file
    output_path = f"results/llm_judge_{dataset_path.split('/')[-1]}"

    with open(dataset_path, "r") as f:
        data = json.load(f)

    LLM_JUDGE = defaultdict(list)
    RESULTS = defaultdict(list)

    index = 0
    for k, v in data.items():
        for x in v:
            question = x["question"]
            gold_answer = x["answer"]
            generated_answer = x["response"]
            category = x["category"]

            if int(category) == 5:
                continue

            label = evaluate_llm_judge(
                question,
                gold_answer,
                generated_answer,
                args.model,
                eval_llm=eval_llm,
            )
            LLM_JUDGE[category].append(label)

            RESULTS[index].append(
                {
                    "question": question,
                    "gt_answer": gold_answer,
                    "response": generated_answer,
                    "category": category,
                    "llm_label": label,
                }
            )

            with open(output_path, "w") as f:
                json.dump(RESULTS, f, indent=4)

            print("All categories accuracy:")
            for cat, results in LLM_JUDGE.items():
                if results:
                    print(
                        f"  Category {cat}: {np.mean(results):.4f} "
                        f"({sum(results)}/{len(results)})"
                    )
            print("------------------------------------------")
        index += 1

    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=4)

    print("PATH: ", dataset_path)
    print("------------------------------------------")
    for k, v in LLM_JUDGE.items():
        print(k, np.mean(v))


if __name__ == "__main__":
    main()
