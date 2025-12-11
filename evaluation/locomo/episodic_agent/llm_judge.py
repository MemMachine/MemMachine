# This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/metrics/llm_judge.py).
# It is modified to remove dependency on the Mem0 library and formatted.

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
logger = logging.getLogger(__name__)

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


def evaluate_llm_judge(question: str, gold_answer: str, generated_answer: str) -> int:
    """Evaluate the generated answer against the gold answer using an LLM judge."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": ACCURACY_PROMPT.format(
                    question=question,
                    gold_answer=gold_answer,
                    generated_answer=generated_answer,
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    label = json.loads(response.choices[0].message.content)["label"]
    return 1 if label == "CORRECT" else 0


def main() -> None:
    """Main function to evaluate RAG results using LLM judge."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Evaluate RAG results using LLM judge")
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("results/default_run_v4_k30_new_graph.json"),
        help="Path to the input dataset file",
    )

    args = parser.parse_args()

    dataset_path = args.input_file
    output_path = Path("results") / f"llm_judge_{dataset_path.name}"

    with dataset_path.open("r") as dataset_file:
        data: dict[str, list[dict[str, Any]]] = json.load(dataset_file)

    llm_labels: defaultdict[str, list[int]] = defaultdict(list)
    results: defaultdict[int, list[dict[str, str]]] = defaultdict(list)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    index = 0
    for conversations in data.values():
        for x in conversations:
            question = x["question"]
            gold_answer = x["answer"]
            generated_answer = x["response"]
            category = x["category"]

            # Skip category 5
            if int(category) == 5:
                continue

            # Evaluate the answer
            label = evaluate_llm_judge(question, gold_answer, generated_answer)
            llm_labels[category].append(label)

            # Store the results
            results[index].append(
                {
                    "question": question,
                    "gt_answer": gold_answer,
                    "response": generated_answer,
                    "category": category,
                    "llm_label": label,
                },
            )

            # Save intermediate results
            with output_path.open("w") as output_file:
                json.dump(results, output_file, indent=4)

            # Print current accuracy for all categories
            logger.info("All categories accuracy:")
            for cat, category_results in llm_labels.items():
                if category_results:
                    logger.info(
                        "  Category %s: %.4f (%s/%s)",
                        cat,
                        np.mean(category_results),
                        sum(category_results),
                        len(category_results),
                    )
            logger.info("------------------------------------------")
        index += 1

    # Save final results
    with output_path.open("w") as output_file:
        json.dump(results, output_file, indent=4)

    # Print final summary
    logger.info("PATH: %s", dataset_path)
    logger.info("------------------------------------------")
    for k, v in llm_labels.items():
        logger.info("%s %s", k, np.mean(v))


if __name__ == "__main__":
    main()
