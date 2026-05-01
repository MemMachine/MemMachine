# This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/metrics/llm_judge.py).
# It is modified to remove dependency on the Mem0 library and formatted.

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Callable

import json_repair
import numpy as np

logger = logging.getLogger(__name__)

_MAX_JUDGE_ATTEMPTS = 2

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


def create_judge_fn(config_path: str) -> Callable[[str], str]:
    """Build a synchronous callable that sends a prompt to the configured LLM.

    Supports providers: ``openai-responses``, ``openai-chat-completions``,
    and ``amazon-bedrock``.

    Args:
        config_path: Path to configuration.yml.

    Returns:
        A callable ``fn(prompt: str) -> str`` that returns the raw text reply.
    """
    from memmachine_server.common.configuration import Configuration

    config = Configuration.load_yml_file(config_path)
    lms = config.resources.language_models

    # Use judge_llm_model if set, otherwise fall back to llm_model
    llm_id = config.retrieval_agent.judge_llm_model or config.retrieval_agent.llm_model
    if not llm_id:
        raise ValueError("retrieval_agent.llm_model is not set in configuration.yml")

    if llm_id in lms.openai_responses_language_model_confs:
        from openai import OpenAI

        conf = lms.openai_responses_language_model_confs[llm_id]
        client = OpenAI(
            api_key=conf.api_key.get_secret_value(),
            base_url=conf.base_url,
        )
        model_name = conf.model

        def _call_responses(prompt: str) -> str:
            resp = client.responses.create(
                model=model_name,
                input=prompt,
                text={"format": {"type": "json_object"}},
            )
            return resp.output_text or ""

        return _call_responses

    if llm_id in lms.openai_chat_completions_language_model_confs:
        from openai import OpenAI

        conf = lms.openai_chat_completions_language_model_confs[llm_id]
        client = OpenAI(
            api_key=conf.api_key.get_secret_value(),
            base_url=conf.base_url,
        )
        model_name = conf.model

        def _call_chat(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content

        return _call_chat

    if llm_id in lms.amazon_bedrock_language_model_confs:
        import boto3

        conf = lms.amazon_bedrock_language_model_confs[llm_id]
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=conf.region,
            aws_access_key_id=(
                conf.aws_access_key_id.get_secret_value()
                if conf.aws_access_key_id
                else None
            ),
            aws_secret_access_key=(
                conf.aws_secret_access_key.get_secret_value()
                if conf.aws_secret_access_key
                else None
            ),
        )
        model_id = conf.model_id

        def _call_bedrock(prompt: str) -> str:
            resp = bedrock_client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
            )
            return resp["output"]["message"]["content"][0]["text"]

        return _call_bedrock

    raise ValueError(
        f"Language model '{llm_id}' not found in configuration.yml under "
        "resources.language_models. Check that the ID matches one of the defined models."
    )


def evaluate_llm_judge(
    question: str,
    gold_answer: str,
    generated_answer: str,
    call_fn: Callable[[str], str],
) -> int:
    """Evaluate a generated answer against the gold answer using an LLM judge.

    Args:
        question: The question being evaluated.
        gold_answer: The ground-truth answer.
        generated_answer: The model-produced answer.
        call_fn: A synchronous callable returned by :func:`create_judge_fn`.

    Returns:
        1 if the answer is CORRECT, 0 if WRONG.
    """
    prompt = ACCURACY_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )
    for attempt in range(1, _MAX_JUDGE_ATTEMPTS + 1):
        raw = call_fn(prompt)
        label: str | None = None
        try:
            result = json_repair.loads(raw)
            raw_label = result.get("label") if isinstance(result, dict) else None
            if isinstance(raw_label, str):
                normalized = raw_label.strip().upper()
                if normalized in {"CORRECT", "WRONG"}:
                    label = normalized
        except Exception:
            label = None
        if label is not None:
            return 1 if label == "CORRECT" else 0
        if attempt < _MAX_JUDGE_ATTEMPTS:
            logger.warning(
                "LLM judge missing or invalid 'label' on attempt %d/%d, retrying",
                attempt,
                _MAX_JUDGE_ATTEMPTS,
            )
    logger.error(
        "LLM judge failed to return a valid 'label' after %d attempts; defaulting to WRONG",
        _MAX_JUDGE_ATTEMPTS,
    )
    return 0


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
        "--config-path",
        type=str,
        required=True,
        help="Path to configuration.yml",
    )

    args = parser.parse_args()

    call_fn = create_judge_fn(args.config_path)
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

            label = evaluate_llm_judge(question, gold_answer, generated_answer, call_fn)
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
