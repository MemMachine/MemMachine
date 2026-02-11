import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from memmachine.common.utils import async_with

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.utils import agent_utils  # noqa: E402

ANSWER_PROMPT = """
You are asked to answer a question based on your memories of a conversation.

<instructions>
1. Prioritize memories that answer the question directly. Be meticulous about recalling details.
2. When there may be multiple answers to the question, think hard to remember and list all possible answers. Do not become satisfied with just the first few answers you remember.
3. When asked about time intervals or to count items, do not rush to answer immediately. Instead, carefully enumerate the items or subtract the times using numbers.
4. Your memories are episodic, meaning that they consist of only your raw observations of what was said. You may need to reason about or guess what the memories imply in order to answer the question.
5. The question may contain typos or be based on the asker's own unreliable memories. Do your best to answer the question using the most relevant information in your memories.
6. Your memories may include small or large jumps in time or context. You are not confused by this. You just did not bother to remember everything in between.
7. Your memories are ordered from earliest to latest.
</instructions>

<memories>
{memories}
</memories>

Question: {question}
Your short response to the question without fluff (no more than a couple of sentences):
"""

def datetime_from_locomo_time(locomo_time_str: str) -> datetime:
    return datetime.strptime(locomo_time_str, "%I:%M %p on %d %B, %Y").replace(
        tzinfo=UTC
    )

async def run_locomo(  # noqa: C901
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
        "--test-target", required=True, help="Testing with memmachine(bypass agent), retrieval_agent, or pure llm", choices=["memmachine", "retrieval_agent", "llm"]
    )

    args = parser.parse_args()

    print("Starting locomo test...")
    print(f"Data path: {args.data_path}")
    print(f"Evaluation result path: {args.eval_result_path}")
    print(f"Test target: {args.test_target}")

    data_path = args.data_path
    eval_result_path = args.eval_result_path

    if dpath:
        data_path = dpath
    if epath:
        eval_result_path = epath

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

    results: dict[str, Any] = {}
    attribute_matrix = agent_utils.init_attribute_matrix()
    start_index = 0
    end_index = 20
    vector_graph_store = agent_utils.init_vector_graph_store(neo4j_uri="bolt://localhost:7687")
    for idx, item in enumerate(locomo_data):
        if idx < start_index:
            continue

        if idx > end_index:
            break

        if "conversation" not in item:
            continue

        conversation = item["conversation"]
        qa_list = item["qa"]
        filtered_list = []
        for qa in qa_list:
            if qa["category"] == 5 or qa["category"] == "5":
                continue
            filtered_list.append(qa)

        print(f"Processing questions for group {idx}...")

        group_id = f"group_{idx}"

        evidence_to_text = {}
        full_content = []
        session_idx = 0
        while True:
            session_idx += 1
            session_id = f"session_{session_idx}"
            if session_id not in conversation:
                break

            session = conversation[session_id]
            session_datetime = datetime_from_locomo_time(
                conversation[f"{session_id}_date_time"]
            )
            for message in session:
                dia_id = message["dia_id"]
                text = message["text"]
                evidence_to_text[dia_id] = text
                speaker = message["speaker"]
                full_content.append(f"[{session_datetime}] {speaker}: {text}")

        memory, model, query_agent = await agent_utils.init_memmachine_params(
            vector_graph_store=vector_graph_store,
            session_id=group_id,
            model_name="gpt-5-mini",
            agent_name="ToolSelectAgent" if args.test_target == "retrieval_agent" else "MemMachineAgent",
        )

        async def respond_question(qa, full_content):
            question = qa["question"]
            answer = qa.get("answer", "")
            category = qa["category"]
            evidence = qa["evidence"]

            adversarial_answer = qa.get("adversarial_answer", "")

            stringified_evidence = []
            for ev in evidence:
                if "," in ev:
                    ids = ev.split(",")
                    for evidence_id in ids:
                        evidence_str = evidence_to_text.get(evidence_id.strip(), "")
                        stringified_evidence.append(evidence_str)
                elif ";" in ev:
                    ids = ev.split(";")
                    for evidence_id in ids:
                        evidence_str = evidence_to_text.get(evidence_id.strip(), "")
                        stringified_evidence.append(evidence_str)
                else:
                    evidence_str = evidence_to_text.get(ev.strip(), "")
                    stringified_evidence.append(evidence_str)

            full_content_str = "\n".join(full_content)

            question_response = await agent_utils.process_question(
                ANSWER_PROMPT,
                query_agent,
                memory,
                model,
                question,
                answer,
                category,
                stringified_evidence,
                adversarial_answer,
                20,
                "gpt-5-mini",
                full_content_str if args.test_target == "llm" else None,
            )
            return question_response

        semaphore = asyncio.Semaphore(30)
        response_tasks = [
            async_with(
                semaphore,
                respond_question(qa, full_content),
            )
            for qa in filtered_list
        ]

        responses = await asyncio.gather(*response_tasks)
        agent_utils.update_results(responses, attribute_matrix, results)

    agent_utils.update_final_attribute_matrix(
        "locomo",
        attribute_matrix,
        results,
    )
    return eval_result_path, results

async def main():
    eval_result_path, results = await run_locomo()
    with open(eval_result_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
