import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.utils import agent_utils  # noqa: E402

# This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/prompts.py).
# It is modified to work with MemMachine.
ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories to answer a question.

    # CONTEXT:
    You have access to memories from a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the speakers.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    <MEMORIES>

    {conversation_memories}

    </MEMORIES>

    Question: {question}

    Answer:
    """


def format_memory(episodes, summary) -> str:
    episode_context = (
        "<LONG TERM MEMORY EPISODES>\n"
        + "\n".join(
            [
                f"[{episode.metadata['source_timestamp']}] {episode.metadata['source_speaker']}: {episode.content}{f' [ATTACHED: {episode.metadata["blip_caption"]}]' if episode.metadata.get('blip_caption') else ''}"
                for episode in episodes
            ],
        )
        + "\n</LONG TERM MEMORY EPISODES>"
    )
    summary_context = (
        f"<WORKING MEMORY SUMMARY>\n{summary}\n</WORKING MEMORY SUMMARY>"
        if summary
        else ""
    )
    return episode_context + "\n" + summary_context


async def process_question(
    resource_manager,
    group_id,
    user,
    question,
    answer,
    category,
    evidence,
    adversarial_answer,
):
    memory_start = time.time()
    memory, answer_model, _ = await agent_utils.init_memmachine_params(
        resource_manager=resource_manager,
        session_id=group_id,
    )

    query_response = await memory.query_memory(query=question, limit=30, expand_context=3)

    if query_response is None:
        long_term_episodes = []
        short_term_episodes = []
        summaries = []
    else:
        long_term_episodes = query_response.long_term_memory.episodes
        short_term_episodes = query_response.short_term_memory.episodes
        summaries = query_response.short_term_memory.episode_summary

    episodes = long_term_episodes + short_term_episodes
    summary = summaries[0] if summaries else ""
    memory_end = time.time()

    await memory.close()

    formatted_context = format_memory(episodes, summary)
    prompt = ANSWER_PROMPT.format(
        conversation_memories=formatted_context,
        question=question,
    )

    llm_start = time.time()
    rsp_text, _ = await answer_model.generate_response(user_prompt=prompt)
    llm_end = time.time()

    print(
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Response: {rsp_text}\n"
        f"Memory retrieval time: {memory_end - memory_start:.2f} seconds\n"
        f"LLM response time: {llm_end - llm_start:.2f} seconds\n"
        f"MEMORIES START\n{formatted_context}MEMORIES END\n",
    )
    return {
        "question": question,
        "locomo_answer": answer,
        "model_answer": rsp_text,
        "category": category,
        "evidence": evidence,
        "adversarial_answer": adversarial_answer,
        "conversation_memories": formatted_context,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the source data file",
    )
    parser.add_argument(
        "--config-path",
        default="locomo_config.yaml",
        help="Path to configuration.yml",
    )
    parser.add_argument(
        "--target-path",
        required=True,
        help="Path to the target data file",
    )

    args = parser.parse_args()

    data_path = args.data_path
    target_path = args.target_path

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

    resource_manager = agent_utils.load_eval_config(args.config_path)

    results: dict[str, Any] = {}
    for idx, item in enumerate(locomo_data):
        if "conversation" not in item:
            continue

        conversation = item["conversation"]
        user = conversation["speaker_a"]

        qa_list = item["qa"]

        print(f"Processing questions for group {idx}...")

        group_id = f"group_{idx}"

        async def respond_question(qa):
            question = qa["question"]
            answer = qa.get("answer", "")
            category = qa["category"]
            evidence = qa["evidence"]

            adversarial_answer = qa.get("adversarial_answer", "")

            question_response = await process_question(
                resource_manager,
                group_id,
                user,
                question,
                answer,
                category,
                evidence,
                adversarial_answer,
            )
            return (
                category,
                question_response,
            )

        responses = []
        for qa in qa_list:
            responses.append(await respond_question(qa))

        for category, response in responses:
            category_result = results.get(category, [])
            category_result.append(response)
            results[category] = category_result

    with open(target_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
