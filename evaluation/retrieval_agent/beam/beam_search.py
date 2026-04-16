import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.retrieval_agent.cli_utils import positive_int  # noqa: E402
from evaluation.utils import agent_utils  # noqa: E402

# Default context window size for LLM (in tokens, approximate)
# This is used for tail truncation when test_target is "llm"
DEFAULT_CONTEXT_WINDOW = 128000  # 128K tokens

# Approximate characters per token (English text)
CHARS_PER_TOKEN = 4


def truncate_history_to_context_window(
    full_content: str,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> str:
    """Truncate conversation history to fit within context window.

    Uses tail truncation to keep the most recent conversations.

    Args:
        full_content: Full conversation history.
        context_window: Context window size in tokens.

    Returns:
        Truncated conversation history.
    """
    # Convert tokens to approximate characters
    max_chars = context_window * CHARS_PER_TOKEN

    if len(full_content) <= max_chars:
        return full_content

    # Keep the tail (most recent conversations)
    truncated = full_content[-max_chars:]

    # Try to start from a new line to avoid cutting in the middle
    first_newline = truncated.find("\n")
    if first_newline != -1:
        truncated = truncated[first_newline + 1 :]

    return truncated


# BEAM benchmark answer prompt
BEAM_ANSWER_PROMPT = """You are asked to answer the following question based on your conversation history stored in memory.

<instructions>
1. Use the retrieved memories as your primary source of knowledge.
2. If memories contain sufficient evidence to answer the question, use them.
3. If memories are empty or irrelevant, you may use general world knowledge.
4. Be precise and specific in your answers.
5. If the question cannot be answered from available information, state that clearly.

<memories>
{memories}
</memories>

Question: {question}

Answer:
"""


def load_beam_questions(questions_path: str) -> dict[str, list[dict]]:
    """Load BEAM probing questions from JSON file.

    Args:
        questions_path: Path to probing_questions.json file.

    Returns:
        Dictionary mapping question categories to lists of questions.
    """
    print(f"Loading BEAM probing questions from {questions_path}")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    total_questions = sum(len(v) for v in questions_data.values())
    print(f"Loaded {total_questions} questions across {len(questions_data)} categories")
    return questions_data


def load_beam_chat_data(chat_path: str) -> list[dict]:
    """Load BEAM chat data from chat.json file.

    Args:
        chat_path: Path to chat.json file.

    Returns:
        List of chat batches.
    """
    print(f"Loading BEAM chat data from {chat_path}")
    with open(chat_path, "r", encoding="utf-8") as f:
        chat_data = json.load(f)
    print(f"Loaded {len(chat_data)} batches from BEAM chat")
    return chat_data


def build_full_content_from_chat(chat_data: list[dict]) -> str:
    """Build full conversation content from chat data.

    Args:
        chat_data: List of chat batches.

    Returns:
        Full conversation history as a string.
    """
    lines = []
    for batch in chat_data:
        batch_num = batch.get("batch_number", 0)
        turns = batch.get("turns", [])

        for turn_idx, turn in enumerate(turns):
            for msg in turn:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                time_anchor = msg.get("time_anchor", "")

                if content:
                    time_info = f" [{time_anchor}]" if time_anchor else ""
                    lines.append(
                        f"[{batch_num}:{turn_idx}] {role}:{time_info} {content}"
                    )

    return "\n".join(lines)


async def beam_search(
    chat_data_path: str,
    questions_path: str,
    config_path: str,
    session_id: str,
    eval_result_path: str | None = None,
    agent_name: str = "ToolSelectAgent",
    concurrency: int = 10,
    test_target: str = "retrieval_agent",
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> tuple[str, dict[str, Any]]:
    """Search BEAM probing questions against MemMachine memory.

    Args:
        chat_data_path: Path to chat.json file (for llm target history).
        questions_path: Path to probing_questions.json file.
        config_path: Path to configuration.yml.
        session_id: Session ID for memory lookup.
        eval_result_path: Path to save evaluation results.
        agent_name: Name of the retrieval agent to use.
        concurrency: Maximum concurrent search requests.
        test_target: Target type ("memmachine", "retrieval_agent", "llm").
        context_window: Context window size in tokens (for llm target).

    Returns:
        Tuple of (eval_result_path, results dictionary).
    """
    questions_data = load_beam_questions(questions_path)

    # Load chat data and build full content for llm target
    full_content = ""
    if test_target == "llm":
        chat_data = load_beam_chat_data(chat_data_path)
        full_content = build_full_content_from_chat(chat_data)
        print(f"Built full content: {len(full_content)} characters")
        full_content = truncate_history_to_context_window(full_content, context_window)
        print(f"After truncation: {len(full_content)} characters")

    resource_manager = agent_utils.load_eval_config(config_path)
    memory, answer_model, query_agent = await agent_utils.init_memmachine_params(
        resource_manager=resource_manager,
        session_id=session_id,
        agent_name=agent_name,
    )

    tasks = []
    results: dict[str, Any] = {}
    attribute_matrix = agent_utils.init_attribute_matrix()
    num_processed = 0

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_process(category: str, question: dict):
        async with semaphore:
            question_text = question.get("question", "")
            ideal_answer = question.get("ideal_response", question.get("answer", ""))
            difficulty = question.get("difficulty", "unknown")

            # Extract extra attributes for BEAM-specific metadata
            extra_attrs = {
                "category": category,
                "difficulty": difficulty,
            }
            for key in [
                "ideal_response",
                "answer",
                "abstention_type",
                "contradiction_type",
                "ordering_type",
                "rubric",
            ]:
                if key in question:
                    extra_attrs[key] = question[key]

            return await agent_utils.process_question(
                BEAM_ANSWER_PROMPT,
                query_agent,
                memory,
                answer_model,
                question_text,
                ideal_answer,
                category,
                [],  # BEAM doesn't provide explicit supporting facts for search
                "",  # No adversarial answer
                search_limit=20,
                full_content=full_content if test_target == "llm" else None,
                extra_attributes=extra_attrs,
            )

    # Process questions by category
    for category, questions in questions_data.items():
        for question in questions:
            tasks.append(bounded_process(category, question))

        # Process in batches to respect concurrency
        if len(tasks) >= concurrency:
            responses = await asyncio.gather(*tasks)
            tasks = []
            agent_utils.update_results(responses, attribute_matrix, results)
            num_processed += len(responses)
            print(f"Processed {num_processed} questions...")

    # Flush any remaining tasks after all categories/questions are processed
    if tasks:
        responses = await asyncio.gather(*tasks)
        agent_utils.update_results(responses, attribute_matrix, results)
        num_processed += len(responses)
        print(f"Processed {num_processed} questions...")

    agent_utils.update_final_attribute_matrix(
        "beam",
        attribute_matrix,
        results,
    )

    if eval_result_path:
        with open(eval_result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {eval_result_path}")

    return eval_result_path, results


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for BEAM search."""
    parser = argparse.ArgumentParser(
        description="Search BEAM benchmark probing questions against MemMachine"
    )
    parser.add_argument(
        "--chat-data-path",
        required=False,
        help="Path to chat.json file (required for llm test target)",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to probing_questions.json file",
    )
    parser.add_argument(
        "--eval-result-path",
        required=True,
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to configuration.yml",
    )
    parser.add_argument(
        "--session-id",
        default="beam_default",
        help="Session ID for memory lookup (default: beam_default)",
    )
    parser.add_argument(
        "--test-target",
        required=True,
        choices=["memmachine", "retrieval_agent", "llm"],
        help="Testing with memmachine, retrieval_agent, or pure llm",
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        default=10,
        help="Maximum concurrent search requests (default: 10)",
    )
    parser.add_argument(
        "--context-window",
        type=positive_int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=f"Context window size in tokens for llm target (default: {DEFAULT_CONTEXT_WINDOW})",
    )
    return parser


async def main():
    """Main entry point for BEAM search."""
    args = build_parser().parse_args()

    # Validate chat-data-path for llm target
    if args.test_target == "llm" and not args.chat_data_path:
        print("Error: --chat-data-path is required when test-target is 'llm'")
        sys.exit(1)

    print("Starting BEAM benchmark search...")
    print(f"Chat data path: {args.chat_data_path}")
    print(f"Questions path: {args.data_path}")
    print(f"Evaluation result path: {args.eval_result_path}")
    print(f"Config path: {args.config_path}")
    print(f"Session ID: {args.session_id}")
    print(f"Test target: {args.test_target}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Context window: {args.context_window} tokens")

    # Determine agent name based on test target
    agent_name = (
        "ToolSelectAgent"
        if args.test_target == "retrieval_agent"
        else "MemMachineAgent"
    )

    await beam_search(
        chat_data_path=args.chat_data_path,
        questions_path=args.data_path,
        config_path=args.config_path,
        session_id=args.session_id,
        eval_result_path=args.eval_result_path,
        agent_name=agent_name,
        concurrency=args.concurrency,
        test_target=args.test_target,
        context_window=args.context_window,
    )


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
