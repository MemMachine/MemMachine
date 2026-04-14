import argparse
import json
import re
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package is required.")
    print("Install with: pip install datasets")
    sys.exit(1)

from tqdm import tqdm


def convert_chats_to_json(data: list) -> list:
    """Convert BEAM chat data from pickle format to JSON.

    Args:
        data: List of chat batches in pickle format.

    Returns:
        List of chat batches in JSON format.
    """
    json_object = []

    for index, batch in enumerate(data):
        batch_number = index + 1
        turns = []
        single_turn = []

        for message in batch:
            if "question_type" in message.keys() and message["question_type"] == "main_question" and single_turn:
                turns.append(single_turn)
                single_turn = []
                single_turn.append(message)
            else:
                single_turn.append(message)

        if single_turn:
            turns.append(single_turn)

        json_object.append({
            "batch_number": batch_number,
            "turns": turns
        })

    return json_object


def convert_user_messages_to_json(data: list) -> list:
    """Convert BEAM user messages from pickle format to JSON.

    Args:
        data: List of user message batches in pickle format.

    Returns:
        List of user message batches in JSON format.
    """
    json_object = []

    for batch_index, batch in enumerate(data, start=1):
        time_anchor = batch.get("time_anchor", "")
        messages = batch.get("messages", [[]])[0] if batch.get("messages") else []

        batch_messages = []
        for message in messages:
            if "->->" not in message:
                continue
            batch_messages.append({
                "role": "user",
                "content": message.strip()
            })

        json_object.append({
            "batch": batch_index,
            "time_anchor": time_anchor,
            "messages": batch_messages
        })

    return json_object


def parse_plans(plan_text: str) -> list:
    """Parse conversation plans into batches.

    Args:
        plan_text: Raw plan text with BATCH headers.

    Returns:
        List of plan strings.
    """
    raw_batches = re.split(r"(BATCH \d+ PLAN)", plan_text)
    plans = []
    for i in range(1, len(raw_batches), 2):
        header = raw_batches[i].strip()
        content = raw_batches[i + 1].strip()
        plans.append(f"{header}\n{content}")
    return plans


def get_dataset_name(size: str) -> str:
    """Get the Hugging Face dataset name for a given size.

    Args:
        size: Dataset size (e.g., "100K", "500K", "1M").

    Returns:
        Hugging Face dataset name.
    """
    return "Mohammadta/BEAM"


def download_and_convert(size: str, output_dir: Path):
    """Download and convert BEAM dataset for a specific size.

    Args:
        size: Dataset size (e.g., "100K", "500K", "1M", "10M").
        output_dir: Output directory path.
    """
    dataset_name = get_dataset_name(size)
    print(f"Downloading BEAM dataset ({size}) from {dataset_name}...")

    # Load dataset
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("You may need to accept the dataset license at:")
        print(f"  https://huggingface.co/datasets/{dataset_name}")
        sys.exit(1)

    # Create output directory
    size_dir = output_dir / size
    size_dir.mkdir(parents=True, exist_ok=True)

    # Get conversations based on size
    conversations = dataset[size]

    num_conversations = len(conversations)
    print(f"Found {num_conversations} conversations")

    for idx, conversation in tqdm(enumerate(conversations), total=num_conversations, desc=f"Processing {size}"):
        conversation_id = conversation["conversation_id"]
        conv_dir = size_dir / conversation_id
        conv_dir.mkdir(parents=True, exist_ok=True)

        # Extract conversation data
        chat = conversation.get("chat", [])
        probing_questions = conversation.get("probing_questions", "{}")
        labels = conversation.get("narratives", "")
        main_spec = conversation.get("user_profile", {}).get("user_info", "")
        relationships = conversation.get("user_profile", {}).get("user_relationships", "")
        topic = conversation.get("conversation_seed", {})
        user_messages = conversation.get("user_questions", [])

        # Convert and save chat.json
        json_chat = convert_chats_to_json(chat)
        chat_path = conv_dir / "chat.json"
        with open(chat_path, "w", encoding="utf-8") as f:
            json.dump(json_chat, f, ensure_ascii=False, indent=4)

        # Parse probing_questions (string representation of dict)
        try:
            import ast
            probing_questions_data = ast.literal_eval(probing_questions)
        except (ValueError, SyntaxError):
            try:
                probing_questions_data = json.loads(probing_questions)
            except json.JSONDecodeError:
                probing_questions_data = {}

        # Save probing_questions.json
        pq_dir = conv_dir / "probing_questions"
        pq_dir.mkdir(parents=True, exist_ok=True)
        pq_path = pq_dir / "probing_questions.json"
        with open(pq_path, "w", encoding="utf-8") as f:
            json.dump(probing_questions_data, f, ensure_ascii=False, indent=4)

        # Save text files
        with open(conv_dir / "labels.txt", "w", encoding="utf-8") as f:
            f.write(labels)

        with open(conv_dir / "main_spec.txt", "w", encoding="utf-8") as f:
            f.write(main_spec)

        with open(conv_dir / "relationships.txt", "w", encoding="utf-8") as f:
            f.write(relationships)

        with open(conv_dir / "topic.json", "w", encoding="utf-8") as f:
            json.dump(topic, f, ensure_ascii=False, indent=4)

        # Save user_messages.json
        json_user_messages = convert_user_messages_to_json(user_messages)
        with open(conv_dir / "user_messages.json", "w", encoding="utf-8") as f:
            json.dump(json_user_messages, f, ensure_ascii=False, indent=4)
        print(f"  Created: {conv_dir}")

    print(f"\nCompleted: {num_conversations} conversations saved to {size_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert BEAM dataset for MemMachine evaluation"
    )
    parser.add_argument(
        "--size",
        nargs="+",
        required=True,
        choices=["100K", "500K", "1M"],
        help="Dataset size(s) to download (e.g., 100K, 500K, 1M)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./beam"),
        help="Output directory (default: ./beam)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BEAM Dataset Downloader for MemMachine")
    print("=" * 60)
    print(f"Sizes: {', '.join(args.size)}")
    print(f"Output: {args.output.absolute()}")
    print("=" * 60)

    # Download and convert each size
    for size in args.size:
        print(f"\n{'=' * 60}")
        print(f"Processing size: {size}")
        print(f"{'=' * 60}\n")

        download_and_convert(size, args.output)

    print("\n" + "=" * 60)
    print("All downloads completed!")
    print("=" * 60)
    print(f"\nOutput directory: {args.output.absolute()}")
    print("\nTo use with MemMachine evaluation:")
    print(f"  ./run_test.sh beam exp1 search retrieval_agent \\")
    print(f"    {args.output.absolute()}/100K/{{conversation_id}}/chat.json \\")
    print(f"    {args.output.absolute()}/100K/{{conversation_id}}/probing_questions/probing_questions.json")


if __name__ == "__main__":
    main()
