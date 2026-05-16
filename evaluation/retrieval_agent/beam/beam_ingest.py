import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from memmachine_server.common.episode_store import Episode  # noqa: E402

from evaluation.utils import agent_utils  # noqa: E402


def load_beam_chat_data(data_path: str) -> list[dict]:
    """Load BEAM chat data from chat.json file.

    Supports both 100K/500K/1M format (nested with plan-X keys)
    and 10M format (flat batch structure).

    Args:
        data_path: Path to the chat.json file.

    Returns:
        List of chat batches, each containing turns with messages.
    """
    print(f"Loading BEAM chat data from {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Detect format: 100K/500K/1M has nested "plan-X" keys
    # 10M has flat batch structure
    chat_data = []
    if raw_data and isinstance(raw_data, list) and len(raw_data) > 0:
        first_item = raw_data[0]
        if isinstance(first_item, dict):
            # Check if first item has "plan-X" key (100K/500K/1M format)
            has_plan_key = any(key.startswith("plan-") for key in first_item)
            if has_plan_key:
                # 100K/500K/1M format: flatten nested structure
                print("Detected 100K/500K/1M format (nested plan-X keys)")
                for item in raw_data:
                    for plan_key, batches in item.items():
                        if plan_key.startswith("plan-"):
                            chat_data.extend(batches)
            else:
                # 10M format: use as-is
                print("Detected 10M format (flat batch structure)")
                chat_data = raw_data

    print(f"Loaded {len(chat_data)} batches from BEAM dataset")
    return chat_data


async def beam_ingest(
    data_path: str,
    config_path: str,
    session_id: str,
) -> dict:
    """Ingest BEAM chat data into MemMachine episodic memory.

    Args:
        data_path: Path to the chat.json file.
        config_path: Path to configuration.yml.
        session_id: Session ID for this ingestion run.

    Returns:
        Dictionary containing ingestion statistics.
    """
    resource_manager = agent_utils.load_eval_config(config_path)
    memory, _, _, _, _ = await agent_utils.init_memmachine_params(
        resource_manager=resource_manager,
        session_id=session_id,
    )

    chat_data = load_beam_chat_data(data_path)

    t1 = datetime.now(tz=UTC)
    total_messages = 0
    total_batches = 0
    episodes = []

    for batch in chat_data:
        batch_num = batch.get("batch_number", 0)
        turns = batch.get("turns", [])
        total_batches += 1

        for turn_idx, turn in enumerate(turns):
            for msg in turn:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                msg_id = msg.get("id", None)
                time_anchor = msg.get("time_anchor", None)
                question_type = msg.get("question_type", None)
                index = msg.get("index", None)

                if not content:
                    continue

                metadata = {
                    "batch_number": str(batch_num),
                    "turn_index": str(turn_idx),
                    "message_id": str(msg_id) if msg_id else "",
                    "time_anchor": str(time_anchor) if time_anchor else "",
                    "question_type": str(question_type) if question_type else "",
                    "index": str(index) if index else "",
                    "source": "beam_chat",
                }

                ts = t1 + timedelta(seconds=total_messages + 1)
                episodes.append(
                    Episode(
                        uid=str(uuid4()),
                        content=content,
                        session_key=session_id,
                        created_at=ts,
                        producer_id=f"beam_{role}",
                        producer_role=role,
                        metadata=metadata,
                    )
                )
                total_messages += 1

    # Ingest all episodes at once
    if episodes:
        t = time.perf_counter()
        await memory.add_memory_episodes(episodes=episodes)
        elapsed = time.perf_counter() - t
        print(
            f"Ingested {total_messages} messages from {total_batches} batches in {elapsed:.3f}s"
        )

    stats = {
        "session_id": session_id,
        "total_batches": total_batches,
        "total_messages": total_messages,
        "data_path": data_path,
    }

    print(
        f"\nBEAM ingestion completed: {total_messages} messages from {total_batches} batches"
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for BEAM ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest BEAM benchmark chat data into MemMachine"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to chat.json file",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to configuration.yml",
    )
    parser.add_argument(
        "--session-id",
        default="beam_default",
        help="Session ID for this ingestion (default: beam_default)",
    )
    return parser


async def main():
    """Main entry point for BEAM ingestion."""
    args = build_parser().parse_args()

    print("Starting BEAM benchmark ingestion...")
    print(f"Data path: {args.data_path}")
    print(f"Config path: {args.config_path}")
    print(f"Session ID: {args.session_id}")

    stats = await beam_ingest(
        data_path=args.data_path,
        config_path=args.config_path,
        session_id=args.session_id,
    )

    print("\nIngestion statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
