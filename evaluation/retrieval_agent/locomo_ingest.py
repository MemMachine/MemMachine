import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.retrieval_agent.cli_utils import positive_int  # noqa: E402

DEFAULT_CONCURRENCY = 10


def datetime_from_locomo_time(locomo_time_str: str) -> datetime:
    return datetime.strptime(locomo_time_str, "%I:%M %p on %d %B, %Y").replace(
        tzinfo=UTC
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to the data file")
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to configuration.yml",
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        default=DEFAULT_CONCURRENCY,
        help="Maximum number of concurrent LoCoMo ingestion tasks",
    )
    return parser


async def main():
    from memmachine_server.common.episode_store import Episode
    from memmachine_server.common.utils import async_with

    from evaluation.utils import agent_utils

    args = build_parser().parse_args()

    data_path = args.data_path

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

    resource_manager = agent_utils.load_eval_config(args.config_path)

    async def process_conversation(idx, item):
        if "conversation" not in item:
            return

        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        print(
            f"Processing conversation for group {idx} with speakers {speaker_a} and {speaker_b}..."
        )

        group_id = f"group_{idx}"

        memory, _, _, _, _ = await agent_utils.init_memmachine_params(
            resource_manager=resource_manager,
            session_id=group_id,
        )

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

            await memory.add_memory_episodes(
                episodes=[
                    Episode(
                        uid=str(uuid4()),
                        content=message["text"]
                        + (
                            f" [Attached {blip_caption}: {image_query}]"
                            if (
                                (
                                    (
                                        (blip_caption := message.get("blip_caption"))
                                        or True
                                    )
                                    and ((image_query := message.get("query")) or True)
                                )
                                and blip_caption
                                and image_query
                            )
                            else (
                                f" [Attached {blip_caption}]"
                                if blip_caption
                                else (
                                    f" [Attached a photo: {image_query}]"
                                    if image_query
                                    else ""
                                )
                            )
                        ),
                        session_key=group_id,
                        created_at=session_datetime
                        + message_index * timedelta(seconds=1),
                        producer_id=message["speaker"],
                        producer_role=message["speaker"],
                        metadata={
                            "locomo_session_id": session_id,
                        },
                    )
                    for message_index, message in enumerate(session)
                ]
            )

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        async_with(semaphore, process_conversation(idx, item))
        for idx, item in enumerate(locomo_data)
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
