import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from memmachine_server.common.episode_store.episode_model import Episode

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.utils import agent_utils  # noqa: E402


def datetime_from_locomo_time(locomo_time_str: str) -> datetime:
    return datetime.strptime(locomo_time_str, "%I:%M %p on %d %B, %Y").replace(
        tzinfo=UTC
    )


async def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")
    parser.add_argument(
        "--config-path",
        default="locomo_config.yaml",
        help="Path to configuration.yml",
    )

    args = parser.parse_args()

    data_path = args.data_path

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

    resource_manager = agent_utils.load_eval_config(args.config_path)

    async def process_conversation(
        idx,
        item,
    ) -> None:
        if "conversation" not in item:
            return

        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        print(
            f"Processing conversation for group {idx} with speakers {speaker_a} and {speaker_b}...",
        )

        group_id = f"group_{idx}"

        memory, _, _ = await agent_utils.init_memmachine_params(
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
            session_date_time = conversation[f"{session_id}_date_time"]
            session_datetime = datetime_from_locomo_time(session_date_time)

            for message_index, message in enumerate(session):
                speaker = message["speaker"]
                blip_caption = message.get("blip_caption")
                message_text = message["text"]

                await memory.add_memory_episodes(
                    episodes=[
                        Episode(
                            uid=str(uuid4()),
                            content=message_text,
                            session_key=group_id,
                            created_at=session_datetime
                            + message_index * timedelta(seconds=1),
                            producer_id=speaker,
                            producer_role=speaker,
                            metadata={
                                "source_timestamp": session_date_time,
                                "source_speaker": speaker,
                                "blip_caption": blip_caption,
                            },
                        )
                    ],
                )

        await memory.close()

    tasks = [process_conversation(idx, item) for idx, item in enumerate(locomo_data)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
