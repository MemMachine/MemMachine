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

from evaluation.utils import agent_utils  # noqa: E402
from memmachine.episodic_memory.declarative_memory import (  # noqa: E402
    ContentType,
    Episode,
)


def datetime_from_locomo_time(locomo_time_str: str) -> datetime:
    return datetime.strptime(locomo_time_str, "%I:%M %p on %d %B, %Y").replace(
        tzinfo=UTC
    )


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")

    args = parser.parse_args()

    data_path = args.data_path

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

    vector_graph_store = agent_utils.init_vector_graph_store(
        neo4j_uri="bolt://localhost:7687"
    )

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

        memory, _, _ = await agent_utils.init_memmachine_params(
            vector_graph_store=vector_graph_store,
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

            await memory.add_episodes(
                episodes=[
                    Episode(
                        uid=str(uuid4()),
                        timestamp=session_datetime
                        + message_index * timedelta(seconds=1),
                        source=message["speaker"],
                        content_type=ContentType.MESSAGE,
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
                        user_metadata={
                            "locomo_session_id": session_id,
                        },
                    )
                    for message_index, message in enumerate(session)
                ]
            )

    tasks = [process_conversation(idx, item) for idx, item in enumerate(locomo_data)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
