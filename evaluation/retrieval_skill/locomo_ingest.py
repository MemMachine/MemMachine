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

from memmachine_server.common.episode_store import Episode  # noqa: E402

from evaluation.utils import skill_utils  # noqa: E402


def datetime_from_locomo_time(locomo_time_str: str) -> datetime:
    return datetime.strptime(locomo_time_str, "%I:%M %p on %d %B, %Y").replace(
        tzinfo=UTC
    )


async def main():  # noqa: C901
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")

    args = parser.parse_args()

    data_path = args.data_path

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

    async def process_conversation(idx, item):
        if "conversation" not in item:
            return None

        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        print(
            f"Processing conversation for group {idx} with speakers {speaker_a} and {speaker_b}..."
        )

        group_id = f"group_{idx}"
        warmup_query = ""
        for qa in item.get("qa", []):
            if str(qa.get("category")) == "5":
                continue
            question = str(qa.get("question", "")).strip()
            if question:
                warmup_query = question
                break

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

            await skill_utils.add_episodes_via_rest(
                session_id=group_id,
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
                ],
            )

        return group_id, warmup_query

    tasks = [process_conversation(idx, item) for idx, item in enumerate(locomo_data)]
    session_warmups = await asyncio.gather(*tasks)
    for session_warmup in session_warmups:
        if session_warmup is None:
            continue
        group_id, warmup_query = session_warmup
        if not warmup_query:
            continue
        elapsed = await skill_utils.warmup_rest_evaluation_search(
            session_id=group_id,
            query=warmup_query,
        )
        if elapsed is None:
            print(
                f"Warmup search for {group_id} did not finish before the retry budget was exhausted."
            )
        else:
            print(f"Warmed up search for {group_id} in {elapsed:.3f}s")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
