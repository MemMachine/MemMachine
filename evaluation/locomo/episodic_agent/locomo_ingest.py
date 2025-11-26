import argparse
import asyncio
import json
from collections import deque
from datetime import datetime
from typing import cast

from dotenv import load_dotenv

from memmachine.episodic_memory.data_types import ContentType
from memmachine.episodic_memory.episodic_memory import EpisodicMemory
from memmachine.episodic_memory.episodic_memory_manager import (
    EpisodicMemoryManager,
)


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")

    args = parser.parse_args()

    data_path = args.data_path

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

    memory_manager = EpisodicMemoryManager.create_episodic_memory_manager(
        "locomo_config.yaml"
    )

    async def process_conversation(idx, item, memory_manager: EpisodicMemoryManager):
        if "conversation" not in item:
            return

        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        print(
            f"Processing conversation for group {idx} with speakers {speaker_a} and {speaker_b}..."
        )

        group_id = f"group_{idx}"

        memory = cast(
            EpisodicMemory,
            await memory_manager.get_episodic_memory_instance(
                group_id=group_id,
                session_id=group_id,
                user_id=[speaker_a, speaker_b],
            ),
        )

        session_idx = 0
        while True:
            session_idx += 1
            session_id = f"session_{session_idx}"

            if session_id not in conversation:
                break

            session = conversation[session_id]
            session_date_time = conversation[f"{session_id}_date_time"]

            context_messages: deque[str] = deque(maxlen=5)
            for message in session:
                speaker = message["speaker"]
                blip_caption = message.get("blip_caption")
                message_text = message["text"]

                context_messages.append(
                    f"[{session_date_time}] {speaker}: {message_text}"
                )

                await memory.add_memory_episode(
                    producer=speaker,
                    produced_for=speaker,
                    episode_content=message_text,
                    episode_type="default",
                    content_type=ContentType.STRING,
                    timestamp=datetime.now(),
                    metadata={
                        "source_timestamp": session_date_time,
                        "source_speaker": speaker,
                        "blip_caption": blip_caption,
                    },
                )

        await memory.close()

    tasks = [
        process_conversation(idx, item, memory_manager)
        for idx, item in enumerate(locomo_data)
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
