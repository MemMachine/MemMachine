import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.utils import agent_utils  # noqa: E402


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

        await memory.delete_session_episodes()
        await memory.close()

    tasks = [
        process_conversation(idx, item)
        for idx, item in enumerate(locomo_data)
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
