import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


async def main():
    from evaluation.utils import agent_utils

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to the data file")
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to configuration.yml",
    )
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        locomo_data = json.load(f)

    resource_manager = agent_utils.load_eval_config(args.config_path)

    async def process_conversation(idx, item):
        if "conversation" not in item:
            return

        group_id = f"group_{idx}"
        print(f"Deleting episodes for group {group_id}...")

        memory, _, _ = await agent_utils.init_memmachine_params(
            resource_manager=resource_manager,
            session_id=group_id,
        )
        await memory.delete_session_episodes()

    tasks = [process_conversation(idx, item) for idx, item in enumerate(locomo_data)]
    await asyncio.gather(*tasks)
    print(f"Completed LoCoMo delete for {len(locomo_data)} groups.")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
