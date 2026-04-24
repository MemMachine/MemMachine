import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


async def main():
    from evaluation.utils import agent_utils

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to configuration.yml",
    )
    args = parser.parse_args()

    resource_manager = agent_utils.load_eval_config(args.config_path)
    memory, _, _ = await agent_utils.init_memmachine_params(
        resource_manager=resource_manager,
        session_id="group1",
    )

    print("Deleting episodes for session_id='group1'...")
    await memory.delete_session_episodes()
    print("Completed WikiMultiHop delete.")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
