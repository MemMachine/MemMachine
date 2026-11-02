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
    parser.add_argument(
        "--session-id",
        default="beam_default",
        help="Session ID for delete (default: beam_default)",
    )
    args = parser.parse_args()

    resource_manager = agent_utils.load_eval_config(args.config_path)

    print(f"Deleting episodes for session_id='{args.session_id}'...")
    await agent_utils.delete_session_episodes(resource_manager, args.session_id)
    print(f"Completed BEAM delete for session '{args.session_id}'.")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
