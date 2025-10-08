# test migration.py using locomo data
# 1. cd ~/MemMachine; ./memmachine-compose.sh  # start MemMachine
# 2. cd ~/MemMachine/tools/chatgpt2memmachine  # change to this dir
# 3. uv run python3 test_migration.py  # run test

import os

from migration import MigrationHack


def test_migration(dry_run=True):
    base_url = "http://localhost:8080"
    my_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(my_dir)
    chat_history = "../../evaluation/locomo/locomo10.json"
    chat_type = "locomo"
    start_time = 0
    max_messages = 0
    summarize = False
    summarize_every = 20
    migration_hack = MigrationHack(
        base_url=base_url,
        chat_history_file=chat_history,
        chat_type=chat_type,
        start_time=start_time,
        max_messages=max_messages,
        dry_run=dry_run,
    )

    migration_hack.migrate(summarize=summarize, summarize_every=summarize_every)
    print("== All completed successfully")


if __name__ == "__main__":
    test_migration(dry_run=False)
